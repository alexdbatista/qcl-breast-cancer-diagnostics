"""
04_model_training.py
====================
Training Pipeline for the Hyperspectral U-Net Spatial Segmentation Model.

Implements:
  - Custom PyTorch Dataset for Spero® QCL .mat hyperspectral cubes
  - Patch-based sampling to handle multi-GB TMA cores efficiently
  - Multi-class segmentation loss: Weighted CrossEntropy + Soft Dice
  - Training loop with validation, early stopping, and checkpoint saving
  - Per-class evaluation: IoU (Intersection over Union) and Dice coefficient

Histological target classes (4):
  0 — Background / Artefact
  1 — Benign Epithelium
  2 — Benign Stroma
  3 — Malignant Stroma

Usage:
    python 04_model_training.py
"""

import os
import time
import json
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List

# Local architecture import
from spatial_cnn_segmentation import HyperspectralUNet  # noqa: F401


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR      = Path("./data/processed")   # PCA-reduced .npy cubes
LABEL_DIR     = Path("./data/labels")      # Ground-truth annotation masks
MODELS_DIR    = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE    = 64       # Pixels — avoids VRAM exhaustion on large tissue cores
BATCH_SIZE    = 8
NUM_CLASSES   = 4
BANDS         = 10       # PCA components retained in Module 02
EPOCHS        = 50
LR            = 1e-4
PATIENCE      = 10       # Early stopping patience (epochs without improvement)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Background", "Benign Epithelium", "Benign Stroma", "Malignant Stroma"]

print(f"[INFO] Training on device: {DEVICE}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HyperspectralTMADataset(Dataset):
    """
    PyTorch Dataset for patch-based loading of PCA-reduced QCL hyperspectral cubes.

    Each tissue core is stored as a `.npy` array of shape (H, W, Bands).
    The corresponding annotation mask is stored as (H, W) uint8 with integer
    class labels.

    Rather than loading full gigabyte cubes into VRAM, this Dataset generates
    random (Patch_Size × Patch_Size) crops at each iteration, providing
    virtually unlimited augmented training samples from a limited number of
    annotated cores.

    Parameters
    ----------
    cube_dir : Path
        Directory containing PCA-reduced `.npy` hyperspectral cubes.
    label_dir : Path
        Directory containing matching `.npy` integer annotation masks.
    patch_size : int
        Height and width (in pixels) of each randomly sampled crop.
    n_patches_per_core : int
        Number of random patches to extract from each core per epoch.
    augment : bool
        If True, applies random horizontal/vertical flips during training.
    """

    def __init__(
        self,
        cube_dir: Path,
        label_dir: Path,
        patch_size: int = 64,
        n_patches_per_core: int = 50,
        augment: bool = True,
    ):
        self.cube_paths  = sorted(cube_dir.glob("*.npy"))
        self.label_dir   = label_dir
        self.patch_size  = patch_size
        self.n_patches   = n_patches_per_core
        self.augment     = augment

        if not self.cube_paths:
            raise FileNotFoundError(
                f"No .npy cubes found in {cube_dir}. "
                "Run 02_spectral_dimensionality_reduction.ipynb first."
            )
        print(f"[INFO] Dataset loaded: {len(self.cube_paths)} tissue cores.")

    def __len__(self) -> int:
        return len(self.cube_paths) * self.n_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a single (cube_patch, label_patch) pair.

        Parameters
        ----------
        idx : int
            Linear index into the dataset (stride across cores × patches).

        Returns
        -------
        cube_patch : torch.Tensor
            Shape (Bands, Patch_H, Patch_W), dtype float32.
        label_patch : torch.Tensor
            Shape (Patch_H, Patch_W), dtype int64 (class indices).
        """
        core_idx  = idx // self.n_patches
        cube_path = self.cube_paths[core_idx]
        label_path = self.label_dir / cube_path.name

        cube  = np.load(str(cube_path)).astype(np.float32)     # (H, W, Bands)
        label = np.load(str(label_path)).astype(np.int64)       # (H, W)

        H, W, _ = cube.shape

        # Random crop
        max_y = H - self.patch_size
        max_x = W - self.patch_size
        y0 = random.randint(0, max(0, max_y))
        x0 = random.randint(0, max(0, max_x))
        y1, x1 = y0 + self.patch_size, x0 + self.patch_size

        cube_patch  = cube[y0:y1, x0:x1, :]     # (P, P, Bands)
        label_patch = label[y0:y1, x0:x1]        # (P, P)

        # Convert to (Bands, H, W) for PyTorch channels-first convention
        cube_patch = torch.from_numpy(cube_patch.transpose(2, 0, 1))

        if self.augment:
            if random.random() > 0.5:
                cube_patch  = torch.flip(cube_patch, dims=[2])
                label_patch = np.fliplr(label_patch).copy()
            if random.random() > 0.5:
                cube_patch  = torch.flip(cube_patch, dims=[1])
                label_patch = np.flipud(label_patch).copy()

        label_patch = torch.from_numpy(label_patch)
        return cube_patch, label_patch


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class CombinedSegmentationLoss(nn.Module):
    """
    Weighted CrossEntropy + Soft Dice Loss for imbalanced tissue segmentation.

    In breast TMA data, Malignant Stroma pixels are typically outnumbered
    by Benign Stroma and Background by a wide margin. A pure CrossEntropy
    loss will bias the model toward the majority class.

    The Dice component directly optimizes the F1-score per class, providing
    gradient signal even when malignant pixels are sparse in a batch.

    Parameters
    ----------
    class_weights : torch.Tensor or None
        Optional tensor of shape (num_classes,) for CrossEntropy weighting.
    dice_weight : float
        Relative weight of the Dice loss term (default 0.5).
    ce_weight : float
        Relative weight of the CrossEntropy loss term (default 0.5).
    smooth : float
        Laplace smoothing constant for the Dice denominator.
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.smooth      = smooth

    def soft_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-class Soft Dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output, shape (B, C, H, W).
        targets : torch.Tensor
            Integer class labels, shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Mean Dice loss averaged across all classes (scalar).
        """
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        dice_total = 0.0

        for c in range(num_classes):
            target_c = (targets == c).float()
            prob_c   = probs[:, c, :, :]
            intersection = (prob_c * target_c).sum()
            dice_total  += 1.0 - (2.0 * intersection + self.smooth) / (
                prob_c.sum() + target_c.sum() + self.smooth
            )
        return dice_total / num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.soft_dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_iou_dice(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Dict[str, List[float]]:
    """
    Compute per-class IoU and Dice coefficient for a batch.

    Parameters
    ----------
    preds : torch.Tensor
        Integer predicted classes, shape (B, H, W).
    targets : torch.Tensor
        Integer ground-truth classes, shape (B, H, W).
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    dict
        Keys: 'iou', 'dice' — each a list of floats, one per class.
    """
    iou_per_class  = []
    dice_per_class = []

    for c in range(num_classes):
        pred_c   = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().float()
        union        = (pred_c | target_c).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred_c.sum() + target_c.sum() + 1e-6)

        iou_per_class.append(iou.item())
        dice_per_class.append(dice.item())

    return {"iou": iou_per_class, "dice": dice_per_class}


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one complete training epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : optim.Optimizer
    criterion : nn.Module
    device : torch.device

    Returns
    -------
    float
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for cubes, labels in loader:
        cubes  = cubes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(cubes)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """
    Evaluate the model on the validation split.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
    device : torch.device

    Returns
    -------
    val_loss : float
        Mean validation loss.
    metrics : dict
        Per-class mean IoU and Dice across all validation batches.
    """
    model.eval()
    total_loss = 0.0
    all_iou    = [[] for _ in range(NUM_CLASSES)]
    all_dice   = [[] for _ in range(NUM_CLASSES)]

    for cubes, labels in loader:
        cubes  = cubes.to(device)
        labels = labels.to(device)

        logits = model(cubes)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        m = compute_iou_dice(preds, labels, NUM_CLASSES)
        for c in range(NUM_CLASSES):
            all_iou[c].append(m["iou"][c])
            all_dice[c].append(m["dice"][c])

    mean_iou  = [float(np.mean(all_iou[c]))  for c in range(NUM_CLASSES)]
    mean_dice = [float(np.mean(all_dice[c])) for c in range(NUM_CLASSES)]

    return total_loss / len(loader), {"iou": mean_iou, "dice": mean_dice}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Datasets & Loaders --
    print("\n[INFO] Building datasets...")
    train_ds = HyperspectralTMADataset(DATA_DIR / "train", LABEL_DIR / "train",
                                       patch_size=PATCH_SIZE, augment=True)
    val_ds   = HyperspectralTMADataset(DATA_DIR / "val",   LABEL_DIR / "val",
                                       patch_size=PATCH_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # -- Model --
    print("[INFO] Initializing Hyperspectral U-Net...")
    model = HyperspectralUNet(in_channels=BANDS, num_classes=NUM_CLASSES).to(DEVICE)

    # Class-weighted loss — upweight Malignant Stroma (class 3)
    class_weights = torch.tensor([0.5, 1.0, 1.0, 2.5], device=DEVICE)
    criterion  = CombinedSegmentationLoss(class_weights=class_weights)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # -- Training --
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    print(f"\n[INFO] Starting training for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        t0  = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, metrics = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        mean_iou  = float(np.mean(metrics["iou"]))
        mean_dice = float(np.mean(metrics["dice"]))
        malignant_iou  = metrics["iou"][3]
        malignant_dice = metrics["dice"][3]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"mIoU: {mean_iou:.3f} | mDice: {mean_dice:.3f} | "
            f"Malignant IoU: {malignant_iou:.3f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "malignant_iou": malignant_iou,
            "malignant_dice": malignant_dice,
        })

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            checkpoint_path = MODELS_DIR / "best_unet_qcl.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics,
                "config": {"bands": BANDS, "num_classes": NUM_CLASSES, "patch_size": PATCH_SIZE},
            }, checkpoint_path)
            print(f"  ✅ New best model saved → {checkpoint_path}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[DONE] Training history saved → {history_path}")

    # Final class-level summary
    print("\n--- Final Validation Metrics (Best Model) ---")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name}: IoU={metrics['iou'][c]:.3f} | Dice={metrics['dice'][c]:.3f}")
