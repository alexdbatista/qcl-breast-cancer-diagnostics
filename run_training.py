"""
run_training.py
================
Self-contained training runner — adapts 04_model_training.py to actual
data layout (flat processed/ and labels/ dirs, no pre-split subdirs).

Workflow:
  1. Discovers all PCA cubes and label masks
  2. Splits 80/20 train/val (chronological, no shuffle to avoid leakage)
  3. Runs the Hyperspectral U-Net training loop
  4. Saves best checkpoint + training history

Quick-test mode (default): 5 epochs, small patches, fast iteration on CPU.
Full mode: set QUICK_TEST = False for a full 50-epoch run.
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List

# ---------------------------------------------------------------------------
# Import UNet from local module (rename import to match actual filename)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# Dynamically import from the numbered module file
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "seg_module",
    Path(__file__).parent / "03_spatial_cnn_segmentation.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HyperspectralUNet = _mod.HyperspectralUNet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUICK_TEST    = False    # Set False for full 50-epoch run
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LABELS_DIR    = BASE_DIR / "data" / "labels"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE    = 32  if QUICK_TEST else 64
BATCH_SIZE    = 4   if QUICK_TEST else 8
N_PATCHES_TR  = 20  if QUICK_TEST else 50
EPOCHS        = 5   if QUICK_TEST else 50
NUM_CLASSES   = 4
BANDS         = 15       # Must match PCA components from run_pipeline.py
LR            = 1e-4
PATIENCE      = 3   if QUICK_TEST else 10
DEVICE        = torch.device("mps")  if torch.backends.mps.is_available() else \
                torch.device("cuda") if torch.cuda.is_available() else \
                torch.device("cpu")

CLASS_NAMES = ["Background", "Benign Epithelium", "Benign Stroma", "Malignant Stroma"]

print(f"[INFO] Device       : {DEVICE}")
print(f"[INFO] Quick Test   : {QUICK_TEST}")
print(f"[INFO] Patch size   : {PATCH_SIZE}x{PATCH_SIZE}")
print(f"[INFO] Epochs       : {EPOCHS}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HyperspectralPatchDataset(Dataset):
    """
    Patch-based Dataset matching the actual flat processed/ and labels/ layout.

    Parameters
    ----------
    cube_paths : list of Path
        PCA-reduced .npy cube files (H, W, Bands).
    label_paths : list of Path
        Matching pixel-level cluster label masks (H, W) uint8.
    patch_size : int
        Spatial crop size in pixels.
    n_patches : int
        Number of random patches to yield per tile per epoch.
    augment : bool
        Apply random flips during training.
    """

    def __init__(
        self,
        cube_paths: List[Path],
        label_paths: List[Path],
        patch_size: int = 64,
        n_patches: int = 50,
        augment: bool = True,
    ):
        self.cube_paths  = cube_paths
        self.label_paths = label_paths
        self.patch_size  = patch_size
        self.n_patches   = n_patches
        self.augment     = augment
        print(f"  Dataset: {len(cube_paths)} tiles x {n_patches} patches = "
              f"{len(cube_paths)*n_patches} samples/epoch")

    def __len__(self) -> int:
        return len(self.cube_paths) * self.n_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_idx   = idx // self.n_patches
        cube       = np.load(str(self.cube_paths[tile_idx])).astype(np.float32)
        label      = np.load(str(self.label_paths[tile_idx])).astype(np.int64)
        H, W, B    = cube.shape
        P          = self.patch_size
        y0 = random.randint(0, max(0, H - P))
        x0 = random.randint(0, max(0, W - P))
        patch_cube  = cube[y0:y0+P, x0:x0+P, :]       # (P, P, B)
        patch_label = label[y0:y0+P, x0:x0+P]          # (P, P)

        # (B, P, P)  channels-first for PyTorch
        t_cube  = torch.from_numpy(patch_cube.transpose(2, 0, 1))
        t_label = torch.from_numpy(patch_label.copy())

        if self.augment:
            if random.random() > 0.5:
                t_cube  = torch.flip(t_cube,  dims=[2])
                t_label = torch.flip(t_label, dims=[1])
            if random.random() > 0.5:
                t_cube  = torch.flip(t_cube,  dims=[1])
                t_label = torch.flip(t_label, dims=[0])

        return t_cube, t_label


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedSegmentationLoss(nn.Module):
    """Weighted Cross-Entropy + Soft Dice — upweights Malignant Stroma class."""

    def __init__(self, class_weights, smooth=1.0):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth = smooth

    def soft_dice(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        n_cls = logits.shape[1]
        dice  = 0.0
        for c in range(n_cls):
            target_c = (targets == c).float()
            prob_c   = probs[:, c]
            inter    = (prob_c * target_c).sum()
            dice    += 1.0 - (2 * inter + self.smooth) / (
                prob_c.sum() + target_c.sum() + self.smooth)
        return dice / n_cls

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.soft_dice(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, targets, n_cls):
    iou_list, dice_list = [], []
    for c in range(n_cls):
        pc = (preds == c); tc = (targets == c)
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        iou_list.append(((inter + 1e-6) / (union + 1e-6)).cpu().item())
        denom = (pc.sum() + tc.sum()).float()
        dice_list.append(((2 * inter + 1e-6) / (denom + 1e-6)).cpu().item())
    return iou_list, dice_list


# ---------------------------------------------------------------------------
# Training & Validation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for cubes, labels in loader:
        cubes, labels = cubes.to(device), labels.to(device)
        opt.zero_grad()
        loss = criterion(model(cubes), labels)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, n_cls):
    model.eval()
    total = 0.0
    all_iou  = [[] for _ in range(n_cls)]
    all_dice = [[] for _ in range(n_cls)]
    for cubes, labels in loader:
        cubes, labels = cubes.to(device), labels.to(device)
        logits = model(cubes)
        total += criterion(logits, labels).item()
        preds  = logits.argmax(dim=1)
        iou, dice = compute_metrics(preds, labels, n_cls)
        for c in range(n_cls):
            all_iou[c].append(iou[c])
            all_dice[c].append(dice[c])
    mean_iou  = [float(np.mean(all_iou[c]))  for c in range(n_cls)]
    mean_dice = [float(np.mean(all_dice[c])) for c in range(n_cls)]
    return total / len(loader), mean_iou, mean_dice


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Discover data files --
    cube_files  = sorted(PROCESSED_DIR.glob("*_pca.npy"))
    label_files = sorted(LABELS_DIR.glob("*_labels.npy"))
    assert len(cube_files) == len(label_files), \
        f"Mismatch: {len(cube_files)} cubes vs {len(label_files)} labels"
    print(f"\n[INFO] Found {len(cube_files)} tile pairs.")

    # -- 80/20 chronological split --
    split = int(len(cube_files) * 0.8)
    train_cubes,  val_cubes  = cube_files[:split],  cube_files[split:]
    train_labels, val_labels = label_files[:split], label_files[split:]
    print(f"[INFO] Train: {len(train_cubes)} tiles | Val: {len(val_cubes)} tiles")

    # -- Datasets --
    print("\n[INFO] Building datasets...")
    train_ds = HyperspectralPatchDataset(
        train_cubes, train_labels,
        patch_size=PATCH_SIZE, n_patches=N_PATCHES_TR, augment=True)
    val_ds = HyperspectralPatchDataset(
        val_cubes, val_labels,
        patch_size=PATCH_SIZE, n_patches=10, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # -- Model --
    print(f"\n[INFO] Initializing HyperspectralUNet (bands={BANDS}, classes={NUM_CLASSES})...")
    model     = HyperspectralUNet(in_channels=BANDS, num_classes=NUM_CLASSES).to(DEVICE)
    weights   = torch.tensor([0.5, 1.0, 1.0, 2.5], device=DEVICE)
    criterion = CombinedSegmentationLoss(class_weights=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # -- Training loop --
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    print(f"\n{'='*60}")
    print(f"  Training Started — {EPOCHS} epochs on {DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES)
        scheduler.step()
        elapsed = time.time() - t0

        m_iou  = np.mean(val_iou)
        m_dice = np.mean(val_dice)
        mal_iou = val_iou[3]

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: {tr_loss:.4f} | Val: {val_loss:.4f} | "
              f"mIoU: {m_iou:.3f} | mDice: {m_dice:.3f} | "
              f"Malignant IoU: {mal_iou:.3f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": val_loss, "mean_iou": m_iou,
                         "mean_dice": m_dice, "malignant_iou": mal_iou})

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            ckpt_path = MODELS_DIR / "best_unet_qcl.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "mean_iou": m_iou,
                        "config": {"bands": BANDS, "num_classes": NUM_CLASSES}},
                       ckpt_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch}.")
                break

    # Save history
    hist_path = MODELS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Training Complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {MODELS_DIR / 'best_unet_qcl.pth'}")
    print(f"  History       : {hist_path}")
    print(f"\n  Final Per-Class Metrics:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name}: IoU={val_iou[c]:.3f} | Dice={val_dice[c]:.3f}")
    print(f"{'='*60}")
