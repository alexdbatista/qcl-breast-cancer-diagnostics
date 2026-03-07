"""
06_inference_and_evaluation.py
================================
Post-training evaluation, visual results, and portfolio reporting.

Run this after training completes (models/best_unet_qcl.pth must exist).

Generates:
  - Training curves (loss, mIoU, mDice, Malignant IoU per epoch)
  - Full-tile inference: predicted segmentation map vs. K-Means pseudo-labels
  - Per-class Confusion Matrix (normalised)
  - Per-class IoU / Dice summary table
  - Final portfolio-ready composite figure
"""

import sys
import json
import importlib.util as _ilu
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Setup & Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LABELS_DIR    = BASE_DIR / "data" / "labels"
MODELS_DIR    = BASE_DIR / "models"
FIGURES_DIR   = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODELS_DIR / "best_unet_qcl.pth"
HISTORY    = MODELS_DIR / "training_history.json"

NUM_CLASSES = 4
BANDS       = 15
CLASS_NAMES = ["Background", "Benign\nEpithelium", "Benign\nStroma", "Malignant\nStroma"]
CLASS_SHORT = ["BG", "Ben. Epi", "Ben. Str", "Mal. Str"]
CLASS_COLOURS = ["#1e3a5f", "#22c55e", "#3b82f6", "#ef4444"]

DEVICE = torch.device("mps")  if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")

print(f"[INFO] Evaluation device: {DEVICE}")

# Load UNet architecture dynamically
_spec = _ilu.spec_from_file_location(
    "seg_module", BASE_DIR / "03_spatial_cnn_segmentation.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HyperspectralUNet = _mod.HyperspectralUNet

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
if not CHECKPOINT.exists():
    print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
    print("        Run run_training.py first.")
    sys.exit(1)

if not HISTORY.exists():
    print(f"[WARNING] training_history.json not found — skipping training curves.")

# ---------------------------------------------------------------------------
# FIGURE 1: Training Curves
# ---------------------------------------------------------------------------
if HISTORY.exists():
    print("\n[STEP 1] Plotting training curves...")
    with open(HISTORY) as f:
        history = json.load(f)

    epochs      = [h["epoch"] for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"] for h in history]
    mean_iou    = [h["mean_iou"] for h in history]
    mean_dice   = [h["mean_dice"] for h in history]
    mal_iou     = [h["malignant_iou"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Hyperspectral U-Net — Training Diagnostics", fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_loss, color="#3b82f6", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, val_loss,   color="#ef4444", linewidth=2, label="Val Loss")
    best_ep = epochs[int(np.argmin(val_loss))]
    axes[0].axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch {best_ep}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Combined Loss (CE + Dice)")
    axes[0].set_title("Loss Curves"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # mIoU & mDice
    axes[1].plot(epochs, mean_iou,  color="#22c55e", linewidth=2, label="Mean IoU")
    axes[1].plot(epochs, mean_dice, color="#f59e0b", linewidth=2, label="Mean Dice")
    axes[1].set_ylim(0, 1); axes[1].set_xlabel("Epoch")
    axes[1].set_title("Mean IoU & Dice (All Classes)"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Malignant Stroma IoU — the clinical metric
    axes[2].plot(epochs, mal_iou, color="#ef4444", linewidth=2.5, label="Malignant Stroma IoU")
    axes[2].fill_between(epochs, mal_iou, alpha=0.15, color="#ef4444")
    axes[2].axhline(0.90, color="gray", linestyle=":", alpha=0.6, label="Target >0.90")
    axes[2].set_ylim(0, 1); axes[2].set_xlabel("Epoch")
    axes[2].set_title("Malignant Stroma IoU\n(Primary Clinical Metric)"); axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "06_training_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Training curves → {fig_path}")

    best_idx = int(np.argmin(val_loss))
    print(f"\n  Best Epoch    : {best_ep}")
    print(f"  Best Val Loss : {val_loss[best_idx]:.4f}")
    print(f"  Best mIoU     : {mean_iou[best_idx]:.3f}")
    print(f"  Best mDice    : {mean_dice[best_idx]:.3f}")
    print(f"  Best Mal. IoU : {mal_iou[best_idx]:.3f}")


# ---------------------------------------------------------------------------
# STEP 2: Load model & run full-tile inference
# ---------------------------------------------------------------------------
print("\n[STEP 2] Loading best checkpoint and running inference...")
ckpt  = torch.load(str(CHECKPOINT), map_location=DEVICE)
model = HyperspectralUNet(in_channels=BANDS, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  Loaded checkpoint from epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")

# Use validation tiles (files 17-20 = last 20% of the 20 processed)
val_cubes  = sorted(PROCESSED_DIR.glob("*_pca.npy"))[-4:]
val_labels = sorted(LABELS_DIR.glob("*_labels.npy"))[-4:]

if not val_cubes:
    print("[ERROR] No processed .npy cubes found. Run run_pipeline.py first.")
    sys.exit(1)

# Inference on one full tile via overlapping patches
def infer_full_tile(model, cube_path, device, n_cls, patch_size=64, stride=32):
    """
    Sliding-window inference to produce a full-resolution segmentation map.

    Parameters
    ----------
    model : nn.Module
    cube_path : Path
    device : torch.device
    n_cls : int
    patch_size : int
    stride : int

    Returns
    -------
    pred_map : np.ndarray (H, W) int64
    prob_map : np.ndarray (H, W, n_cls) float32
    """
    cube = np.load(str(cube_path)).astype(np.float32)   # (H, W, B)
    H, W, B = cube.shape

    accum  = np.zeros((n_cls, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = cube[y:y+patch_size, x:x+patch_size, :]   # (P, P, B)
                t = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
                logits = model(t)                                   # (1, C, P, P)
                probs  = torch.softmax(logits, dim=1).squeeze(0)   # (C, P, P)
                accum[:, y:y+patch_size, x:x+patch_size] += probs.cpu().numpy()
                counts[y:y+patch_size, x:x+patch_size]   += 1.0

    # Avoid division by zero at borders
    counts = np.maximum(counts, 1.0)
    prob_map = (accum / counts[np.newaxis]).transpose(1, 2, 0)     # (H, W, C)
    pred_map = prob_map.argmax(axis=2).astype(np.int64)
    return pred_map, prob_map


print("  Running sliding-window inference on validation tile...")
val_cube_path  = val_cubes[0]
val_label_path = val_labels[0]

pred_map, prob_map = infer_full_tile(model, val_cube_path, DEVICE, NUM_CLASSES)
true_labels = np.load(str(val_label_path))
print(f"  Done. Pred map shape: {pred_map.shape}")


# ---------------------------------------------------------------------------
# FIGURE 2: Segmentation Results — Side-by-Side
# ---------------------------------------------------------------------------
print("\n[STEP 3] Generating segmentation visual comparison...")

cube = np.load(str(val_cube_path))
def norm01(a): return (a - a.min()) / (a.max() - a.min() + 1e-8)
rgb  = np.stack([norm01(cube[:,:,0]), norm01(cube[:,:,1]), norm01(cube[:,:,2])], axis=2)

cmap = ListedColormap(CLASS_COLOURS)
patches = [mpatches.Patch(color=CLASS_COLOURS[i], label=CLASS_NAMES[i].replace("\n", " "))
           for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"U-Net Inference — Tile {val_cube_path.stem.replace('_pca','')}\n"
             f"Best Checkpoint (Epoch {ckpt['epoch']}, Val Loss={ckpt['val_loss']:.4f})",
             fontsize=13, fontweight="bold")

axes[0].imshow(rgb)
axes[0].set_title("Input: PCA False-Colour\n(PC1=R, PC2=G, PC3=B)", fontsize=10)
axes[0].axis("off")

axes[1].imshow(true_labels, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
axes[1].set_title("Ground Truth\n(K-Means Pseudo-Labels)", fontsize=10)
axes[1].axis("off")

axes[2].imshow(pred_map, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
axes[2].set_title("U-Net Prediction\n(Sliding-Window Inference)", fontsize=10)
axes[2].axis("off")

fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
fig_path = FIGURES_DIR / "07_segmentation_results.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Segmentation comparison → {fig_path}")


# ---------------------------------------------------------------------------
# FIGURE 3: Confusion Matrix
# ---------------------------------------------------------------------------
print("\n[STEP 4] Computing confusion matrix...")

y_true = true_labels.flatten()
y_pred = pred_map.flatten()

# Trim to shared area (pred may be slightly smaller due to patch stride)
min_len = min(len(y_true), len(y_pred))
y_true, y_pred = y_true[:min_len], y_pred[:min_len]

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)), normalize="true")

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_SHORT)
disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
ax.set_title("Normalised Confusion Matrix\nHyperspectral U-Net vs. Pseudo-Labels", fontsize=11)
plt.tight_layout()
fig_path = FIGURES_DIR / "08_confusion_matrix.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Confusion matrix → {fig_path}")


# ---------------------------------------------------------------------------
# FIGURE 4: Per-Class Metrics Bar Chart
# ---------------------------------------------------------------------------
print("\n[STEP 5] Computing per-class IoU & Dice...")

iou_scores, dice_scores = [], []
for c in range(NUM_CLASSES):
    pc = (y_pred == c); tc = (y_true == c)
    inter = np.logical_and(pc, tc).sum()
    union = np.logical_or(pc, tc).sum()
    iou   = (inter + 1e-6) / (union + 1e-6)
    dice  = (2 * inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)
    iou_scores.append(iou)
    dice_scores.append(dice)
    print(f"  {CLASS_NAMES[c].replace(chr(10),' '):20s}: IoU={iou:.3f} | Dice={dice:.3f}")

x = np.arange(NUM_CLASSES)
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - w/2, iou_scores,  w, label="IoU",  color=CLASS_COLOURS, alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + w/2, dice_scores, w, label="Dice", color=CLASS_COLOURS, alpha=0.55, edgecolor="white")
ax.axhline(0.90, color="gray", linestyle="--", alpha=0.6, label="Target >0.90")
ax.set_xticks(x); ax.set_xticklabels([n.replace("\n", " ") for n in CLASS_NAMES])
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Per-Class IoU & Dice Score\nHyperspectral U-Net — Validation Tile")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig_path = FIGURES_DIR / "09_per_class_metrics.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  ✅ Per-class metrics → {fig_path}")


# ---------------------------------------------------------------------------
# FIGURE 5: Malignant Stroma Confidence Map
# ---------------------------------------------------------------------------
print("\n[STEP 6] Generating Malignant Stroma confidence heatmap...")

mal_prob = prob_map[:pred_map.shape[0], :pred_map.shape[1], 3]  # P(Malignant Stroma)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Malignant Stroma Detection Confidence\n(Class 3 Softmax Probability)", fontsize=12)

im0 = axes[0].imshow(mal_prob, cmap="hot", vmin=0, vmax=1)
axes[0].set_title("P(Malignant Stroma) Heatmap", fontsize=10); axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

mal_binary = (pred_map == 3).astype(np.uint8)
axes[1].imshow(rgb[:pred_map.shape[0], :pred_map.shape[1]])
axes[1].imshow(mal_binary, cmap="Reds", alpha=0.55, vmin=0, vmax=1)
axes[1].set_title("Predicted Malignant Regions\n(Overlay on PCA False Colour)", fontsize=10)
axes[1].axis("off")

plt.tight_layout()
fig_path = FIGURES_DIR / "10_malignant_confidence.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Confidence heatmap → {fig_path}")

print("\n" + "="*60)
print("  ✅ Evaluation Complete! All figures in:")
print(f"  {FIGURES_DIR}")
print("="*60)
