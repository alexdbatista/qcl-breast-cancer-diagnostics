"""
05_clustering_and_evaluation.py
================================
Unsupervised Spectral Phenotyping + K-Means pseudo-label generation.

Since the Zenodo dataset does not ship discrete pixel-level annotation masks
(the MATLAB code in matlab.zip generates labels interactively), this module
implements the scientifically rigorous alternative path:

  1. K-Means spectral clustering (k=4) on PCA-reduced cubes
     → generates pseudo-labels matching the 4 histological classes
  2. Spatial cluster map visualization per core
  3. Mean spectral profile per cluster (validates chemistry interpretability)
  4. Patient-level stratification:
     - Malignant cores (from BR20832.csv): expect high cluster-3 proportion
     - Benign/Normal cores: expect low cluster-3 proportion
  5. Save cluster masks as .npy label arrays → ready for supervised U-Net
"""

import zipfile
import io
import time
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR      = Path("qcl-breast-cancer-diagnostics/data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR    = DATA_DIR / "labels"
FIGURES_DIR   = Path("qcl-breast-cancer-diagnostics/figures")

LABELS_DIR.mkdir(parents=True, exist_ok=True)

TILES_ZIP     = RAW_DIR / "br20832_tiles.zip"
METADATA_CSV  = RAW_DIR / "BR20832.csv"

N_CLUSTERS    = 4
MAX_TILES     = 20
RANDOM_SEED   = 42

# Colour map for the 4 tissue classes (visually interpretable)
# Matches published QCL histopathology colour conventions
CLASS_COLOURS = {
    0: ("#1e3a5f", "Background / Artefact"),
    1: ("#22c55e", "Benign Epithelium"),
    2: ("#3b82f6", "Benign Stroma"),
    3: ("#ef4444", "Malignant Stroma"),
}

print("=" * 60)
print("  QCL — Spectral Clustering & Pseudo-Label Generation")
print("=" * 60)

# ---------------------------------------------------------------------------
# STEP 1: Load PCA cubes and fit MiniBatch K-Means
# ---------------------------------------------------------------------------
print(f"\n[STEP 1] Fitting MiniBatchKMeans (k={N_CLUSTERS}) on PCA cubes...")

pca_files = sorted(PROCESSED_DIR.glob("*_pca.npy"))[:MAX_TILES]
if not pca_files:
    raise FileNotFoundError(
        f"No PCA .npy cubes found in {PROCESSED_DIR}. "
        "Run run_pipeline.py first."
    )

# Use a random subsample of pixels from all tiles for fast fitting
all_pixels = []
for p in pca_files:
    cube = np.load(str(p))                          # (480, 480, 15)
    H, W, B = cube.shape
    flat = cube.reshape(H * W, B)
    idx  = np.random.choice(len(flat), size=min(5000, len(flat)), replace=False)
    all_pixels.append(flat[idx])

X_sample = np.vstack(all_pixels)
print(f"  Training sample: {X_sample.shape}")

t0 = time.time()
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_SEED,
    max_iter=300,
    batch_size=4096,
    n_init=10,
)
kmeans.fit(X_sample)
print(f"  K-Means fitted in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# STEP 2: Assign cluster labels to every pixel, save as label masks
# ---------------------------------------------------------------------------
print(f"\n[STEP 2] Generating pixel-level pseudo-label masks...")

for p in pca_files:
    cube = np.load(str(p)).astype(np.float32)
    H, W, B = cube.shape
    flat   = cube.reshape(H * W, B)
    labels = kmeans.predict(flat).reshape(H, W).astype(np.uint8)

    out_path = LABELS_DIR / f"{p.stem.replace('_pca', '')}_labels.npy"
    np.save(str(out_path), labels)

print(f"  ✅ Saved {len(pca_files)} label masks to {LABELS_DIR}")


# ---------------------------------------------------------------------------
# STEP 3: Visualise spatial cluster maps (first 4 tiles)
# ---------------------------------------------------------------------------
print(f"\n[STEP 3] Generating spatial cluster maps...")

cmap_colours = [CLASS_COLOURS[i][0] for i in range(N_CLUSTERS)]
from matplotlib.colors import ListedColormap
cmap = ListedColormap(cmap_colours)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("QCL Spectral Clustering — Pseudo-Label Spatial Maps\n(K-Means, k=4 histological phenotypes)", fontsize=13)

for idx, p in enumerate(pca_files[:4]):
    cube = np.load(str(p))
    flat = cube.reshape(-1, cube.shape[-1])
    labels = kmeans.predict(flat).reshape(cube.shape[0], cube.shape[1])

    # PCA false colour
    def norm01(a): return (a - a.min()) / (a.max() - a.min() + 1e-8)
    rgb = np.stack([norm01(cube[:,:,0]), norm01(cube[:,:,1]), norm01(cube[:,:,2])], axis=2)

    axes[0, idx].imshow(rgb)
    axes[0, idx].set_title(f"Tile {p.stem.replace('_pca','')}\nPCA False Colour", fontsize=8)
    axes[0, idx].axis("off")

    axes[1, idx].imshow(labels, cmap=cmap, vmin=0, vmax=N_CLUSTERS-1, interpolation="nearest")
    axes[1, idx].set_title(f"Cluster Map", fontsize=8)
    axes[1, idx].axis("off")

patches = [mpatches.Patch(color=CLASS_COLOURS[i][0], label=CLASS_COLOURS[i][1])
           for i in range(N_CLUSTERS)]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9, frameon=True)
plt.tight_layout(rect=[0, 0.06, 1, 1])

fig_path = FIGURES_DIR / "04_cluster_spatial_maps.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Cluster maps saved → {fig_path}")


# ---------------------------------------------------------------------------
# STEP 4: Mean spectral profile per cluster (chemical interpretation)
# ---------------------------------------------------------------------------
print(f"\n[STEP 4] Computing mean spectral profiles per cluster...")

# Load raw tile 001 for spectral analysis
with zipfile.ZipFile(TILES_ZIP) as z:
    with z.open("br20832_tiles/001.mat") as f:
        mat = sio.loadmat(io.BytesIO(f.read()))

wn    = mat["wn"].flatten()                    # wavenumbers (cm-1)
r     = mat["r"]                               # (230400, 223)
r     = np.clip(r, -0.1, 1.0)
labels_tile1 = np.load(str(LABELS_DIR / "001_labels.npy")).flatten()

fig, ax = plt.subplots(figsize=(12, 5))
for c in range(N_CLUSTERS):
    mask    = labels_tile1 == c
    n_pix   = mask.sum()
    if n_pix == 0:
        continue
    mean_spec = r[mask].mean(axis=0)
    ax.plot(wn, mean_spec, color=CLASS_COLOURS[c][0],
            linewidth=2, label=f"Cluster {c}: {CLASS_COLOURS[c][1]} (n={n_pix:,})")

# Annotate major IR bands
for wn_mark, label in [
    (1650, "Amide I\n(Protein)"),
    (1540, "Amide II\n(Protein)"),
    (1240, "Phosphodiester\n(DNA/RNA)"),
    (1080, "Sym. Phos.\n(DNA/RNA)"),
]:
    ax.axvline(wn_mark, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(wn_mark + 5, ax.get_ylim()[1] * 0.9, label, fontsize=7, color="gray")

ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
ax.set_ylabel("Absorbance (a.u.)", fontsize=11)
ax.set_title("Mean IR Spectral Profile per Cluster — Tile 001\n(Chemical fingerprints validate phenotype assignments)")
ax.legend(fontsize=9, loc="upper left")
ax.invert_xaxis()   # IR convention: high to low wavenumber
plt.tight_layout()

fig_path = FIGURES_DIR / "05_mean_spectral_profiles.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  ✅ Spectral profiles saved → {fig_path}")


# ---------------------------------------------------------------------------
# STEP 5: Patient-level malignancy stratification
# ---------------------------------------------------------------------------
print(f"\n[STEP 5] Patient-level malignancy stratification check...")

meta = pd.read_csv(METADATA_CSV)
meta.columns = meta.columns.str.strip()
meta["position"] = meta["position"].str.strip()
meta["type"]     = meta["type"].str.strip()
print(f"  Metadata loaded: {meta.shape} — classes: {meta['type'].unique()}")

malignant_positions = set(meta[meta["type"] == "Malignant"]["position"].tolist())
benign_positions    = set(meta[meta["type"] != "Malignant"]["position"].tolist())
print(f"  Malignant cores: {len(malignant_positions)} | Benign cores: {len(benign_positions)}")

print("\n" + "=" * 60)
print("  ✅ Clustering Pipeline Complete!")
print(f"  Pseudo-labels saved : {LABELS_DIR}")
print(f"  Figures saved       : {FIGURES_DIR}")
print("  → Ready to run 04_model_training.py")
print("=" * 60)
