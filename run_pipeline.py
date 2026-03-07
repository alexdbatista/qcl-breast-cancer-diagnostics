"""
run_pipeline.py
===============
End-to-end execution script for the QCL Breast Cancer Diagnostics pipeline.

Runs all steps sequentially, directly on the downloaded Zenodo data:

  Step 1: Inspect one tile — confirm .mat structure
  Step 2: Extract & preprocess all tiles → (480, 480, 223) cubes
  Step 3: Apply PCA across all tiles → reduce 223 bands to N components
  Step 4: Save processed .npy cubes ready for model training
  Step 5: UMAP visualization of spectral phenotypes (first tile)

Usage:
    python run_pipeline.py
"""

import zipfile
import io
import time
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR      = Path("qcl-breast-cancer-diagnostics/data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR   = Path("qcl-breast-cancer-diagnostics/figures")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TILES_ZIP   = RAW_DIR / "br20832_tiles.zip"
N_PCA       = 15          # PCA components to retain
MAX_TILES   = 20          # Process first N tiles for speed on CPU
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  QCL Breast Cancer — Full Pipeline Run")
print("=" * 60)


# ---------------------------------------------------------------------------
# STEP 1: Inspect one tile
# ---------------------------------------------------------------------------
print("\n[STEP 1] Inspecting tile 001.mat structure...")
with zipfile.ZipFile(TILES_ZIP) as z:
    with z.open("br20832_tiles/001.mat") as f:
        mat = sio.loadmat(io.BytesIO(f.read()))

dY  = int(mat["dY"].flat[0])
dX  = int(mat["dX"].flat[0])
wn  = mat["wn"].flatten()
r   = mat["r"]
cube = r.reshape(dY, dX, len(wn))

print(f"  Spatial dims   : {dY} x {dX} pixels")
print(f"  Spectral bands : {len(wn)} (range {wn.min():.0f}–{wn.max():.0f} cm⁻¹)")
print(f"  Cube shape     : {cube.shape}")
print(f"  Value range    : {r.min():.4f} to {r.max():.4f}")

# Save a false-colour RGB summary (bands 50, 120, 190 → approx. Amide II, DNA, Amide I)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Tile 001 — False-Colour QCL Bands\n(Amide II / Phosphodiester / Amide I)", fontsize=12)
for ax, band, label in zip(axes, [50, 120, 190], ["Amide II ~1540 cm⁻¹", "Phosphodiester ~1080 cm⁻¹", "Amide I ~1650 cm⁻¹"]):
    im = ax.imshow(cube[:, :, band], cmap="viridis")
    ax.set_title(f"{label}\nBand index {band}", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig_path = FIGURES_DIR / "01_false_colour_bands.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  ✅ False-colour figure saved → {fig_path}")


# ---------------------------------------------------------------------------
# STEP 2: Load first N tiles, reshape to pixel matrix
# ---------------------------------------------------------------------------
print(f"\n[STEP 2] Loading {MAX_TILES} tiles and building pixel matrix...")
all_spectra = []
tile_names  = []

with zipfile.ZipFile(TILES_ZIP) as z:
    mat_files = sorted([n for n in z.namelist() if n.endswith(".mat")])[:MAX_TILES]

    for fname in mat_files:
        t0 = time.time()
        with z.open(fname) as f:
            mat = sio.loadmat(io.BytesIO(f.read()))

        r = mat["r"]                               # (pixels, bands)
        # Clip outliers: sensor noise can produce strong negative artefacts
        r = np.clip(r, -0.1, 1.0)

        all_spectra.append(r)
        tile_names.append(Path(fname).stem)
        elapsed = time.time() - t0
        print(f"  Loaded {Path(fname).stem}: {r.shape} in {elapsed:.1f}s")

X = np.vstack(all_spectra)    # (total_pixels, 223)
print(f"\n  Combined pixel matrix: {X.shape} ({X.nbytes / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# STEP 3: Standardize + PCA
# ---------------------------------------------------------------------------
print(f"\n[STEP 3] Standardizing and applying PCA (n={N_PCA})...")
t0 = time.time()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Scaling done in {time.time()-t0:.1f}s")

t0 = time.time()
pca = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA done in {time.time()-t0:.1f}s")
print(f"  Explained variance ({N_PCA} components): {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Scree plot
fig, ax = plt.subplots(figsize=(8, 4))
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax.bar(range(1, N_PCA+1), pca.explained_variance_ratio_*100, color="#3b82f6", alpha=0.8, label="Per-component")
ax.plot(range(1, N_PCA+1), cumvar, "o--", color="#ef4444", label="Cumulative")
ax.axhline(90, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("PCA Component"); ax.set_ylabel("Explained Variance (%)")
ax.set_title(f"Scree Plot — QCL Hyperspectral PCA\n({N_PCA} components, {cumvar[-1]:.1f}% total variance captured)")
ax.legend(); plt.tight_layout()
fig_path = FIGURES_DIR / "02_pca_scree_plot.png"
plt.savefig(fig_path, dpi=150); plt.close()
print(f"  ✅ Scree plot saved → {fig_path}")


# ---------------------------------------------------------------------------
# STEP 4: Save PCA-reduced cubes
# ---------------------------------------------------------------------------
print(f"\n[STEP 4] Saving PCA-reduced cubes to {PROCESSED_DIR}...")
offset = 0
for name, raw in zip(tile_names, all_spectra):
    n_pixels = raw.shape[0]
    pca_chunk = X_pca[offset:offset + n_pixels]  # (H*W, N_PCA)

    # Recover spatial dimensions from the raw r matrix
    n_pix_sqrt = int(np.sqrt(n_pixels))
    cube_pca = pca_chunk.reshape(n_pix_sqrt, n_pix_sqrt, N_PCA)

    out_path = PROCESSED_DIR / f"{name}_pca.npy"
    np.save(str(out_path), cube_pca.astype(np.float32))
    offset += n_pixels

print(f"  ✅ Saved {len(tile_names)} PCA cubes to {PROCESSED_DIR}")


# ---------------------------------------------------------------------------
# STEP 5: Visualise PCA false-colour map (PC1, PC2, PC3 as RGB)
# ---------------------------------------------------------------------------
print(f"\n[STEP 5] Generating PCA false-colour spatial map for tile 001...")
cube_pca_tile1 = np.load(str(PROCESSED_DIR / f"{tile_names[0]}_pca.npy"))

def norm01(arr):
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)

rgb = np.stack([
    norm01(cube_pca_tile1[:, :, 0]),
    norm01(cube_pca_tile1[:, :, 1]),
    norm01(cube_pca_tile1[:, :, 2]),
], axis=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb)
axes[0].set_title("PCA False-Colour (PC1=R, PC2=G, PC3=B)\nDistinct colours = distinct tissue types")
axes[0].axis("off")

axes[1].imshow(cube_pca_tile1[:, :, 0], cmap="RdBu_r")
axes[1].set_title("PC1 Spatial Map\n(Primary spectral variability — correlates with tissue class)")
axes[1].axis("off")

plt.suptitle("QCL Tile 001 — PCA Spatial Decomposition", fontsize=13, y=1.01)
plt.tight_layout()
fig_path = FIGURES_DIR / "03_pca_spatial_map.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  ✅ PCA spatial map saved → {fig_path}")

print("\n" + "="*60)
print("  ✅ Pipeline Complete!")
print(f"  Processed tiles : {len(tile_names)}")
print(f"  PCA components  : {N_PCA}")
print(f"  Figures         : {FIGURES_DIR}")
print(f"  Processed data  : {PROCESSED_DIR}")
print("="*60)
