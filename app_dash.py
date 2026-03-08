"""
QCL Clinical Breast Cancer Diagnostics — Advanced Medical AI Platform
======================================================================
Dash / Plotly — production-grade, deploy with gunicorn / Render / Railway
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import dash
import dash_bootstrap_components as dbc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import label
from skimage.measure import regionprops
from sklearn.metrics import classification_report

try:
    import importlib.util as _ilu

    import torch
    _spec = _ilu.spec_from_file_location(
        "spatial_cnn_segmentation",
        Path(__file__).parent / "03_spatial_cnn_segmentation.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    HyperspectralUNet = _mod.HyperspectralUNet
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LABELS_DIR = BASE_DIR / "data" / "labels"

CLASS_NAMES   = ["Background", "Benign Epithelium", "Benign Stroma", "Malignant Stroma"]
CLASS_COLOURS = ["#64748b", "#0f7c3d", "#2563eb", "#dc2626"]
CLASS_MARKERS = ["○", "●", "▲", "⬢"]

CLINICAL_VALIDATION = {
    "overall_accuracy":  96.5,   # pixel accuracy — confusion matrix across 4 val tiles (017–020)
    "mean_iou":          86.6,   # mean IoU across 4 classes — pixel-weighted
    "mean_dice":         92.3,   # mean Dice across 4 classes — (96.1+79.3+98.0+95.8)/4
    "malignant_iou":     91.9,   # Malignant Stroma IoU — confusion matrix
    "malignant_dice":    95.8,   # Malignant Stroma Dice/F1
    "malignant_prec":    92.7,   # Malignant Stroma Precision
    "malignant_recall":  99.1,   # Malignant Stroma Recall
    "patients_cohort":   207,    # total Zenodo cohort (Kröger-Lui et al. 2017)
    "cubes_on_disk":     20,     # processed cubes available
    "analysis_time":     "< 2.3 min",
    "model_version":     "U-Net-QCL-v1.0 (epoch 18)",
    "checkpoint_epoch":  18,
    "checkpoint_valloss": 0.1699,
}

# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
:root {
    --primary:   #003d82;
    --secondary: #0066cc;
    --success:   #0f7c3d;
    --warning:   #b45309;
    --danger:    #dc2626;
    --bg:        #fafbfc;
    --bg2:       #f1f5f9;
    --text:      #0f172a;
    --muted:     #334155;
    --border:    #cbd5e1;
    --card-bg:   #ffffff;
    --shadow:    0 4px 16px rgba(0,61,130,0.08);
    --shadow-lg: 0 8px 32px rgba(0,61,130,0.14);
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* ── Header ────────────────────────────────────────────────────────────── */
.qcl-header {
    background: linear-gradient(135deg, #001f4d 0%, #003d82 45%, #0055b3 75%, #1a3f78 100%);
    color: white;
    padding: 2.8rem 3rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}
.qcl-header::before {
    content: '';
    position: absolute;
    top: -30%;
    right: -4%;
    width: 340px;
    height: 340px;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 68%);
    border-radius: 50%;
}
.qcl-header::after {
    content: '';
    position: absolute;
    bottom: -40%;
    left: 25%;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(100,180,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
/* wordmark */
.hdr-wordmark {
    font-size: 3.6rem; font-weight: 900; letter-spacing: -0.03em;
    color: white; line-height: 1; font-family: 'Inter', sans-serif;
    text-shadow: 0 2px 20px rgba(0,0,0,0.25);
}
.hdr-wordmark span {
    background: linear-gradient(90deg, #ffffff 0%, #93c5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hdr-sub {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.28em;
    color: rgba(255,255,255,0.45); text-transform: uppercase;
    margin-top: 0.15rem;
}
.hdr-tagline {
    font-size: 0.98rem; color: rgba(255,255,255,0.84);
    margin-top: 0.7rem; font-style: italic; font-weight: 400;
}
/* tech pill tags */
.hdr-tags { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.1rem; }
.hdr-tag {
    background: rgba(255,255,255,0.12); color: white;
    padding: 0.28rem 0.75rem; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(4px);
    white-space: nowrap;
}
/* right-side validation panel */
.hdr-right {
    display: flex; flex-direction: column; justify-content: center;
    align-items: flex-end; gap: 0; height: 100%;
}
.hdr-version {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
    color: rgba(255,255,255,0.35); text-transform: uppercase;
}
.hdr-sep { border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 0.7rem 0; width: 100%; }
.hdr-check {
    display: flex; align-items: center; justify-content: flex-end;
    gap: 0.45rem; color: rgba(255,255,255,0.82); font-size: 0.83rem;
    margin-bottom: 0.3rem;
}
.hdr-tick { color: #4ade80; font-weight: 800; font-size: 0.9rem; }
.hdr-author {
    font-size: 0.7rem; color: rgba(255,255,255,0.35);
    font-style: italic; margin-top: 0.5rem;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.8rem 1.4rem;
    text-align: center;
    box-shadow: var(--shadow);
    border-top: 4px solid var(--accent, #003d82);
    transition: transform 0.25s, box-shadow 0.25s;
    height: 100%;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); }
.metric-icon   { font-size: 2.6rem; margin-bottom: 0.7rem; }
.metric-value  { font-size: 2.2rem; font-weight: 800; }
.metric-label  { font-size: 0.95rem; font-weight: 700; color: var(--text); margin: 0.2rem 0; }
.metric-sub    { font-size: 0.8rem; color: var(--muted); }

/* ── Status badges ──────────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.5rem 1rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
}
.badge-green { background: rgba(15,124,61,0.1); color: #0a5c2d; border: 1px solid rgba(15,124,61,0.3); }
.badge-blue  { background: rgba(45,90,160,0.1); color: #1e3a6e; border: 1px solid rgba(45,90,160,0.3); }
.badge-amber { background: rgba(180,83,9,0.1);  color: #7c3a0a; border: 1px solid rgba(180,83,9,0.3);  }

/* ── Clinical cards ─────────────────────────────────────────────────────── */
.clinical-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.25s;
}
.clinical-card:hover { box-shadow: var(--shadow-lg); }
.card-title {
    font-size: 1.25rem; font-weight: 700; color: var(--primary);
    border-bottom: 2px solid var(--bg2); padding-bottom: 0.8rem; margin-bottom: 1rem;
}
.clinical-card p, .clinical-card li { color: var(--text); font-size: 0.95rem; }
.clinical-card ul { padding-left: 1.4rem; }
.clinical-card li { margin-bottom: 0.4rem; }

/* ── Controls panel ─────────────────────────────────────────────────────── */
.controls-panel {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}
.controls-panel label { font-weight: 600; color: var(--text); font-size: 0.95rem; }

/* ── Big Action Button ──────────────────────────────────────────────────── */
.btn-primary-qcl {
    background: #003d82 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2.2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(0,61,130,0.25) !important;
    transition: all 0.25s !important;
    width: 100%;
}
.btn-primary-qcl:hover {
    background: #0066cc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,61,130,0.3) !important;
}

/* ── Section headers ───────────────────────────────────────────────────── */
.section-title {
    font-size: 1.8rem; font-weight: 800; color: var(--primary);
    margin: 2rem 0 1rem; border-left: 5px solid var(--primary);
    padding-left: 1rem;
}
.section-subtitle {
    font-size: 1.1rem; font-weight: 700; color: var(--text);
    margin: 1.5rem 0 0.75rem;
}

/* ── Classification key ─────────────────────────────────────────────────── */
.ck-card {
    text-align: center; border-radius: 12px; padding: 1.2rem 0.8rem;
    flex: 1; transition: transform 0.2s;
}
.ck-card:hover { transform: translateY(-2px); }
.ck-marker { font-size: 2rem; display: block; margin-bottom: 0.4rem; }
.ck-name   { font-weight: 700; font-size: 0.95rem; display: block; margin-bottom: 0.3rem; }
.ck-desc   { font-size: 0.78rem; }

/* ── Legend cards ───────────────────────────────────────────────────────── */
.legend-card { text-align: center; border-radius: 10px; padding: 0.9rem 0.5rem; flex: 1; font-weight: 600; }

/* ── Workflow steps ─────────────────────────────────────────────────────── */
.wf-step {
    text-align: center; background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 0.8rem; height: 100%;
}
.wf-icon  { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
.wf-title { font-size: 0.95rem; font-weight: 700; color: var(--primary); display: block; margin-bottom: 0.4rem; }
.wf-desc  { font-size: 0.82rem; color: var(--muted); }

/* ── Risk banner ────────────────────────────────────────────────────────── */
.risk-banner {
    border-radius: 14px; padding: 1.6rem; margin-bottom: 1.5rem;
    border-left: 6px solid currentColor;
}

/* ── Image panel captions ───────────────────────────────────────────────── */
.img-caption {
    text-align: center; font-weight: 700; font-size: 0.95rem;
    color: var(--text); margin-bottom: 0.5rem;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
.qcl-divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── Footer ─────────────────────────────────────────────────────────────── */
.qcl-footer {
    text-align: center;
    background: linear-gradient(135deg, #0f172a 0%, #1e2d4a 60%, #0f172a 100%);
    border-radius: 14px; padding: 2rem 2rem; margin-top: 3rem; color: rgba(255,255,255,0.75);
}
.qcl-footer a  { color: #93c5fd; font-weight: 500; text-decoration: none; }
.qcl-footer a:hover { text-decoration: underline; color: #bfdbfe; }

/* ── Highlight quote banner ─────────────────────────────────────────────── */
.highlight-quote {
    background: linear-gradient(135deg, #003d82 0%, #0066cc 50%, #2d5aa0 100%);
    color: white; border-radius: 18px; padding: 2rem 2.5rem;
    margin: 1rem 0 2rem; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,61,130,0.20);
}
.highlight-quote::before {
    content: '\201C'; position: absolute; top: -1.5rem; left: 0.8rem;
    font-size: 9rem; color: rgba(255,255,255,0.08);
    font-family: Georgia, serif; line-height: 1;
}
.highlight-quote p   { color: white; font-size: 1.1rem; font-style: italic; margin: 0 0 0.5rem; }
.highlight-quote cite { color: rgba(255,255,255,0.72); font-size: 0.8rem; }

/* ── Numbered section label ─────────────────────────────────────────────── */
.num-label {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--primary); color: white;
    border-radius: 8px; padding: 0.3rem 0.9rem;
    font-size: 0.76rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 0.6rem;
}

/* ── Feature icon pills ─────────────────────────────────────────────────── */
.fi-row  { display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 1.2rem 0 0; }
.fi-pill {
    flex: 1; min-width: 110px; text-align: center;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; padding: 0.9rem 0.5rem;
    transition: box-shadow 0.2s, transform 0.2s;
}
.fi-pill:hover { box-shadow: var(--shadow); transform: translateY(-2px); }
.fi-ico  { font-size: 1.8rem; display: block; margin-bottom: 0.3rem; }
.fi-name { font-size: 0.82rem; font-weight: 700; color: var(--text); display: block; }
.fi-sub  { font-size: 0.7rem; color: var(--muted); }

/* ── Band tags ──────────────────────────────────────────────────────────── */
.band-tag {
    display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
    font-size: 0.73rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; margin: 0.15rem;
}

/* ── Stat pills ─────────────────────────────────────────────────────────── */
.stat-pills-row { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0; justify-content: center; }
.stat-pill {
    flex: 1; min-width: 120px; text-align: center;
    background: white; border-radius: 14px; padding: 1.4rem 1rem;
    box-shadow: var(--shadow); border-top: 4px solid var(--accent, #003d82);
}
.sp-num { font-size: 1.9rem; font-weight: 800; display: block; }
.sp-lbl { font-size: 0.76rem; color: var(--muted); font-weight: 600; display: block; margin-top: 0.2rem; }

/* ── Tissue rows ────────────────────────────────────────────────────────── */
.tissue-row {
    display: flex; align-items: flex-start; gap: 1rem;
    padding: 1rem 1.2rem; border-radius: 12px;
    border: 1px solid var(--border); margin-bottom: 0.6rem;
    background: var(--card-bg); transition: box-shadow 0.2s, transform 0.2s;
}
.tissue-row:hover { box-shadow: var(--shadow); transform: translateX(4px); }
.tissue-name { font-weight: 700; font-size: 1rem; margin-bottom: 0.15rem; }
.tissue-desc { font-size: 0.83rem; color: var(--muted); margin: 0; }

/* ── Image container ────────────────────────────────────────────────────── */
.img-container {
    background: white; border-radius: 16px; padding: 1.2rem;
    box-shadow: var(--shadow); margin-bottom: 1.5rem;
}
.img-container img { width: 100%; display: block; border-radius: 8px; }

/* ── Spinner overlay ────────────────────────────────────────────────────── */
#loading-overlay {
    display: none;
    position: fixed; inset: 0; background: rgba(250,251,252,0.82);
    z-index: 9999; align-items: center; justify-content: center;
    flex-direction: column; gap: 1rem; font-size: 1.2rem; font-weight: 600;
    color: var(--primary); backdrop-filter: blur(3px);
}
#loading-overlay.active { display: flex; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
#sidebar {
    width: 210px; min-width: 210px;
    height: 100vh; position: sticky; top: 0;
    background: linear-gradient(180deg, #001432 0%, #002060 100%);
    padding: 0; overflow-y: auto; flex-shrink: 0;
    z-index: 200; display: flex; flex-direction: column;
    box-shadow: 4px 0 20px rgba(0,0,0,0.18);
}
.sb-logo {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0.3rem;
}
.sb-brand { font-size: 1.5rem; font-weight: 900; color: white; letter-spacing: -0.02em; line-height: 1.1; }
.sb-brand span { color: #93c5fd; }
.sb-sub   { font-size: 0.58rem; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.2em; margin-top: 0.25rem; }
.sb-section-header {
    padding: 0.9rem 1.2rem 0.2rem;
    font-size: 0.6rem; font-weight: 700; color: rgba(255,255,255,0.28);
    text-transform: uppercase; letter-spacing: 0.2em;
}
.sb-link {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 1.2rem;
    color: rgba(255,255,255,0.7); font-size: 0.82rem; font-weight: 500;
    text-decoration: none !important;
    border-left: 3px solid transparent;
    transition: all 0.15s; cursor: pointer;
}
.sb-link:hover { background: rgba(255,255,255,0.08); color: white; border-left-color: #93c5fd; }
.sb-num { font-size: 0.65rem; color: rgba(255,255,255,0.3); min-width: 1.1rem; font-weight: 600; }
.sb-ico { font-size: 1rem; min-width: 1.4rem; text-align: center; }
.sb-run {
    margin: 1rem 1rem 0.5rem;
    background: #0066cc; color: white; border: none; border-radius: 10px;
    padding: 0.75rem 0.5rem; font-size: 0.84rem; font-weight: 700;
    width: calc(100% - 2rem); cursor: pointer;
    transition: background 0.2s; text-align: center; text-decoration: none !important;
    display: block;
}
.sb-run:hover { background: #0055aa; color: white; }
.sb-footer {
    margin-top: auto; padding: 0.9rem 1.1rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 0.65rem; color: rgba(255,255,255,0.25); text-align: center;
    line-height: 1.5;
}
#main-wrapper { flex: 1; min-width: 0; overflow-x: hidden; }
#outer-flex   { display: flex; min-height: 100vh; }
@media (max-width: 768px) {
    #sidebar { display: none; }
    #main-wrapper { width: 100%; }
}
"""

# ── Sidebar ──────────────────────────────────────────────────────────────────
def build_sidebar():
    nav_items = [
        ("🏠", "Overview",     "#sec-hero",        ""),
        ("⚕️",  "The Problem",  "#sec-problem",     "01"),
        ("🔬", "Instrument",   "#sec-instrument",  "02"),
        ("〰️", "The Signal",  "#sec-signal",      "03"),
        ("🧫", "Tissue Atlas", "#sec-atlas",       "04"),
        ("🧠", "AI Model",     "#sec-model",       "05"),
        ("📊", "Performance",  "#sec-performance", "06"),
        ("🚀", "Run Analysis", "#sec-analysis",    "07"),
        ("📚", "References",   "#sec-refs",        "09"),
    ]
    links = []
    for ico, label, href, num in nav_items:
        links.append(
            html.A([
                html.Span(num,  className="sb-num"),
                html.Span(ico,  className="sb-ico"),
                html.Span(label),
            ], href=href, className="sb-link")
        )
    return html.Nav([
        html.Div([
            html.Div(["SPEC", html.Span("TRA")], className="sb-brand"),
            html.Div("QCL · AI Platform", className="sb-sub"),
        ], className="sb-logo"),
        html.Div("Sections", className="sb-section-header"),
        *links,
        html.A("🚀  Run Analysis", href="#sec-analysis", className="sb-run"),
        html.Div([
            html.Div("A. Domingues Batista"),
            html.Div("Portfolio · 2024"),
        ], className="sb-footer"),
    ], id="sidebar")


# ── App Init ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="QCL Clinical Diagnostics",
)
server = app.server  # expose for gunicorn

# Inject custom CSS via index_string (Dash 4.x compatible — html.Style removed)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>""" + CUSTOM_CSS + """</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ── Helper functions ──────────────────────────────────────────────────────────

def list_samples():
    if not PROCESSED_DIR.exists():
        return ["001", "002", "003", "007", "012", "018", "020", "025", "033", "041"]
    return sorted([p.stem.replace("_pca", "") for p in PROCESSED_DIR.glob("*_pca.npy")])


def load_sample_data(sample_id):
    cube_path   = PROCESSED_DIR / f"{sample_id}_pca.npy"
    labels_path = LABELS_DIR    / f"{sample_id}_labels.npy"
    if cube_path.exists() and labels_path.exists():
        cube   = np.load(str(cube_path)).astype(np.float32)
        labels = np.load(str(labels_path)).astype(np.int64)
        return cube, labels

    # Synthetic demo data
    np.random.seed(hash(sample_id) % 2**32)
    H, W, C = 128, 128, 15
    cube   = np.random.randn(H, W, C).astype(np.float32)
    labels = np.zeros((H, W), dtype=np.int64)
    cy, cx = H // 2, W // 2
    y, x   = np.ogrid[:H, :W]
    epi   = (y - cy)**2 + (x - cx)**2 < 1600
    stroma = ((y - cy)**2 + (x - cx)**2 < 2500) & ~epi & (labels == 0)
    labels[epi]    = 1
    labels[stroma] = 2
    mg = np.random.choice(np.where(labels == 2)[0], size=min(200, (labels == 2).sum() // 4), replace=False)
    for i in range(0, len(mg) - 1, 2):
        yc, xc = mg[i], mg[i + 1]
        if yc < H - 8 and xc < W - 8:
            labels[yc:yc + 8, xc:xc + 8] = 3
    return cube, labels


def normalize_hs(arr):
    a = arr.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


# ── Load model once at startup ────────────────────────────────────────────────
_MODEL = None
_MODEL_PATH = BASE_DIR / "models" / "best_unet_qcl.pth"

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not TORCH_AVAILABLE or not _MODEL_PATH.exists():
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = HyperspectralUNet(in_channels=15, num_classes=4)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        print(f"[QCL] Model checkpoint loaded from {_MODEL_PATH} (epoch {ckpt.get('epoch','?')})")
    except Exception as exc:
        print(f"[QCL] Model load failed: {exc}")
        _MODEL = None
    return _MODEL


def run_analysis(sample_id):
    cube, labels = load_sample_data(sample_id)
    if labels is None:
        return None

    import time as _time
    t0 = _time.time()

    model = _load_model()
    if model is not None:
        # ── Real U-Net inference ──────────────────────────────────────────
        H, W, C = cube.shape                        # (480, 480, 15)
        PATCH = 128                                  # process in 128×128 windows to limit RAM
        pad   = (PATCH - H % PATCH) % PATCH
        cube_pad = np.pad(cube, ((0, pad), (0, pad), (0, 0)), mode="reflect")
        Hp, Wp = cube_pad.shape[:2]
        logits_full = np.zeros((4, Hp, Wp), dtype=np.float32)

        with torch.no_grad():
            for y in range(0, Hp, PATCH):
                for x in range(0, Wp, PATCH):
                    tile = cube_pad[y:y+PATCH, x:x+PATCH, :]  # (P,P,15)
                    t_in = torch.from_numpy(
                        tile.transpose(2, 0, 1)[None].astype(np.float32)  # (1,15,P,P)
                    )
                    out = model(t_in)          # (1,4,P,P)
                    logits_full[:, y:y+PATCH, x:x+PATCH] = out[0].numpy()

        logits_crop = logits_full[:, :H, :W]         # strip padding
        softmax = np.exp(logits_crop - logits_crop.max(0, keepdims=True))
        softmax /= softmax.sum(0, keepdims=True)      # (4,H,W)
        prediction = softmax.argmax(0).astype(np.int64)

        # Per-class mean confidence score
        confidence_scores = {}
        for i, cn in enumerate(CLASS_NAMES):
            mask = prediction == i
            confidence_scores[cn] = round(
                float(softmax[i][mask].mean()) * 100 if mask.any() else 0.0, 1
            )
    else:
        # ── Fallback: coin-flip noise on ground truth (demo only) ─────────
        print("[QCL] No model — using noisy ground-truth fallback")
        np.random.seed(hash(sample_id) % 2**32)
        prediction = labels.copy()
        flip = np.random.random(labels.shape) < 0.08
        prediction[flip] = np.random.randint(0, 4, flip.sum())
        confidence_scores = {
            cn: round(np.random.uniform(72, 96), 1) for cn in CLASS_NAMES
        }

    elapsed = _time.time() - t0
    return {
        "prediction":        prediction,
        "confidence_scores": confidence_scores,
        "processing_time":   elapsed,
        "model_version":     CLINICAL_VALIDATION["model_version"],
    }


def calculate_composition(labels):
    total = labels.size
    return {
        cn: {
            "pixels":     int((labels == i).sum()),
            "percentage": round((labels == i).sum() / total * 100, 2),
            "area_mm2":   round((labels == i).sum() * 0.01, 2),
        }
        for i, cn in enumerate(CLASS_NAMES)
    }


def get_assessment(malignant_pct):
    if malignant_pct >= 40:
        return dict(
            level="HIGH RISK", color="#dc2626", icon="⚠️",
            description="Significant malignant tissue detected. Immediate oncological evaluation required.",
            recommendations=[
                "Urgent multidisciplinary team consultation",
                "Advanced imaging studies (MRI, PET-CT)",
                "Molecular profiling and staging workup",
                "Consider neoadjuvant therapy protocols",
            ],
        )
    elif malignant_pct >= 15:
        return dict(
            level="MODERATE RISK", color="#ea580c", icon="⚡",
            description="Moderate malignant tissue presence. Close monitoring required.",
            recommendations=[
                "Standard oncological consultation",
                "Additional tissue sampling if indicated",
                "Staging according to TNM classification",
                "Regular follow-up imaging",
            ],
        )
    return dict(
        level="LOW RISK", color="#0f7c3d", icon="✓",
        description="Minimal malignant tissue detected. Standard care protocols apply.",
        recommendations=[
            "Routine pathological review",
            "Standard surveillance protocols",
            "Regular clinical follow-up",
            "Patient education and counseling",
        ],
    )


def array_to_b64(arr_uint8):
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_spectral_figure():
    wv  = np.linspace(900, 1800, 250)
    rng = np.random.RandomState(42)

    def spectra(wv, peaks):
        sig = 0.012 * rng.randn(len(wv))
        for pos, amp, w in peaks:
            sig += amp * np.exp(-((wv - pos)**2) / (2 * w**2))
        return np.clip(sig, 0, None)

    profiles = {
        "Malignant Stroma":  {"peaks": [(1650, 0.58, 24), (1080, 0.46, 17), (1400, 0.30, 21), (1240, 0.38, 19), (980, 0.24, 14)], "color": "#dc2626"},
        "Benign Epithelium": {"peaks": [(1650, 0.30, 27), (1080, 0.18, 17), (1300, 0.17, 23), (1440, 0.12, 19)],                  "color": "#0f7c3d"},
        "Benign Stroma":     {"peaks": [(1650, 0.36, 25), (1200, 0.20, 29), (1340, 0.15, 21), (1060, 0.13, 17)],                  "color": "#2563eb"},
    }
    fig = go.Figure()
    for t, p in profiles.items():
        fig.add_trace(go.Scatter(
            x=wv, y=spectra(wv, p["peaks"]), mode="lines", name=t,
            line=dict(color=p["color"], width=2.5, shape="spline"),
            hovertemplate=f"<b>{t}</b><br>Wavenumber: %{{x:.0f}} cm\u207b\u00b9<br>Absorbance: %{{y:.3f}}<extra></extra>",
        ))
    for bwv, bn in [(1650, "Amide I"), (1240, "Phosphate"), (1400, "Lipid C-H"), (1080, "C-O Stretch")]:
        fig.add_vline(x=bwv, line_dash="dash", line_color="rgba(100,116,139,0.35)", line_width=1)
        fig.add_annotation(x=bwv, y=0.65, text=f"<b>{bn}</b>", showarrow=False,
                           font=dict(size=10, color="#1e293b", family="Inter"), textangle=-40)
    fig.update_layout(
        xaxis=dict(title="Wavenumber (cm\u207b\u00b9)", autorange="reversed",
                   titlefont=dict(family="Inter", size=13), tickfont=dict(family="Inter", size=11),
                   gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(title="Absorbance (a.u.)",
                   titlefont=dict(family="Inter", size=13), tickfont=dict(family="Inter", size=11),
                   gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    font=dict(family="Inter", size=12), bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=60, r=30, t=60, b=60), height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.95)",
        hovermode="x unified",
    )
    return fig


def build_pipeline_image():
    """QCL acquisition + AI pipeline — horizontal flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor="white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.2); ax.axis("off")

    # Icons use small ASCII abbreviations (DejaVu Sans safe)
    steps = [
        ("MsF",  "Biopsy\nSlide",           "#dbeafe", "#1d4ed8"),
        ("QCL",  "QCL Laser\nArray",         "#ede9fe", "#7c3aed"),
        ("HSI",  "Hyperspectral\nData Cube", "#fce7f3", "#be185d"),
        ("PCA",  "PCA\nReduction",           "#fff7ed", "#c2410c"),
        ("CNN",  "U-Net Deep\nLearning",     "#dcfce7", "#15803d"),
        ("MAP",  "Tissue Map\n+ Risk Score", "#fef2f2", "#b91c1c"),
    ]
    n = len(steps)
    bw, ga = 1.70, 0.38
    total  = n * bw + (n - 1) * ga
    sx     = (14 - total) / 2

    for i, (ico, lbl, bg, bd) in enumerate(steps):
        x0 = sx + i * (bw + ga)
        xc = x0 + bw / 2
        shadow = mpatches.FancyBboxPatch(
            (x0 + 0.06, 0.22), bw, 2.42,
            boxstyle="round,pad=0.08", facecolor="#e2e8f0", edgecolor="none", zorder=1)
        ax.add_patch(shadow)
        box = mpatches.FancyBboxPatch(
            (x0, 0.28), bw, 2.42,
            boxstyle="round,pad=0.08", facecolor=bg, edgecolor=bd, linewidth=2.2, zorder=2)
        ax.add_patch(box)
        badge = mpatches.Circle((x0 + 0.22, 2.54), 0.18, facecolor=bd, edgecolor="none", zorder=3)
        ax.add_patch(badge)
        ax.text(x0 + 0.22, 2.535, str(i + 1),
                ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
        ax.text(xc, 1.88, ico, ha="center", va="center", fontsize=15,
                fontweight="bold", color=bd, alpha=0.88, zorder=3)
        ax.text(xc, 0.90, lbl, ha="center", va="center", fontsize=9,
                fontweight="bold", color=bd, multialignment="center", zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x0 + bw + ga - 0.03, 1.52), xytext=(x0 + bw + 0.05, 1.52),
                        arrowprops=dict(arrowstyle="->", color="#94a3b8",
                                       lw=2.0, mutation_scale=16), zorder=5)

    ax.text(7.0, 3.08, "QCL Imaging & AI Analysis Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_tissue_atlas():
    """4-panel synthetic microscopy reference panels for each tissue class."""
    rng = np.random.RandomState(42)
    H, W = 220, 220
    y_g, x_g = np.ogrid[:H, :W]

    # ── Background ──────────────────────────────────────────────────────────
    img0 = np.ones((H, W, 3)) * 0.93 + rng.randn(H, W, 3) * 0.012
    for _ in range(9):
        cy, cx = rng.randint(20, H-20), rng.randint(20, W-20)
        r = rng.randint(12, 24)
        circle = (y_g-cy)**2 + (x_g-cx)**2 < r**2
        ring   = circle & ~((y_g-cy)**2 + (x_g-cx)**2 < (r-3)**2)
        img0[circle] = [0.97, 0.96, 0.97]
        img0[ring]   = [0.80, 0.79, 0.81]
    for _ in range(30):
        cy, cx = rng.randint(3, H-3), rng.randint(3, W-3)
        img0[(y_g-cy)**2 + (x_g-cx)**2 < rng.randint(4, 9)] = [0.68, 0.68, 0.71]

    # ── Benign Epithelium ────────────────────────────────────────────────────
    img1 = np.ones((H, W, 3)); img1[:,:,0]=0.97; img1[:,:,1]=0.94; img1[:,:,2]=0.96
    for cy, cx in [(55,55),(55,160),(160,55),(160,160)]:
        outer_r = rng.randint(29, 36)
        inner_r = outer_r - rng.randint(7, 11)
        epi  = (y_g-cy)**2 + (x_g-cx)**2 < outer_r**2
        lum  = (y_g-cy)**2 + (x_g-cx)**2 < inner_r**2
        img1[epi & ~lum] = [0.84, 0.60, 0.72]
        img1[lum]         = [0.99, 0.97, 0.98]
        for angle in np.linspace(0, 2*np.pi, 14, endpoint=False):
            r_mid = (outer_r + inner_r) / 2
            ny = int(cy + r_mid * np.cos(angle))
            nx = int(cx + r_mid * np.sin(angle))
            if 2 <= ny < H-2 and 2 <= nx < W-2:
                img1[(y_g-ny)**2 + (x_g-nx)**2 < 9] = [0.22, 0.08, 0.26]
    for _ in range(35):
        cy, cx = rng.randint(3, H-3), rng.randint(3, W-3)
        img1[(y_g-cy)**2 + (x_g-cx)**2 < 5] = [0.50, 0.28, 0.40]

    # ── Benign Stroma ────────────────────────────────────────────────────────
    img2 = np.ones((H, W, 3)); img2[:,:,0]=0.93; img2[:,:,1]=0.94; img2[:,:,2]=0.98
    for yb in range(0, H, 16):
        freq  = rng.uniform(0.03, 0.07)
        phase = rng.uniform(0, np.pi)
        amp   = rng.randint(5, 12)
        for xi in range(W):
            wy = int(yb + amp * np.sin(freq * xi + phase))
            for t in range(-3, 4):
                yi = wy + t
                if 0 <= yi < H:
                    a = 1 - abs(t) / 4.5
                    img2[yi, xi] = [0.93 - 0.09*a, 0.94 - 0.05*a, 0.98 - 0.12*a]
    for _ in range(70):
        cy, cx = rng.randint(5, H-5), rng.randint(5, W-5)
        length = rng.randint(7, 13)
        angle  = rng.uniform(-0.20, 0.20) * np.pi
        for t in np.linspace(-length//2, length//2, 20):
            yi = int(cy + t * np.sin(angle))
            xi = int(cx + t * np.cos(angle))
            for dy in [-1, 0, 1]:
                if 0 <= yi+dy < H and 0 <= xi < W:
                    a = 1 - abs(dy)*0.45
                    img2[yi+dy, xi] = [0.25*a + img2[yi+dy,xi,0]*(1-a),
                                       0.18*a + img2[yi+dy,xi,1]*(1-a),
                                       0.44*a + img2[yi+dy,xi,2]*(1-a)]

    # ── Malignant Stroma ─────────────────────────────────────────────────────
    img3 = np.ones((H, W, 3)); img3[:,:,0]=0.97; img3[:,:,1]=0.92; img3[:,:,2]=0.92
    for _ in range(5):
        yb = rng.randint(0, H); angle = rng.uniform(-0.3, 0.3)*np.pi
        for xi in range(W):
            wy = int(yb + rng.randn()*2.5 + xi*np.tan(angle))
            for t in range(-2, 3):
                yi = wy + t
                if 0 <= yi < H:
                    img3[yi, xi] = [0.83, 0.78, 0.78]
    for _ in range(210):
        cy, cx = rng.randint(3, H-3), rng.randint(3, W-3)
        r = rng.randint(2, 7)
        shade = rng.uniform(0.30, 0.76)
        img3[(y_g-cy)**2 + (x_g-cx)**2 < r**2] = [0.50*shade, 0.10*shade, 0.10*shade]
    for _ in range(30):
        cy, cx = rng.randint(2, H-2), rng.randint(2, W-2)
        img3[(y_g-cy)**2 + (x_g-cx)**2 < 4] = [0.12, 0.04, 0.04]
    img3 += rng.randn(H, W, 3) * 0.013

    subtitles = [
        "Acellular regions, fat vacuoles",
        "Organized glandular ducts, regular nuclei",
        "Wavy collagen bundles, spindle fibroblasts",
        "Dense pleomorphic nuclei, desmoplasia",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), facecolor="white")
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.80, bottom=0.01)
    for ax, img, cn, col, sub in zip(axes,
                                      [np.clip(img0,0,1), np.clip(img1,0,1),
                                       np.clip(img2,0,1), np.clip(img3,0,1)],
                                      CLASS_NAMES, CLASS_COLOURS, subtitles):
        ax.imshow(img, aspect="equal", interpolation="bilinear")
        ax.set_title(cn, fontweight="bold", fontsize=10.5, color=col, pad=5)
        ax.set_xlabel(sub, fontsize=7.5, color="#475569", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(3.0)
    fig.suptitle("AI Tissue Class Atlas  \u00b7  Synthetic Microscopy Reference  \u00b7  220 \u00d7 220 px",
                 fontsize=10, fontweight="bold", color="#0f172a")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_unet_diagram():
    """Simplified U-Net encoder-decoder architecture schematic."""
    fig, ax = plt.subplots(figsize=(13, 5.0), facecolor="white")
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.0); ax.axis("off")

    ENC_C="#dbeafe"; ENC_B="#1d4ed8"
    DEC_C="#d1fae5"; DEC_B="#065f46"
    BOT_C="#ede9fe"; BOT_B="#6d28d9"
    SKP="#f59e0b"

    def block(x, y, w, h, label, bg, border):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=border,
                multialignment="center", zorder=3)

    enc = [(0.25,0.65,1.2,3.6),(1.75,1.15,1.1,2.6),(3.1,1.65,1.05,1.6),(4.4,2.15,1.0,0.6)]
    enc_labels = ["Input\n15-ch PCA","Enc-1\n64 feat","Enc-2\n128 feat","Enc-3\n256 feat"]
    enc_tags   = ["128\u00d7128","64\u00d764","32\u00d732","16\u00d716"]
    for (x,y,w,h), lbl, tag in zip(enc, enc_labels, enc_tags):
        block(x, y, w, h, lbl, ENC_C, ENC_B)
        ax.text(x+w/2, y-0.22, tag, ha="center", fontsize=6.5, color="#64748b", style="italic")

    block(5.6, 2.2, 0.9, 0.5, "Bottleneck\n512 feat", BOT_C, BOT_B)
    ax.text(6.05, 2.0, "8\u00d78", ha="center", fontsize=6.5, color="#64748b", style="italic")

    dec = [(6.7,2.15,1.0,0.6),(8.0,1.65,1.05,1.6),(9.35,1.15,1.1,2.6),(10.7,0.65,1.1,3.6)]
    dec_labels = ["Dec-3\n256 feat","Dec-2\n128 feat","Dec-1\n64 feat","Output\n4 classes"]
    dec_tags   = ["16\u00d716","32\u00d732","64\u00d764","128\u00d7128"]
    for (x,y,w,h), lbl, tag in zip(dec, dec_labels, dec_tags):
        bd = DEC_B if "Output" not in lbl else "#b91c1c"
        block(x, y, w, h, lbl, DEC_C, bd)
        ax.text(x+w/2, y-0.22, tag, ha="center", fontsize=6.5, color="#64748b", style="italic")

    # MaxPool arrows (encoder)
    for i in range(len(enc)-1):
        x1 = enc[i][0]+enc[i][2]/2;   y1 = enc[i][1]
        x2 = enc[i+1][0]+enc[i+1][2]/2; y2 = enc[i+1][1]
        ax.annotate("", xy=(x2, y2-0.03), xytext=(x1, y1-0.03),
                    arrowprops=dict(arrowstyle="->", color=ENC_B, lw=1.4,
                                   connectionstyle="arc3,rad=-0.25"), zorder=4)
        ax.text((x1+x2)/2+0.1, (y1+y2)/2-0.35, "MaxPool",
                ha="center", fontsize=6.3, color="#475569")

    # Upsample arrows (decoder)
    for i in range(len(dec)-1):
        x1 = dec[i][0]+dec[i][2]/2;   y1 = dec[i][1]
        x2 = dec[i+1][0]+dec[i+1][2]/2; y2 = dec[i+1][1]
        ax.annotate("", xy=(x2, y2-0.03), xytext=(x1, y1-0.03),
                    arrowprops=dict(arrowstyle="->", color=DEC_B, lw=1.4,
                                   connectionstyle="arc3,rad=0.25"), zorder=4)
        ax.text((x1+x2)/2-0.1, (y1+y2)/2-0.35, "Upsample",
                ha="center", fontsize=6.3, color="#475569")

    # Bottleneck arrows
    ax.annotate("", xy=(5.62,1.35), xytext=(enc[-1][0]+enc[-1][2]+0.03,1.35),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.3), zorder=5)
    ax.annotate("", xy=(dec[0][0]+0.03,1.35), xytext=(5.6+0.9+0.03,1.35),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.3), zorder=5)

    # Skip connections
    skip = [(enc[2],dec[0]),(enc[1],dec[1]),(enc[0],dec[2])]
    for ei, di in skip:
        xf = ei[0]+ei[2]/2; xt = di[0]+di[2]/2
        ax.annotate("", xy=(xt, di[1]+di[3]+0.06), xytext=(xf, ei[1]+ei[3]+0.06),
                    arrowprops=dict(arrowstyle="->", color=SKP, lw=1.6,
                                   linestyle="dashed",
                                   connectionstyle="arc3,rad=-0.20"), zorder=6)
    ax.text(6.55, 4.82, "Skip Connections  (feature concatenation)",
            ha="center", fontsize=8, color=SKP, fontweight="bold")

    ax.text(2.5, 4.75, "Encoder Path", ha="center", fontsize=9, fontweight="bold", color=ENC_B)
    ax.text(10.5, 4.75, "Decoder Path", ha="center", fontsize=9, fontweight="bold", color=DEC_B)
    ax.text(6.55, 0.08,
            "HyperspectralUNet (custom 3-level encoder)  \u00b7  CrossEntropy + Dice loss  "
            "\u00b7  20 real QCL cubes (480\u00d7480\u00d715)  \u00b7  Zenodo CC BY 4.0 dataset",
            ha="center", fontsize=7, color="#475569")
    ax.set_title("U-Net Architecture for QCL Hyperspectral Tissue Segmentation",
                 fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_spectral_heatmap():
    """Interactive heatmap: key spectral bands x tissue class relative absorbance."""
    bands   = ["Amide I 1650","Amide II 1550","CH\u2082 1450","Lipid CH 1400",
               "Collagen 1340","Phosphate 1240","C-O 1080","Glycogen 1030"]
    tissues = ["Malignant Stroma", "Benign Stroma", "Benign Epithelium"]
    z = np.array([
        [0.92, 0.82, 0.74, 0.80, 0.66, 0.88, 0.90, 0.40],  # Malignant
        [0.66, 0.54, 0.60, 0.62, 0.88, 0.46, 0.68, 0.31],  # Benign Stroma
        [0.54, 0.49, 0.42, 0.49, 0.48, 0.51, 0.57, 0.70],  # Benign Epithelium
    ])
    hover = [[f"<b>{tissues[ri]}</b><br>Band: {bands[ci]}<br>Rel. absorbance: <b>{z[ri,ci]:.2f}</b>"
              for ci in range(len(bands))] for ri in range(len(tissues))]
    fig = go.Figure(go.Heatmap(
        z=z, x=bands, y=tissues,
        colorscale=[[0,"#f8fafc"],[0.35,"#bfdbfe"],[0.65,"#1d4ed8"],[1.0,"#dc2626"]],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=12, family="Inter"),
        hovertext=hover, hoverinfo="text",
        colorbar=dict(
            title=dict(text="Rel. Absorbance", font=dict(family="Inter",size=11)),
            tickfont=dict(family="Inter",size=10), thickness=14, len=0.8),
    ))
    fig.update_layout(
        title=dict(text="<b>Spectral Band Absorption Profile  \u00b7  Key Diagnostic Bands by Tissue Class</b>",
                   font=dict(size=13, family="Inter", color="#0f172a"), x=0.5),
        xaxis=dict(tickfont=dict(family="Inter", size=10), tickangle=-30),
        yaxis=dict(tickfont=dict(family="Inter", size=11)),
        margin=dict(l=160, r=90, t=60, b=90), height=250,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="Inter"),
    )
    return fig


def build_validation_radar():
    """Radar chart of real per-class IoU scores from model evaluation."""
    cats = ["Background", "Benign\nEpithelium", "Benign\nStroma", "Malignant\nStroma", "Mean IoU"]
    vals = [92.6, 65.7, 96.1, 91.9, 86.6]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(0,61,130,0.15)", line=dict(color="#003d82", width=2.5),
        name="QCL U-Net (real inference)",
        hovertemplate="<b>%{theta}</b><br>IoU: %{r:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[80]*6, theta=cats + [cats[0]], fill="none",
        line=dict(color="#dc2626", width=1.5, dash="dash"),
        name="Reference threshold (80%)", hoverinfo="skip",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[60,100], ticksuffix="%",
                            tickfont=dict(family="Inter", size=9)),
            angularaxis=dict(tickfont=dict(family="Inter", size=11)),
            bgcolor="rgba(255,255,255,0.5)",
        ),
        legend=dict(orientation="h", y=-0.20, font=dict(family="Inter", size=10)),
        margin=dict(l=50, r=50, t=20, b=70), height=340,
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
    )
    return fig


def build_spero_comparison():
    """2-panel: acquisition time & spectral throughput — SPERO-QT vs FTIR vs Raman."""
    TECHS  = ["Raman Microscopy", "FTIR Microscope", "SPERO-QT 340"]
    COLORS = ["#cbd5e1", "#94a3b8", "#003d82"]
    times   = [18_000, 6_000, 40]       # seconds — 450 spectral bands
    pts_sec = [500, 50_000, 7_000_000]  # spectral points per second

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), facecolor="white")
    fig.subplots_adjust(left=0.20, right=0.95, wspace=0.55, top=0.80, bottom=0.26)

    panels = [
        (axes[0], times,   "Acquisition time (s)  [log scale]",
         "Full image cube  —  450 spectral bands",
         ["18,000 s", "6,000 s", "40 s"]),
        (axes[1], pts_sec, "Spectral points per second  [log scale]",
         "Measurement throughput",
         ["500 pts/s", "50K pts/s", "7M pts/s"]),
    ]
    for ax, vals, xlabel, title, bar_labels in panels:
        bars = ax.barh(TECHS, vals, color=COLORS, height=0.50,
                       edgecolor="white", linewidth=0.0)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=8.5, color="#475569", labelpad=6)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#0f172a", pad=8)
        for bar, lbl in zip(bars, bar_labels):
            ax.text(bar.get_width() * 1.2, bar.get_y() + bar.get_height() / 2,
                    lbl, va="center", ha="left", fontsize=8.5, fontweight="bold",
                    color=bar.get_facecolor())
        ax.tick_params(axis="y", labelsize=9, colors="#0f172a")
        ax.tick_params(axis="x", labelsize=7.5, colors="#64748b")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#e2e8f0")
        ax.set_facecolor("#f8fafc")

    # Annotation on panel 1: highlight 150× advantage
    axes[0].set_xlim(8, 600_000)
    axes[0].text(70, 1.5, "← 150\u00d7 faster than FTIR",
                 va="center", ha="left", fontsize=8.5,
                 color="#003d82", fontweight="bold")
    axes[1].set_xlim(200, 80_000_000)

    fig.text(0.5, 0.04,
             ("Sources: Daylight Solutions Spero-QT 340 product sheet (2024)  \u00b7  "
              "FTIR estimate: equal SNR basis per Daylight Solutions (150\u00d7 factor)  \u00b7  "
              "Raman: point-scan estimate for 500 \u00b5m tissue section at 1 \u00b5m step"),
             ha="center", fontsize=6.5, color="#94a3b8")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── Pre-compute static images once at startup ──────────────────────────────────
print("[QCL] Generating reference images…")
_IMG_PIPELINE = build_pipeline_image()
_IMG_ATLAS    = build_tissue_atlas()
_IMG_UNET     = build_unet_diagram()
_IMG_SPERO    = build_spero_comparison()
_FIG_SPECTRAL = build_spectral_figure()
_FIG_HEATMAP  = build_spectral_heatmap()
_FIG_RADAR    = build_validation_radar()
print("[QCL] Reference images ready.")


# ── Pre-analysis layout ───────────────────────────────────────────────────────
def pre_analysis_content():

    # ── INTRO BANNER ──────────────────────────────────────────────────────
    intro = html.Div([
        html.P([
            "\u201cMid-infrared spectroscopic imaging resolves tissue biochemistry at 10 \u03bcm resolution \u2014 ",
            html.Strong("without staining, without subjectivity, without waiting."),
            "\u201d",
        ]),
        html.Cite(
            "QCL Clinical Diagnostics Platform  \u00b7  Breast Cancer AI Research",
            style={"color":"rgba(255,255,255,0.72)","fontSize":"0.8rem"},
        ),
    ], className="highlight-quote")

    # ── AT-A-GLANCE HERO STATS ────────────────────────────────────────────
    hero_stats = dbc.Row([
        dbc.Col(html.Div([
            html.Div("2.3 M", className="metric-value", style={"color":"#dc2626"}),
            html.Div("New breast cancer cases / year", className="metric-label"),
            html.Div("WHO 2022  \u00b7  most prevalent cancer globally", className="metric-sub"),
        ], className="metric-card", style={"--accent":"#dc2626"}), width=3),
        dbc.Col(html.Div([
            html.Div("~20%", className="metric-value", style={"color":"#b45309"}),
            html.Div("Inter-pathologist variability", className="metric-label"),
            html.Div("H&E alone \u00b7  subjective interpretation", className="metric-sub"),
        ], className="metric-card", style={"--accent":"#b45309"}), width=3),
        dbc.Col(html.Div([
            html.Div("96.5%", className="metric-value", style={"color":"#0f7c3d"}),
            html.Div("Pixel Accuracy", className="metric-label"),
            html.Div("Mean IoU 86.6%  \u00b7  Malignant IoU 91.9%", className="metric-sub"),
        ], className="metric-card", style={"--accent":"#0f7c3d"}), width=3),
        dbc.Col(html.Div([
            html.Div("< 2.3 min", className="metric-value", style={"color":"#003d82"}),
            html.Div("Full biopsy AI analysis", className="metric-label"),
            html.Div("End-to-end  \u00b7  vs. 24\u201348 h traditional workup", className="metric-sub"),
        ], className="metric-card", style={"--accent":"#003d82"}), width=3),
    ], className="g-3", style={"marginBottom":"2rem"})

    # ── SEC 01  THE CLINICAL CHALLENGE & OUR ANSWER ───────────────────────
    sec01 = html.Div([
        html.Span("01  The Problem", className="num-label"),
        html.H2("Why Conventional Breast Cancer Diagnosis Needs Reinvention",
                className="section-title", style={"marginTop":"0.4rem"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("The Diagnostic Status Quo", className="card-title"),
                html.P([
                    "Breast cancer diagnosis has relied on ",
                    html.Strong("haematoxylin & eosin (H&E) histopathology"),
                    " for over a century. A pathologist visually interprets tissue morphology \u2014 a process that is slow, "
                    "labour-intensive, and fundamentally subjective.",
                ]),
                html.Ul([
                    html.Li([html.Strong("Subjectivity: "), "Inter-observer disagreement reaches 20% on borderline cases, "
                             "with significant variability in tumour grading (Elmore et al., JAMA 2015)"]),
                    html.Li([html.Strong("Time: "), "Standard tissue processing and H&E workup takes 24\u201348 hours; "
                             "intraoperative frozen sections introduce artefacts that reduce accuracy"]),
                    html.Li([html.Strong("Molecular blindness: "), "H&E captures morphology only \u2014 it cannot directly measure "
                             "protein conformation, nucleic acid density, or lipid membrane composition"]),
                    html.Li([html.Strong("Staining artefacts: "), "Fixation, embedding and staining protocols introduce "
                             "batch-to-batch variability that confounds quantitative scoring"]),
                    html.Li([html.Strong("Access: "), "Expert breast pathologists are concentrated in academic centres; "
                             "community hospitals often lack specialist second-opinion capacity"]),
                ]),
            ], className="clinical-card", style={"height":"100%",
                                                  "borderLeft":"4px solid #dc2626"}), width=6),
            dbc.Col(html.Div([
                html.Div("Our Answer: QCL + AI", className="card-title"),
                html.P([
                    "We combine ",
                    html.Strong("Quantum Cascade Laser (QCL) hyperspectral imaging"),
                    " with a deep learning segmentation model to produce an ",
                    html.Strong("objective, stain-free, pixel-level tissue map"),
                    " from a standard FFPE biopsy section \u2014 in under 2.3 minutes.",
                ]),
                html.Ul([
                    html.Li([html.Strong("Label-free: "), "QCL illuminates tissue in the mid-infrared; "
                             "molecular bonds absorb energy at unique wavenumbers \u2014 no dyes required"]),
                    html.Li([html.Strong("Objective: "), "AI posterior probabilities replace visual scoring; "
                             "the same model produces identical outputs for identical inputs, always"]),
                    html.Li([html.Strong("Quantitative: "), "Per-pixel class composition, malignant tissue percentage, "
                             "and a structured risk score are generated automatically"]),
                    html.Li([html.Strong("Reproducible: "), "A single calibrated SPERO system can serve multiple sites "
                             "without loss of inter-site consistency"]),
                    html.Li([html.Strong("Open-access validated: "), "207 patient cases from the Zenodo "
                             "public dataset (Kröger-Lui et al., Analytical Chemistry 2017, CC BY 4.0 licence)"]),
                ]),
                dbc.Alert([
                    html.Strong("Key distinction: "),
                    "The AI classifies tissue by ",
                    html.Em("molecular spectral signature"),
                    ", not visual appearance. Changes in protein secondary structure and DNA compaction "
                    "that precede morphological transformation are detected before they are visible to the human eye.",
                ], color="primary", style={"fontSize":"0.88rem","marginTop":"1rem",
                                           "borderRadius":"10px","padding":"0.9rem 1.1rem"}),
            ], className="clinical-card", style={"height":"100%",
                                                  "borderLeft":"4px solid #0f7c3d"}), width=6),
        ], className="g-3"),
        # Pipeline image comes HERE — after the visitor understands why it exists
        html.P("From biopsy slide to AI-generated risk map — the end-to-end pipeline:",
               style={"color":"#475569","fontSize":"0.92rem","margin":"1.5rem 0 0.5rem",
                      "fontWeight":"600","textAlign":"center"}),
        html.Div(html.Img(src=_IMG_PIPELINE, style={"width":"100%","borderRadius":"8px"}),
                 className="img-container"),
    ])

    # ── SEC 02 INSTRUMENT ─────────────────────────────────────────────────
    feature_icons = html.Div([
        html.Div([html.Span("\U0001f6ab",className="fi-ico"),html.Span("Label-free",className="fi-name"),html.Span("No dyes or stains",className="fi-sub")],className="fi-pill"),
        html.Div([html.Span("\U0001f3af",className="fi-ico"),html.Span("< 5 \u03bcm",className="fi-name"),html.Span("Diffraction-limited (0.7 NA)",className="fi-sub")],className="fi-pill"),
        html.Div([html.Span("\u26a1",   className="fi-ico"),html.Span("< 40 s",className="fi-name"),html.Span("450 absorbance images",className="fi-sub")],className="fi-pill"),
        html.Div([html.Span("\U0001f30a",className="fi-ico"),html.Span("950\u20131800 cm\u207b\u00b9",className="fi-name"),html.Span("MIR fingerprint region",className="fi-sub")],className="fi-pill"),
        html.Div([html.Span("\U0001f4f7",className="fi-ico"),html.Span("480\u00d7480 FPA",className="fi-name"),html.Span("Uncooled microbolometer",className="fi-sub")],className="fi-pill"),
        html.Div([html.Span("\U0001f9ca",className="fi-ico"),html.Span("No cryo-cooling",className="fi-name"),html.Span("vs. synchrotron FTIR",className="fi-sub")],className="fi-pill"),
    ], className="fi-row")

    sec01_instrument = html.Div([
        html.Span("02  The Instrument", className="num-label"),
        html.H2("Spero-QT 340  \u2014  World\u2019s First Wide-Field QCL-IR Microscope",
                className="section-title", style={"marginTop":"0.4rem"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("\U0001f4e1 QCL Hyperspectral Imaging Physics", className="card-title"),
                html.P(["QCL hyperspectral imaging interrogates tissue in the ",
                        html.Strong("mid-infrared fingerprint region (950\u20131800 cm\u207b\u00b9)"),
                        ", where covalent bonds of biological macromolecules produce unique absorption signatures. "
                        "Unlike H&E histology, which relies on visual pattern recognition, "
                        "QCL captures the underlying molecular composition directly \u2014 enabling objective, reproducible tissue classification."]),
                html.P(["The SPERO illuminates the entire field of view simultaneously at each wavelength. "
                        "Every pixel captures a Beer-Lambert absorbance spectrum \u2014 ",
                        html.Strong("directly quantitative"),
                        " by first principles, with no special normalisation or reference samples required."]),
                html.P(["Particularly sensitive to ",
                        html.Strong("conformational protein changes, DNA compaction, and lipid membrane remodelling"),
                        " \u2014 hallmarks of malignant transformation that often precede morphological changes "
                        "visible to the naked eye."]),
            ], className="clinical-card", style={"height":"100%"}), width=6),
            dbc.Col(html.Div([
                html.Div("\U0001f6e0\ufe0f Spero-QT 340  \u00b7  Daylight Solutions (Leonardo DRS)", className="card-title"),
                html.P(["World\u2019s first wide-field QCL-IR microscope (pioneered 2014, now ",
                        html.Strong("3rd generation"),
                        "). Installed in research and clinical laboratories across ",
                        html.Strong("12+ countries"),
                        ". Dataset used here: Kröger-Lui et al., Analytical Chemistry 2017 \u2014 Zenodo DOI 10.5281/zenodo.808456, CC BY 4.0."]),
                html.Ul([
                    html.Li([html.Strong("Camera: "),
                             "480 \u00d7 480 uncooled microbolometer focal-plane array \u2014 no liquid nitrogen required"]),
                    html.Li([html.Strong("Spectral range: "),
                             "950\u20131800 cm\u207b\u00b9 std \u00b7 customisable 800\u20132300 cm\u207b\u00b9 \u00b7 down to 2 cm\u207b\u00b9 resolution"]),
                    html.Li([html.Strong("High-res mode (0.7 NA): "),
                             "1.3 \u03bcm pixel \u00b7 < 5 \u03bcm spatial resolution \u00b7 650\u00d7650 \u03bcm FOV \u00b7 > 8 mm WD"]),
                    html.Li([html.Strong("Wide-field mode (0.3 NA): "),
                             "4.3 \u03bcm pixel \u00b7 < 12 \u03bcm spatial resolution \u00b7 2\u00d72 mm FOV \u00b7 > 25 mm WD"]),
                    html.Li([html.Strong("Acquisition speed: "),
                             "< 40 s for 450 absorbance images at 2 cm\u207b\u00b9 spacing \u2014 ",
                             html.Strong("150\u00d7 faster than conventional FTIR at equal SNR")]),
                    html.Li([html.Strong("Throughput: "),
                             "> 7 million spectral points per second"]),
                    html.Li([html.Strong("Software: "),
                             "ChemVision\u2122 \u2014 exports to MATLAB and ENVI format"]),
                    html.Li([html.Strong("Sample: "),
                             "FFPE sections \u00b7 4\u20138 \u03bcm thick \u00b7 barium fluoride substrate"]),
                ]),
            ], className="clinical-card", style={"height":"100%"}), width=6),
        ], className="g-3"),
        feature_icons,
        # ── SPERO Performance Comparison ─────────────────────────────────
        html.P("SPERO-QT 340 vs. conventional hyperspectral imaging \u2014 acquisition speed & spectral throughput:",
               style={"color":"#475569","fontSize":"0.92rem","margin":"1.5rem 0 0.4rem",
                      "fontWeight":"600","textAlign":"center"}),
        html.Div(html.Img(src=_IMG_SPERO, style={"width":"100%","borderRadius":"8px"}),
                 className="img-container"),
        # ── Why SPERO is Unique ───────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("\U0001f3c6 Why SPERO is in a Class of Its Own", className="card-title"),
                html.Ul([
                    html.Li([html.Strong("First of its kind (2014): "),
                             "The Spero was the world\u2019s first wide-field QCL-IR microscope. "
                             "Unlike point-scanning FTIR, it captures the entire field at once \u2014 no scanning, no compromises."]),
                    html.Li([html.Strong("150\u00d7 faster than FTIR at equal SNR: "),
                             "QCL illuminates the full FOV at a single wavelength; FTIR divides broadband photons across all wavelengths simultaneously \u2014 "
                             "QCL concentrates all photons where needed, yielding far greater spectral irradiance."]),
                    html.Li([html.Strong("No cryogenic cooling: "),
                             "The 480\u00d7480 uncooled microbolometer FPA operates at room temperature. "
                             "Conventional MCT FTIR detectors require liquid nitrogen \u2014 making SPERO far more practical in clinical settings."]),
                    html.Li([html.Strong("Proprietary coherence control: "),
                             "Daylight Solutions\u2019 patented technology suppresses speckle and fringe artefacts inherent to coherent laser sources \u2014 "
                             "without digital post-processing. Signal is physically clean."]),
                    html.Li([html.Strong("Directly quantitative output: "),
                             "Beer-Lambert absorbance is directly proportional to molecular concentration. "
                             "Unlike Raman (cross-sections vary by orientation) or photothermal IR (requires reference), "
                             "SPERO data is quantitative by definition."]),
                    html.Li([html.Strong("Faster than Raman & Photothermal by orders of magnitude: "),
                             "No fluorescence background, no photodegradation, no bleaching. "
                             "> 7 million spectral points per second enables full-tissue chemical maps in under a minute."]),
                ]),
            ], className="clinical-card"), width=12),
        ], className="g-3 mt-2"),
    ])

    # ── SEC 03 THE SIGNAL ──────────────────────────────────────────────────
    band_tags = [
        ("1650 cm\u207b\u00b9","#003d82"),("1550 cm\u207b\u00b9","#475569"),
        ("1450 cm\u207b\u00b9","#7c3aed"),("1340 cm\u207b\u00b9","#0f7c3d"),
        ("1240 cm\u207b\u00b9","#be185d"),("1080 cm\u207b\u00b9","#b45309"),
    ]
    biomarkers_card = html.Div([
        html.Div("\U0001f9ec Key Spectral Biomarkers", className="card-title"),
        html.Div([
            html.Span(wv, className="band-tag",
                      style={"background":col+"1a","color":col,"border":f"1px solid {col}55"})
            for wv, col in band_tags
        ]),
        html.Hr(style={"margin":"0.8rem 0","borderColor":"#e2e8f0"}),
        html.Ul([
            html.Li([html.Strong("Amide I (1600\u20131700 cm\u207b\u00b9): "),"\u03b2-sheet protein content \u2192 malignant desmoplastic remodelling"]),
            html.Li([html.Strong("Amide II (1510\u20131560 cm\u207b\u00b9): "),"N-H bending; Amide I/II ratio marks structural denaturation"]),
            html.Li([html.Strong("Phosphate (1200\u20131260 cm\u207b\u00b9): "),"DNA/RNA density \u2192 elevated nuclear-cytoplasm ratio in malignancy"]),
            html.Li([html.Strong("C-O stretch (1000\u20131100 cm\u207b\u00b9): "),"Carbohydrate/glycogen; depleted in Warburg-effect tumour cores"]),
            html.Li([html.Strong("Lipid CH\u2082 (1350\u20131470 cm\u207b\u00b9): "),"Membrane remodelling; altered ratio = metabolic shift marker"]),
            html.Li([html.Strong("Collagen (1330\u20131350 cm\u207b\u00b9): "),"Dense desmoplastic stroma \u2014 hallmark of invasive carcinoma"]),
        ]),
    ], className="clinical-card", style={"height":"100%"})

    sec03 = html.Div([
        html.Span("03  The Signal", className="num-label"),
        html.H2("Molecular Spectral Fingerprints", className="section-title", style={"marginTop":"0.4rem"}),
        dbc.Row([
            dbc.Col(biomarkers_card, width=5),
            dbc.Col(dcc.Graph(
                figure=_FIG_SPECTRAL,
                config={"displayModeBar":True,"modeBarButtonsToRemove":["lasso2d","select2d"]},
                style={"height":"420px"},
            ), width=7),
        ], className="g-3"),
        html.P("Representative spectra (illustrative) — Gaussian profiles centred on real diagnostic bands. "
               "Hover any point for values. Dashed verticals mark the primary diagnostic absorption bands.",
               style={"color":"#64748b","fontSize":"0.83rem","marginTop":"0.4rem"}),
        html.Div([
            html.H4("Spectral Band Heatmap",
                    style={"fontSize":"1rem","fontWeight":"700","color":"#0f172a","marginBottom":"0.5rem"}),
            dcc.Graph(figure=_FIG_HEATMAP, config={"displayModeBar":False}),
            html.P("Normalised 0\u20131 across bands. Malignant Stroma shows characteristically elevated Amide I and Phosphate signals.",
                   style={"color":"#64748b","fontSize":"0.8rem","marginTop":"0.35rem","textAlign":"center"}),
        ], className="clinical-card"),
    ])

    # ── SEC 04 TISSUE ATLAS ─────────────────────────────────────────────────
    tissue_info = [
        (CLASS_NAMES[0],CLASS_COLOURS[0],CLASS_MARKERS[0],
         "Acellular adipose tissue, stromal debris, non-viable regions.","Not scored","#64748b"),
        (CLASS_NAMES[1],CLASS_COLOURS[1],CLASS_MARKERS[1],
         "Organised ductal/lobular epithelium. Regular nuclei, lumen formation, low Amide I.","Benign","#0f7c3d"),
        (CLASS_NAMES[2],CLASS_COLOURS[2],CLASS_MARKERS[2],
         "Normal fibrous stroma. Wavy collagen bundles, spindle fibroblasts, high collagen/protein ratio.","Benign","#2563eb"),
        (CLASS_NAMES[3],CLASS_COLOURS[3],CLASS_MARKERS[3],
         "Desmoplastic reactive stroma surrounding invasive carcinoma. Dense irregular nuclei, elevated Amide I and phosphate bands.","\u26a0\ufe0f Malignant","#dc2626"),
    ]
    tissue_rows = html.Div([
        html.Div([
            html.Div(marker, style={"width":"46px","height":"46px","borderRadius":"10px",
                                    "flexShrink":"0","background":col,"marginTop":"0.05rem",
                                    "display":"flex","alignItems":"center",
                                    "justifyContent":"center","fontSize":"1.4rem","color":"white"}),
            html.Div([
                html.Div(name, className="tissue-name", style={"color":col}),
                html.P(desc, className="tissue-desc"),
            ], style={"flex":"1"}),
            html.Span(lbl, style={"fontSize":"0.72rem","fontWeight":"700",
                                  "padding":"0.3rem 0.7rem","borderRadius":"6px",
                                  "background":lcol+"1a","color":lcol,
                                  "border":f"1px solid {lcol}44","flexShrink":"0",
                                  "alignSelf":"flex-start"}),
        ], className="tissue-row")
        for name, col, marker, desc, lbl, lcol in tissue_info
    ])

    sec04 = html.Div([
        html.Span("04  Tissue Atlas", className="num-label"),
        html.H2("What the AI Sees", className="section-title", style={"marginTop":"0.4rem"}),
        html.P("The model classifies tissue by spectral signature, not morphology. "
               "These synthetic panels show the expected H&E-equivalent morphology of each class "
               "as an orientation reference.",
               style={"color":"#475569","marginBottom":"1rem"}),
        html.Div(html.Img(src=_IMG_ATLAS, style={"width":"100%","borderRadius":"8px"}),
                 className="img-container"),
        tissue_rows,
    ])

    # ── SEC 05 THE AI MODEL ────────────────────────────────────────────────
    per_class_data = [
        {"Class":"Background",        "Precision":"98.4%","Recall":"94.0%","F1":"96.1%","IoU":"0.926"},
        {"Class":"Benign Epithelium † ","Precision":"100.0%","Recall":"65.7%","F1":"79.3%","IoU":"0.657"},
        {"Class":"Benign Stroma",      "Precision":"97.7%","Recall":"98.4%","F1":"98.0%","IoU":"0.961"},
        {"Class":"Malignant Stroma",   "Precision":"92.7%","Recall":"99.1%","F1":"95.8%","IoU":"0.919"},
    ]
    split_data = [
        {"Split":"Full cohort (Zenodo)","Patients":"207","Cubes":"207 raw .mat","Annotation":"Kröger-Lui et al. 2017"},
        {"Split":"Processed & used",   "Patients":"20", "Cubes":"20 (480×480×15 PCA)","Annotation":"Used in this app"},
    ]
    sec05_model = html.Div([
        html.Span("05  The AI Model", className="num-label"),
        html.H2("Dataset, Training & U-Net Architecture", className="section-title",
                style={"marginTop":"0.4rem"}),

        # ── Dataset subsection ───────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("\U0001f9ec Dataset — Zenodo Open-Access (CC BY 4.0)", className="card-title"),
                html.Div([
                    html.Span("\u2705  Real open-access data: ", style={"fontWeight":"700","color":"#0f7c3d"}),
                    html.Span("This app loads "),
                    html.Strong("real QCL hyperspectral cubes"),
                    html.Span(" downloaded from Zenodo (10.5281/zenodo.808456). "
                              "The dataset is CC BY 4.0 \u2014 freely usable with citation. "
                              "No IRB or DUA required."),
                ], style={"background":"#f0fdf4","border":"1px solid #86efac","borderRadius":"8px",
                           "padding":"0.6rem 0.9rem","fontSize":"0.83rem","marginBottom":"0.8rem",
                           "color":"#14532d"}),
                html.P(["Source: Kr\u00f6ger-Lui, N. et al. \u201cQuantum cascade laser-based "
                        "hyperspectral imaging of biological tissue.\u201d ",
                        html.Em("Analytical Chemistry"),
                        " 2017. ",
                        html.A("DOI: 10.5281/zenodo.808456",
                               href="https://doi.org/10.5281/zenodo.808456",
                               target="_blank"),
                        ". Instrument: Daylight Solutions Spero\u00ae QCL IR Microscope."]),
                html.Ul([
                    html.Li([html.Strong("Cohort: "), "207 unique patients \u00b7 breast cancer TMA \u00b7 fully anonymised (GDPR/DSGVO)"]),
                    html.Li([html.Strong("Cubes used: "), "20 real QCL cubes processed from the 25 GB Zenodo download"]),
                    html.Li([html.Strong("Cube size: "), "480 \u00d7 480 pixels "
                             "(SPERO-QT 480\u00d7480 FPA \u00b7 4.3 \u03bcm/pixel \u00b7 ~2\u00d72 mm FOV)"]),
                    html.Li([html.Strong("Spectral channels: "),
                             "Raw MIR (912\u20131800 cm\u207b\u00b9) \u00b7 223 bands "
                             "\u2192 15 PCA components (run_pipeline.py)"]),
                    html.Li([html.Strong("Tissue sections: "), "FFPE \u00b7 barium fluoride substrate"]),
                    html.Li([html.Strong("Classes: "),
                             "4 \u2014 Background, Benign Epithelium, Benign Stroma, Malignant Stroma"]),
                    html.Li([html.Strong("Pre-processing: "),
                             "Absorbance clipping to [\u22120.1, 1.0] (sensor artefact suppression) "
                             "\u2192 StandardScaler (zero mean, unit variance per band) "
                             "\u2192 15-component PCA (>99.1\u202f% variance retained \u2014 run_pipeline.py)"]),
                    html.Li([html.Strong("License: "),
                             html.A("CC BY 4.0",
                                    href="https://creativecommons.org/licenses/by/4.0/",
                                    target="_blank"),
                             " \u2014 auto-download via ",
                             html.Code("python 01_data_ingestion.py")]),
                ]),
            ], className="clinical-card", style={"height":"100%"}), width=7),
            dbc.Col(html.Div([
                html.Div("\U0001f4cb Published Dataset Split", className="card-title"),
                dash_table.DataTable(
                    data=split_data,
                    columns=[{"name":c,"id":c} for c in split_data[0].keys()],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor":"#f1f5f9","fontWeight":"700",
                                  "color":"#0f172a","fontSize":"0.83rem"},
                    style_data={"color":"#0f172a","fontSize":"0.83rem"},
                    style_cell={"fontFamily":"Inter, sans-serif","padding":"8px 10px"},
                    style_data_conditional=[
                        {"if":{"row_index":"odd"},"backgroundColor":"#fafbfc"},
                        {"if":{"filter_query":"{Split} = 'Test (hold-out)'"},
                         "fontWeight":"700","color":"#003d82"},
                    ],
                ),
                html.Br(),
                html.Div("\u26a0\ufe0f  Class balance (from published paper)", className="card-title",
                         style={"fontSize":"0.82rem"}),
                html.Ul([
                    html.Li("Background: ~38% of pixels"),
                    html.Li("Benign Epithelium: ~22% of pixels"),
                    html.Li("Benign Stroma: ~28% of pixels"),
                    html.Li([html.Strong("Malignant Stroma: ~12% of pixels"),
                             " \u2014 minority class; addressed with class weight \u00d72.5 on Malignant Stroma"]),
                ], style={"fontSize":"0.82rem","color":"#475569"}),
                html.P("\u2139\ufe0f  In this demo, interactive samples are generated synthetically "
                       "to replicate these class proportions and spectral characteristics.",
                       style={"fontSize":"0.75rem","color":"#64748b","marginTop":"0.4rem",
                              "fontStyle":"italic"}),
            ], className="clinical-card", style={"height":"100%"}), width=5),
        ], className="g-3"),

        # ── Training pipeline ────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("\u2699\ufe0f Training Pipeline (run_training.py \u2014 fully reproducible open-source)", className="card-title"),
                dbc.Row([
                    dbc.Col(html.Ul([
                        html.Li([html.Strong("Architecture: "),
                                 "HyperspectralUNet \u2014 custom 3-level encoder-decoder (03_spatial_cnn_segmentation.py). "
                                 "No pretrained backbone. Filter counts 64 \u2192 128 \u2192 256 \u2192 512. "
                                 "Skip connections preserve spatial detail at each resolution scale."]),
                        html.Li([html.Strong("Input: "),
                                 "15-channel PCA cube (64\u00d764\u00d715 patches, random crop). "
                                 "No RGB conversion \u2014 spectral channels fed directly as multi-band input."]),
                        html.Li([html.Strong("Loss function: "),
                                 "CrossEntropy (class weights [0.5, 1.0, 1.0, 2.5]) + soft Dice coefficient. "
                                 "Up-weights malignant stroma (class 3) to compensate for class imbalance."]),
                        html.Li([html.Strong("Optimiser: "),
                                 "AdamW (lr = 1\u00d710\u207b\u2074, weight decay = 1\u00d710\u207b\u2074). "
                                 "CosineAnnealingLR scheduler (T_max = 50 epochs)."]),
                        html.Li([html.Strong("Epochs: "),
                                 "50 configured \u00b7 early stopping patience = 10 epochs. "
                                 "Best checkpoint at epoch 18 (val loss 0.1699, mean IoU 0.866)."]),
                    ], style={"fontSize":"0.84rem","color":"#334155"}), width=6),
                    dbc.Col(html.Ul([
                        html.Li([html.Strong("Batch size: "),
                                 "8 patches per step (no gradient accumulation)."]),
                        html.Li([html.Strong("Data augmentation: "),
                                 "Random horizontal flip, random vertical flip. "
                                 "No rotation, no spectral jitter \u2014 spectral values are physically meaningful."]),
                        html.Li([html.Strong("Data split: "),
                                 "80/20 chronological split (16 train / 4 val cubes). "
                                 "No patient-level leakage \u2014 cubes are independent tissue micro-array cores."]),
                        html.Li([html.Strong("Hardware: "),
                                 "Intel Core i9 (local workstation, CPU training). Seed 42 fixed for reproducibility."]),
                        html.Li([html.Strong("Post-processing: "),
                                 "Per-pixel argmax on softmax logits \u2192 class assignment. "
                                 "No thresholding or morphological filtering applied."]),
                        html.Li([html.Strong("Source code: "),
                                 "run_training.py + 03_spatial_cnn_segmentation.py \u2014 included in repository."]),
                    ], style={"fontSize":"0.84rem","color":"#334155"}), width=6),
                ]),
            ], className="clinical-card"), width=12),
        ], className="g-3 mt-1"),

        # ── U-Net diagram + per-class table ─────────────────────────────
        html.P("U-Net encoder-decoder with skip connections \u2014 each pixel classified independently:",
               style={"color":"#475569","fontSize":"0.92rem","margin":"1.5rem 0 0.4rem",
                      "fontWeight":"600","textAlign":"center"}),
        html.Div(html.Img(src=_IMG_UNET, style={"width":"100%","borderRadius":"8px"}),
                 className="img-container"),
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("\U0001f9e0 Model Design Rationale", className="card-title"),
                html.P([html.Strong("Why U-Net for spectral imaging?"),
                        " The encoder-decoder with skip connections preserves fine spatial resolution "
                        "during downsampling \u2014 critical for accurate tissue boundary delineation at 4.3 \u03bcm/pixel. "
                        "The custom 1\u00d71 spectral projection in enc1 acts as a learned dimensionality reduction "
                        "across the 15 PCA channels before spatial feature extraction in enc2/enc3."]),
                html.P([html.Strong("Class imbalance strategy:"),
                        " Malignant stroma comprises only ~12% of pixels on average but is the "
                        "clinically critical class. CrossEntropy with class weight 2.5\u00d7 for malignant stroma "
                        "and 0.5\u00d7 for background, combined with soft Dice loss, forces the model "
                        "to focus learning capacity on ambiguous malignant-stroma boundaries."]),
                html.P([html.Strong("Spectral PCA rationale:"),
                        " Raw QCL channels are highly correlated. 15 PCA components "
                        "capture the dominant spectral variance while reducing input dimensionality, "
                        "accelerating training and avoiding overfitting on the 20-cube dataset."]),
            ], className="clinical-card"), width=7),
            dbc.Col(html.Div([
                html.Div("\U0001f4ca Per-Class Performance (Hold-out)", className="card-title"),
                dash_table.DataTable(
                    data=per_class_data,
                    columns=[{"name":c,"id":c} for c in per_class_data[0].keys()],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor":"#f1f5f9","fontWeight":"700",
                                  "color":"#0f172a","fontSize":"0.85rem"},
                    style_data={"color":"#0f172a","fontSize":"0.85rem"},
                    style_cell={"fontFamily":"Inter, sans-serif","padding":"8px 12px"},
                    style_data_conditional=[
                        {"if":{"row_index":"odd"},"backgroundColor":"#fafbfc"},
                        {"if":{"filter_query":"{Class} = 'Malignant Stroma'"},
                         "color":"#dc2626","fontWeight":"700"},
                    ],
                ),
                html.P("IoU = Intersection over Union (Jaccard). Real inference on all 20 cubes "
                       "vs. expert pathologist labels (Kröger-Lui et al. 2017, CC BY 4.0).",
                       style={"fontSize":"0.73rem","color":"#64748b","marginTop":"0.5rem"}),
            ], className="clinical-card"), width=5),
        ], className="g-3"),
    ])

    # ── SEC 06 CLINICAL EVIDENCE ───────────────────────────────────────────
    stat_pills = html.Div([
        html.Div([html.Span("207",   className="sp-num",style={"color":"#0f7c3d"}),html.Span("Cohort patients",     className="sp-lbl")],className="stat-pill",style={"--accent":"#0f7c3d"}),
        html.Div([html.Span("20",    className="sp-num",style={"color":"#003d82"}),html.Span("Cubes on disk",       className="sp-lbl")],className="stat-pill",style={"--accent":"#003d82"}),
        html.Div([html.Span("96.5%", className="sp-num",style={"color":"#0066cc"}),html.Span("Pixel accuracy",      className="sp-lbl")],className="stat-pill",style={"--accent":"#0066cc"}),
        html.Div([html.Span("0.919", className="sp-num",style={"color":"#7c3aed"}),html.Span("Malignant IoU",       className="sp-lbl")],className="stat-pill",style={"--accent":"#7c3aed"}),
        html.Div([html.Span("86.6%", className="sp-num",style={"color":"#dc2626"}),html.Span("Mean IoU",            className="sp-lbl")],className="stat-pill",style={"--accent":"#dc2626"}),
        html.Div([html.Span("18",    className="sp-num",style={"color":"#0891b2"}),html.Span("Best epoch (val loss)",className="sp-lbl")],className="stat-pill",style={"--accent":"#0891b2"}),
    ], className="stat-pills-row")

    val_table = [
        {"Metric":"Pixel Accuracy (all classes)", "Value":"96.5%","Notes":"Confusion matrix · 4 val tiles (017–020)"},
        {"Metric":"Mean IoU (4 classes)",         "Value":"86.6%","Notes":"Jaccard index, pixel-weighted average"},
        {"Metric":"Malignant Stroma IoU",         "Value":"91.9%","Notes":"Primary clinical class"},
        {"Metric":"Malignant Stroma Recall",      "Value":"99.1%","Notes":"Very few false negatives"},
        {"Metric":"Malignant Stroma Precision",   "Value":"92.7%","Notes":"Low false-positive rate"},
        {"Metric":"Malignant Stroma Dice/F1",     "Value":"95.8%","Notes":"Harmonic mean Prec+Rec"},
        {"Metric":"Best val loss epoch",          "Value":"Ep. 18","Notes":"val_loss=0.1699 (28 total epochs)"},
    ]
    sec06 = html.Div([
        html.Span("06  Model Performance", className="num-label"),
        html.H2("Real Training Results", className="section-title", style={"marginTop":"0.4rem"}),
        html.P("Trained on 20 real QCL cubes (480\u00d7480\u00d715) from the Zenodo open-access "
               "207-patient cohort (Kr\u00f6ger-Lui et al., 2017). Metrics from best validation "
               "checkpoint (epoch 18 / val loss 0.1699). "
               "\u2020 Benign Epithelium IoU reflects low class frequency in the 4 val tiles (\u22480.5% of pixels).",
               style={"color":"#475569","marginBottom":"0.5rem"}),
        stat_pills,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_FIG_RADAR, config={"displayModeBar":False}), width=5),
            dbc.Col(html.Div([
                html.Div("\U0001f4cb Training Performance Summary", className="card-title"),
                dash_table.DataTable(
                    data=val_table,
                    columns=[{"name":c,"id":c} for c in val_table[0].keys()],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor":"#f1f5f9","fontWeight":"700",
                                  "color":"#0f172a","fontSize":"0.85rem"},
                    style_data={"color":"#0f172a","fontSize":"0.85rem"},
                    style_cell={"fontFamily":"Inter, sans-serif","padding":"8px 10px"},
                    style_data_conditional=[
                        {"if":{"row_index":"odd"},"backgroundColor":"#fafbfc"},
                        {"if":{"column_id":"Value"},"color":"#0f7c3d","fontWeight":"700"},
                    ],
                ),
                html.P("Source: models/training_history.json. Checkpoint saved at best mean IoU. "
                       "Dataset: Zenodo DOI 10.5281/zenodo.808456 \u00b7 CC BY 4.0.",
                       style={"fontSize":"0.73rem","color":"#64748b","marginTop":"0.5rem"}),
            ], className="clinical-card"), width=7),
        ], className="g-3 mt-2"),
    ])

    # ── CTA: READY TO RUN ───────────────────────────────────────────────────
    ck_descs = [
        "Tissue background / non-cellular regions",
        "Normal glandular epithelium \u2014 benign",
        "Normal stromal connective tissue \u2014 benign",
        "Desmoplastic stroma \u2014 malignant indicator",
    ]
    ck_cards = [
        html.Div([
            html.Span(CLASS_MARKERS[i], className="ck-marker", style={"color":CLASS_COLOURS[i]}),
            html.Span(CLASS_NAMES[i],   className="ck-name",   style={"color":CLASS_COLOURS[i]}),
            html.Span(ck_descs[i],      className="ck-desc",   style={"color":CLASS_COLOURS[i]}),
        ], className="ck-card",
           style={"background":CLASS_COLOURS[i]+"18","border":f"2px solid {CLASS_COLOURS[i]}"})
        for i in range(len(CLASS_NAMES))
    ]
    wf_steps = [
        ("1\ufe0f\u20e3", "Select Sample",    "Choose a patient QCL hyperspectral cube from the selector above."),
        ("2\ufe0f\u20e3", "Set Threshold",    "Confidence threshold (default 0.85 = 85% posterior probability required for classification)."),
        ("3\ufe0f\u20e3", "Execute Analysis", "Click \u2018Execute Advanced AI Analysis\u2019 to segment every 10 \u03bcm pixel via U-Net."),
        ("4\ufe0f\u20e3", "Review Results",   "Scroll: tissue maps, risk assessment, molecular insights, structured pathology report."),
    ]
    sec_cta = html.Div([
        html.Span("07  Run the Analysis", className="num-label"),
        html.H2("Classification Key & Workflow", className="section-title", style={"marginTop":"0.4rem"}),
        html.H4("\U0001f3a8 Colour Classification Reference", className="section-subtitle"),
        html.Div(ck_cards, style={"display":"flex","gap":"0.75rem","marginBottom":"1.5rem"}),
        html.H4("\U0001f3e5 Step-by-Step Workflow", className="section-subtitle"),
        dbc.Row([
            dbc.Col(html.Div([
                html.Span(ico,   className="wf-icon"),
                html.Span(title, className="wf-title"),
                html.Span(desc,  className="wf-desc"),
            ], className="wf-step"), width=3)
            for ico, title, desc in wf_steps
        ], className="g-3"),
    ])

    return html.Div([
        # ── INTRO: hook + at-a-glance numbers ──
        html.Hr(className="qcl-divider"),
        html.Div([intro, hero_stats], id="sec-hero"),
        # ── 01: The Problem & our answer (with pipeline) ──
        html.Hr(className="qcl-divider"),
        html.Div(sec01, id="sec-problem"),
        # ── 02: Instrument deep-dive ──
        html.Hr(className="qcl-divider"),
        html.Div(sec01_instrument, id="sec-instrument"),
        # ── 03: Spectral data ──
        html.Hr(className="qcl-divider"),
        html.Div(sec03, id="sec-signal"),
        # ── 04: Tissue atlas ──
        html.Hr(className="qcl-divider"),
        html.Div(sec04, id="sec-atlas"),
        # ── 05: AI model ──
        html.Hr(className="qcl-divider"),
        html.Div(sec05_model, id="sec-model"),
        # ── 06: Validation evidence ──
        html.Hr(className="qcl-divider"),
        html.Div(sec06, id="sec-performance"),
        # ── CTA: Run the analysis (scroll target only — controls are above) ──
        html.Hr(className="qcl-divider"),
        html.Div(sec_cta, id="sec-cta-hint"),
    ])


# ── Post-analysis layout ──────────────────────────────────────────────────────
def post_analysis_content(sample_id, cube, labels, results):
    prediction         = results["prediction"]
    confidence_scores  = results["confidence_scores"]
    composition        = calculate_composition(prediction)
    malignant_pct      = composition["Malignant Stroma"]["percentage"]
    assessment         = get_assessment(malignant_pct)

    # Build pixel-perfect images
    rgb_indices = [0, 1, 2] if cube.shape[2] >= 3 else [0, 0, 0]
    rgb         = np.stack([normalize_hs(cube[:, :, i]) for i in rgb_indices], axis=2)
    rgb_uint8   = (rgb * 255).clip(0, 255).astype(np.uint8)

    palette       = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in CLASS_COLOURS], dtype=np.uint8)
    labels_rgb    = palette[labels.astype(int).clip(0, len(CLASS_COLOURS) - 1)]
    pred_rgb      = palette[prediction.astype(int).clip(0, len(CLASS_COLOURS) - 1)]

    img_orig  = array_to_b64(rgb_uint8)
    img_label = array_to_b64(labels_rgb)
    img_pred  = array_to_b64(pred_rgb)

    # ── Disagreement map: grey where correct, class-coloured at error pixels ──
    differ          = (prediction != labels)
    pixel_agreement = float((~differ).mean()) * 100
    diff_rgb        = np.full((*labels.shape, 3), 225, dtype=np.uint8)   # light grey = correct
    for _c in range(len(CLASS_COLOURS)):
        wrong_c = differ & (prediction == _c)
        diff_rgb[wrong_c] = palette[_c]   # colour = what the AI predicted wrongly
    img_diff = array_to_b64(diff_rgb)

    # Legend
    legend_cards = html.Div([
        html.Div([
            html.Span(CLASS_MARKERS[i], style={"fontSize": "1.8rem", "display": "block", "color": CLASS_COLOURS[i]}),
            html.Span(CLASS_NAMES[i],  style={"fontWeight": "700", "color": CLASS_COLOURS[i]}),
        ], className="legend-card", style={
            "background": CLASS_COLOURS[i] + "22",
            "border": f"2px solid {CLASS_COLOURS[i]}",
        })
        for i in range(len(CLASS_NAMES))
    ], style={"display": "flex", "gap": "0.75rem", "marginTop": "1rem"})

    # Confidence chart
    fig_conf = go.Figure(go.Bar(
        x=list(confidence_scores.values()), y=list(confidence_scores.keys()), orientation="h",
        marker=dict(color=CLASS_COLOURS, line=dict(color="white", width=1.5)),
        text=[f"{v:.1f}%" for v in confidence_scores.values()],
        textposition="inside", textfont=dict(color="white", size=13, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
    ))
    fig_conf.update_layout(
        xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#e2e8f0"),
        yaxis=dict(tickfont=dict(family="Inter", size=12)),
        margin=dict(l=20, r=20, t=10, b=20), height=220,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )

    # Donut chart
    fig_donut = go.Figure(data=[go.Pie(
        labels=list(composition.keys()),
        values=[composition[cn]["percentage"] for cn in composition],
        hole=0.55,
        marker=dict(colors=CLASS_COLOURS, line=dict(color="white", width=3)),
        textinfo="label+percent", textfont=dict(size=12, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Area: %{value:.2f}%<extra></extra>",
    )])
    fig_donut.update_layout(
        title=dict(text="<b>Tissue Composition Map</b>", font=dict(size=14, color="#0f172a"), x=0.5),
        margin=dict(l=10, r=10, t=50, b=30), height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.08, font=dict(family="Inter", size=11)),
        annotations=[dict(text=f"<b>{malignant_pct:.1f}%</b><br>Malignant",
                          x=0.5, y=0.5, font=dict(size=16, color="#003d82"), showarrow=False)],
    )

    # Performance bar chart — real per-class IoU scores
    m_names = ["Background IoU", "Benign Epithelium IoU †", "Benign Stroma IoU", "Malignant IoU", "Pixel Accuracy"]
    m_vals  = [92.6, 65.7, 96.1, 91.9, 96.5]   # real confusion matrix — 4 val tiles (017–020)
    m_cols  = ["#003d82" if v >= 90 else "#ea580c" if v < 70 else "#0066cc" for v in m_vals]
    fig_perf = go.Figure(go.Bar(
        x=m_vals, y=m_names, orientation="h",
        marker=dict(color=m_cols, line=dict(color="white", width=1.5)),
        text=[f"{v:.1f}%" for v in m_vals],
        textposition="inside", textfont=dict(color="white", size=13, family="Inter"),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig_perf.update_layout(
        title=dict(text="<b>Validated Clinical Performance  ·  confusion matrix, val tiles 017–020</b>",
                   font=dict(size=13, color="#0f172a"), x=0.5),
        xaxis=dict(range=[60, 100], ticksuffix="%", gridcolor="#e2e8f0"),
        yaxis=dict(tickfont=dict(family="Inter", size=12)),
        margin=dict(l=20, r=30, t=50, b=20), height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    # Composition table
    comp_rows = [
        {
            "Tissue Type":   cn,
            "Pixels":        f"{composition[cn]['pixels']:,}",
            "Area (%)":      f"{composition[cn]['percentage']:.2f}%",
            "Est. Area (mm²)": f"{composition[cn]['area_mm2']:.2f}",
            "AI Confidence": f"{confidence_scores.get(cn, 0):.1f}%",
        }
        for cn in CLASS_NAMES
    ]

    # Molecular profile table
    mol_props = {
        "Malignant Stroma":  (85.2, 0.65, 1.8),
        "Benign Stroma":     (78.4, 0.52, 1.2),
        "Benign Epithelium": (72.1, 0.48, 1.4),
        "Background":        (0.0,  0.0,  0.0),
    }
    mol_rows = [
        {
            "Tissue Type":         cn,
            "Area (%)":            f"{composition[cn]['percentage']:.1f}%",
            "Protein Content":     f"{mol_props[cn][0]:.1f}%" if mol_props[cn][0] else "—",
            "Lipid/Protein Ratio": f"{mol_props[cn][1]:.2f}"  if mol_props[cn][0] else "—",
            "Rel. DNA Density":    f"{mol_props[cn][2]:.1f}×" if mol_props[cn][0] else "—",
        }
        for cn in CLASS_NAMES
    ]

    # Clinical molecular narrative
    if malignant_pct >= 30:
        mol_narrative = dcc.Markdown("""
**High-Grade Molecular Signature Detected:**
- **Elevated Amide I absorbance** — increased β-sheet content consistent with aberrant protein folding and upregulated collagen cross-linking in desmoplastic stroma
- **Enhanced phosphate band (1080–1240 cm⁻¹)** — elevated nuclear DNA density, indicative of high mitotic index and genomic instability
- **Altered lipid/protein ratio** — membrane remodelling and Warburg-effect metabolic reprogramming

**Clinical Implications:** Spectral phenotype consistent with invasive breast carcinoma (Grade II–III). Immediate multidisciplinary tumour board review recommended.
        """)
    elif malignant_pct >= 10:
        mol_narrative = dcc.Markdown("""
**Intermediate Molecular Risk Profile:**
- **Moderate Amide I perturbation** — cellular stress with early structural protein dysregulation
- **Mild phosphate band elevation** — modestly increased nuclear density, consistent with hyperplasia or DCIS

**Clinical Implications:** Consistent with atypical hyperplasia or early-stage in-situ carcinoma. Close surveillance and biopsy correlation recommended.
        """)
    else:
        mol_narrative = dcc.Markdown("""
**Normal Molecular Architecture:**
- **Balanced Amide I / Amide II ratio** — protein structure consistent with healthy epithelial and stromal populations
- **Stable phosphate signal** — normal nuclear DNA content with no proliferative index elevation

**Clinical Implications:** No spectral evidence of malignant transformation. Standard screening appropriate.
        """)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return html.Div([
        # ── Section 1: Images ─────────────────────────────────────────────
        html.Hr(className="qcl-divider"),
        html.H2("🔬 Comparative Tissue Analysis", className="section-title"),

        # ── Data provenance banner ────────────────────────────────────────
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("🔬", style={"fontSize": "1.4rem", "marginRight": "0.5rem"}),
                        html.Span("Real QCL Hyperspectral Cube",
                                  style={"fontWeight": "800", "fontSize": "0.95rem", "color": "#0f172a"}),
                    ]),
                    html.P("PCA channels 0–2 mapped to R/G/B. Spectral intensities — not true colour.",
                           style={"fontSize": "0.78rem", "color": "#64748b", "margin": "0.2rem 0 0 0"}),
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Span("✅", style={"fontSize": "1.4rem", "marginRight": "0.5rem"}),
                        html.Span("Real Pathologist Annotations",
                                  style={"fontWeight": "800", "fontSize": "0.95rem", "color": "#0f7c3d"}),
                    ]),
                    html.P([
                        "Pixel-level ground truth from ",
                        html.Strong("Kröger-Lui et al., "),
                        html.Em("Analytical Chemistry"),
                        " 2017. Bundled with the Zenodo dataset (CC BY 4.0). ",
                        html.Strong("Not generated — loaded directly from "),
                        html.Code(f"data/labels/{sample_id}_labels.npy"),
                        ".",
                    ], style={"fontSize": "0.78rem", "color": "#64748b", "margin": "0.2rem 0 0 0"}),
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Span("🤖", style={"fontSize": "1.4rem", "marginRight": "0.5rem"}),
                        html.Span("U-Net AI Prediction",
                                  style={"fontWeight": "800", "fontSize": "0.95rem", "color": "#003d82"}),
                    ]),
                    html.P("Segmentation by HyperspectralUNet (custom encoder, epoch 18). Pixel accuracy on this sample is shown in the disagreement panel.",
                           style={"fontSize": "0.78rem", "color": "#64748b", "margin": "0.2rem 0 0 0"}),
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Span("⚡", style={"fontSize": "1.4rem", "marginRight": "0.5rem"}),
                        html.Span("AI vs. Pathologist — Disagreement Map",
                                  style={"fontWeight": "800", "fontSize": "0.95rem", "color": "#b45309"}),
                    ]),
                    html.P([
                        f"Grey = model agrees ({pixel_agreement:.1f}% of pixels). ",
                        "Coloured pixels = class predicted by AI when it diverges from the pathologist annotation.",
                    ], style={"fontSize": "0.78rem", "color": "#64748b", "margin": "0.2rem 0 0 0"}),
                ], width=3),
            ], className="g-3"),
        ], style={"background": "#f8fafc", "border": "1px solid #e2e8f0", "borderRadius": "12px",
                  "padding": "1rem 1.2rem", "marginBottom": "1rem"}),

        dbc.Row([
            dbc.Col([
                html.P("🔬 Original Tissue (False-Colour RGB)", className="img-caption"),
                html.Img(src=img_orig, style={"width": "100%", "borderRadius": "10px", "border": "1px solid #e2e8f0"}),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.P("🏷️ Expert Pathologist Annotation", className="img-caption",
                           style={"display": "inline-block"}),
                    html.Span(" ✅ REAL", style={"background": "#dcfce7", "color": "#0f7c3d",
                                                  "fontWeight": "800", "fontSize": "0.72rem",
                                                  "borderRadius": "4px", "padding": "2px 6px",
                                                  "marginLeft": "6px", "verticalAlign": "middle",
                                                  "border": "1px solid #86efac"}),
                ]),
                html.Img(src=img_label, style={"width": "100%", "borderRadius": "10px",
                                               "border": "2px solid #22c55e"}),
                html.P([
                    html.Code(f"data/labels/{sample_id}_labels.npy"),
                    " — 480×480 uint8, classes 0–3",
                ], style={"fontSize": "0.72rem", "color": "#64748b", "marginTop": "0.3rem", "textAlign": "center"}),
            ], width=3),
            dbc.Col([
                html.P("🤖 AI Clinical Prediction", className="img-caption"),
                html.Img(src=img_pred, style={"width": "100%", "borderRadius": "10px", "border": "1px solid #003d82"}),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.P("⚡ Disagreement Map", className="img-caption",
                           style={"display": "inline-block"}),
                    html.Span(f" {pixel_agreement:.1f}% agree",
                              style={"background": "#fef3c7", "color": "#92400e",
                                     "fontWeight": "800", "fontSize": "0.72rem",
                                     "borderRadius": "4px", "padding": "2px 6px",
                                     "marginLeft": "6px", "verticalAlign": "middle",
                                     "border": "1px solid #fcd34d"}),
                ]),
                html.Img(src=img_diff, style={"width": "100%", "borderRadius": "10px",
                                              "border": "2px solid #f59e0b"}),
                html.P("Grey = correct · Colour = AI error (class-coloured)",
                       style={"fontSize": "0.72rem", "color": "#64748b",
                              "marginTop": "0.3rem", "textAlign": "center"}),
            ], width=3),
        ], className="g-3"),

        html.Hr(className="qcl-divider"),
        html.H4("🎨 Clinical Classification Legend", className="section-subtitle"),
        legend_cards,

        html.Hr(className="qcl-divider"),
        html.H4("🧠 AI Confidence Scores by Tissue Class", className="section-subtitle"),
        dcc.Graph(figure=fig_conf, config={"displayModeBar": False}),

        # ── Section 2: Validation & Risk ─────────────────────────────────
        html.Hr(className="qcl-divider"),
        html.H2("📊 Clinical Validation & Risk Assessment", className="section-title"),

        html.Div([
            html.H4(f"{assessment['icon']} Clinical Risk Assessment: {assessment['level']}",
                    style={"color": assessment["color"], "fontWeight": "800", "marginBottom": "0.7rem"}),
            html.P(assessment["description"], style={"fontSize": "1.05rem"}),
            html.P([html.Strong("Malignant Tissue Percentage: "), f"{malignant_pct:.1f}% of total segmented area"]),
        ], className="clinical-card", style={"borderLeft": f"6px solid {assessment['color']}"}),

        dbc.Row([
            dbc.Col([
                html.H4("🔬 Tissue Composition Analysis", className="section-subtitle"),
                dash_table.DataTable(
                    data=comp_rows,
                    columns=[{"name": c, "id": c} for c in comp_rows[0].keys()],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#f1f5f9", "fontWeight": "700", "color": "#0f172a"},
                    style_data={"color": "#0f172a"},
                    style_cell={"fontFamily": "Inter, sans-serif", "fontSize": "0.9rem", "padding": "10px 14px"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}
                    ],
                ),
            ], width=8),
            dbc.Col([
                html.H4("⚡ Model Performance", className="section-subtitle"),
                dbc.ListGroup([
                    dbc.ListGroupItem([html.Strong("Processing Time: "), f"{results['processing_time']:.1f} s"]),
                    dbc.ListGroupItem([html.Strong("Model Version: "),   results["model_version"]]),
                    dbc.ListGroupItem([html.Strong("Pixel Accuracy: "),   f"{CLINICAL_VALIDATION['overall_accuracy']:.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Mean IoU: "),        f"{CLINICAL_VALIDATION['mean_iou']:.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Malignant IoU: "),   f"{CLINICAL_VALIDATION['malignant_iou']:.1f}%"]),
                ], flush=True, style={"borderRadius": "12px", "overflow": "hidden"}),
            ], width=4),
        ], className="g-3 mt-2"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_donut, config={"displayModeBar": False}), width=6),
            dbc.Col(dcc.Graph(figure=fig_perf,  config={"displayModeBar": False}), width=6),
        ], className="g-3 mt-2"),

        html.H4("💡 Clinical Recommendations", className="section-subtitle"),
        html.Ol([html.Li(r) for r in assessment["recommendations"]]),

        # ── Section 3: Molecular insights ────────────────────────────────
        html.Hr(className="qcl-divider"),
        html.H2("🧬 Molecular Insights & Spectral Analysis", className="section-title"),
        html.P("Representative MIR absorption spectra illustrating typical tissue-class spectral signatures (912–1800 cm⁻¹ fingerprint region). "
               "Gaussian profiles centred on real diagnostic bands — for orientation. Not extracted from this specific sample.",
               style={"color":"#64748b","fontSize":"0.83rem","marginBottom":"0.4rem","fontStyle":"italic"}),
        dcc.Graph(figure=build_spectral_figure(), config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}),

        dbc.Row([
            dbc.Col(html.Div([
                html.Div("🔬 Tissue-Specific Molecular Profile", className="card-title"),
                html.P("Representative literature reference values per tissue class (illustrative — not directly measured from this sample's spectra).",
                       style={"fontSize":"0.75rem","color":"#64748b","marginBottom":"0.5rem","fontStyle":"italic"}),
                dash_table.DataTable(
                    data=mol_rows,
                    columns=[{"name": c, "id": c} for c in mol_rows[0].keys()],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#f1f5f9", "fontWeight": "700", "color": "#0f172a"},
                    style_data={"color": "#0f172a"},
                    style_cell={"fontFamily": "Inter, sans-serif", "fontSize": "0.88rem", "padding": "8px 12px"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}
                    ],
                ),
            ], className="clinical-card"), width=7),
            dbc.Col(html.Div([
                html.Div("🧪 Tissue Area Breakdown", className="card-title"),
                html.P("Derived from AI pixel-level segmentation of this sample.",
                       style={"fontSize":"0.75rem","color":"#64748b","marginBottom":"0.5rem","fontStyle":"italic"}),
                dbc.ListGroup([
                    dbc.ListGroupItem([html.Strong("Malignant Stroma: "),       f"{composition['Malignant Stroma']['percentage']:.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Benign Stroma: "),          f"{composition['Benign Stroma']['percentage']:.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Benign Epithelium: "),       f"{composition['Benign Epithelium']['percentage']:.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Background: "),              f"{composition['Background']['percentage']:.1f}%"]),
                ], flush=True, style={"borderRadius": "12px", "overflow": "hidden"}),
            ], className="clinical-card"), width=5),
        ], className="g-3 mt-2"),

        html.Div(mol_narrative, className="clinical-card"),

        # ── Section 4: Pathology Report ───────────────────────────────────
        html.Hr(className="qcl-divider"),
        html.H2("📋 AI-Assisted Clinical Pathology Report", className="section-title"),

        dbc.Row([
            dbc.Col(html.Div([
                html.Div("📋 Structured Diagnostic Report", className="card-title"),
                html.P([html.Strong("Patient Sample: "), sample_id, " | ",
                        html.Strong("Analysis Date: "), now, " UTC | ",
                        html.Strong("AI Model: "), results["model_version"]]),
                html.Hr(),
                html.H5("Specimen"),
                html.P(f"Breast tissue — hyperspectral QCL acquisition (Spero-QT 340, Daylight Solutions). "
                       f"Wide-field mode: 4.3 μm/pixel (0.3 NA), < 12 μm spatial resolution. "
                       f"Acquisition field: {cube.shape[0]}×{cube.shape[1]} pixels."),
                html.H5("Method"),
                html.P(f"Mid-infrared spectroscopic imaging (912–1800 cm⁻¹ · 223 raw bands). 15-component PCA pre-processing. "
                       f"Deep learning segmentation (HyperspectralUNet, custom 3-level encoder). Processing time: {results['processing_time']:.1f} s."),
                html.H5("Findings"),
                html.P([
                    f"Hyperspectral segmentation identifies {len([k for k,v in composition.items() if v['percentage']>1])} "
                    f"tissue classes with >1% area representation. Malignant stroma comprises ",
                    html.Strong(f"{malignant_pct:.1f}%"),
                    f" of the segmented tissue area. {assessment['description']}",
                ]),
                html.H5("Diagnostic Impression"),
                html.P([
                    html.Strong(f"{assessment['icon']} {assessment['level']}. "),
                    ("AI spectral analysis identifies features consistent with invasive malignancy. "
                     "Immediate histopathological correlation and oncological review are recommended."
                     if malignant_pct >= 30 else
                     "AI spectral analysis identifies features of intermediate risk. Enhanced surveillance and biopsy correlation are warranted."
                     if malignant_pct >= 10 else
                     "AI spectral analysis identifies no spectral evidence of malignant transformation. Standard surveillance appropriate."),
                ]),
                html.Hr(),
                html.P("⚠️ This AI-generated report is a decision-support tool, not a primary diagnostic instrument. "
                       "All findings must be correlated with conventional H&E histopathology, clinical presentation, "
                       "and radiological findings before clinical action. Model metrics: mean Dice 92.3%, "
                       "pixel accuracy 96.5%, malignant IoU 91.9% (checkpoint epoch 18 / 28, Zenodo CC BY 4.0 dataset). "
                       "Not FDA-cleared for standalone clinical use.",
                       style={"fontSize": "0.85rem", "color": "#475569"}),
            ], className="clinical-card"), width=9),

            dbc.Col(html.Div([
                html.Div("📊 Key Metrics", className="card-title"),
                dbc.ListGroup([
                    dbc.ListGroupItem([html.Strong("Malignant Area: "),  f"{malignant_pct:.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Risk Level: "),       assessment["level"]]),
                    dbc.ListGroupItem([html.Strong("AI Confidence: "),   f"{max(confidence_scores.values()):.1f}%"]),
                    dbc.ListGroupItem([html.Strong("Model Accuracy: "),  "Dice 92.3% · IoU 86.6%"]),
                    dbc.ListGroupItem([html.Strong("Malignant IoU: "),   "0.919"]),  # confusion matrix, 4 val tiles
                ], flush=True, style={"borderRadius": "12px", "overflow": "hidden"}),
            ], className="clinical-card"), width=3),
        ], className="g-3"),
    ])


# ── App Layout ────────────────────────────────────────────────────────────────
SAMPLES = list_samples()

app.layout = html.Div([
    # Loading overlay
    html.Div([
        html.Div("🧠", style={"fontSize": "3rem"}),
        html.Span("U-Net inference in progress…"),
        html.Small("Segmenting tissue classes via deep neural network", style={"color": "#475569"}),
    ], id="loading-overlay"),

    dcc.Store(id="analysis-store"),

    # ── Two-column layout: sidebar + main ────────────────────────────────
    html.Div([
        build_sidebar(),
        html.Div([  # main content column

    dbc.Container([
        # ── Header ──────────────────────────────────────────────────────
        html.Div([
            dbc.Row([
                # Left: wordmark + tagline + tech pills
                dbc.Col([
                    html.Div([
                        html.Span("SPECTRA",
                            style={"background":"linear-gradient(90deg,#ffffff 0%,#93c5fd 100%)",
                                   "-webkit-background-clip":"text","-webkit-text-fill-color":"transparent",
                                   "background-clip":"text","font-size":"3.6rem","font-weight":"900",
                                   "letter-spacing":"-0.03em","font-family":"Inter,sans-serif",
                                   "text-shadow":"none","line-height":"1","display":"inline-block"}),
                    ]),
                    html.Div("QCL · AI PLATFORM  ·  BREAST CANCER DIAGNOSTICS", className="hdr-sub"),
                    html.Div(
                        "Spectral Pathology Engine for Cancer Tissue Recognition & Assessment",
                        className="hdr-tagline",
                    ),
                    html.Div([
                        html.Span("QCL Imaging",           className="hdr-tag"),
                        html.Span("U-Net Deep Learning",   className="hdr-tag"),
                        html.Span("MIR 912–1800 cm⁻¹",   className="hdr-tag"),
                        html.Span("< 5 μm Resolution",     className="hdr-tag"),
                        html.Span("< 2.3 min / case",      className="hdr-tag"),
                        html.Span("Label-free",            className="hdr-tag"),
                    ], className="hdr-tags"),
                ], width=8),

                # Right: version + validation checklist + author
                dbc.Col([
                    html.Div([
                        html.Div("v 3.2.1  ·  RESEARCH GRADE", className="hdr-version"),
                        html.Hr(className="hdr-sep"),
                        html.Div([html.Span("✓", className="hdr-tick"),
                                  html.Span("207-patient Zenodo cohort \u00b7 CC BY 4.0")], className="hdr-check"),
                        html.Div([html.Span("✓", className="hdr-tick"),
                                  html.Span("Mean Dice 92.3%  \u00b7  Malignant IoU 91.9%")],  className="hdr-check"),
                        html.Div([html.Span("✓", className="hdr-tick"),
                                  html.Span("Real 480\u00d7480\u00d715 QCL cubes loaded")],        className="hdr-check"),
                        html.Div([html.Span("✓", className="hdr-tick"),
                                  html.Span("Checkpoint epoch 18 / 28 \u00b7 val loss 0.1699")],  className="hdr-check"),
                        html.Hr(className="hdr-sep"),
                        html.Div("A. Domingues Batista  ·  Portfolio 2026", className="hdr-author"),
                    ], className="hdr-right"),
                ], width=4, style={"display":"flex","alignItems":"center","justifyContent":"flex-end"}),
            ], align="center"),
        ], className="qcl-header"),

        # ── Metric cards ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("🎯", className="metric-icon"),
                html.Div("96.5%", className="metric-value", style={"color": "#003d82"}),
                html.Div("Pixel Accuracy", className="metric-label"),
                html.Div("Real inference \u00b7 mean IoU 86.6%", className="metric-sub"),
            ], className="metric-card", style={"--accent": "#003d82"}), width=3),
            dbc.Col(html.Div([
                html.Div("⚡", className="metric-icon"),
                html.Div("< 2.3min", className="metric-value", style={"color": "#0891b2"}),
                html.Div("Analysis Time", className="metric-label"),
                html.Div("Real-time Processing", className="metric-sub"),
            ], className="metric-card", style={"--accent": "#0891b2"}), width=3),
            dbc.Col(html.Div([
                html.Div("🧬", className="metric-icon"),
                html.Div("207", className="metric-value", style={"color": "#0f7c3d"}),
                html.Div("Zenodo Cohort", className="metric-label"),
                html.Div("CC BY 4.0 Open Access", className="metric-sub"),
            ], className="metric-card", style={"--accent": "#0f7c3d"}), width=3),
            dbc.Col(html.Div([
                html.Div("🔬", className="metric-icon"),
                html.Div("20", className="metric-value", style={"color": "#7c3aed"}),
                html.Div("Cubes on Disk", className="metric-label"),
                html.Div("From 25 GB Zenodo Download", className="metric-sub"),
            ], className="metric-card", style={"--accent": "#7c3aed"}), width=3),
        ], className="g-3 mb-3"),

        html.Hr(className="qcl-divider"),

        # ── Controls ─────────────────────────────────────────────────────
        html.Div([
            html.H3("🗂️ Patient Sample Selection", style={"color": "#003d82", "fontWeight": "700"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Patient Sample ID:"),
                    dcc.Dropdown(
                        id="sample-selector",
                        options=[{"label": f"Sample {s}", "value": s} for s in SAMPLES],
                        value=SAMPLES[-1] if SAMPLES else None,
                        clearable=False,
                        style={"fontFamily": "Inter, sans-serif"},
                    ),
                ], width=8),
                dbc.Col([
                    html.Label("Total Available Samples"),
                    html.Div(str(len(SAMPLES)), style={"fontSize": "2rem", "fontWeight": "800", "color": "#003d82"}),
                ], width=4),
            ], className="g-3 mb-3"),

            html.Div(id="sample-info"),

            dbc.Row([
                dbc.Col([
                    html.Label("AI Confidence Threshold"),
                    dcc.Slider(
                        id="conf-threshold", min=0.5, max=0.95, step=0.05, value=0.85,
                        marks={v: f"{int(v*100)}%" for v in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], width=8),
                dbc.Col([
                    html.Label("Primary Spectral Band"),
                    dcc.Dropdown(
                        id="spectral-band",
                        options=[
                            {"label": "PC-1  · 81.0% var  — dominant tissue signal",  "value": 0},
                            {"label": "PC-2  · 10.4% var  — protein/lipid contrast",  "value": 1},
                            {"label": "PC-3  ·  4.8% var  — glycogen / DNA features", "value": 2},
                            {"label": "PC-4  ·  1.8% var  — stromal variation",       "value": 3},
                            {"label": "PC-5  ·  1.1% var  — epithelial contrast",     "value": 4},
                            {"label": "PC-6  ·  0.29% var — subtle morphology",       "value": 5},
                            {"label": "PC-7  ·  0.17% var — residual biochemistry",   "value": 6},
                            {"label": "PC-8  ·  0.11% var",                           "value": 7},
                            {"label": "PC-9  ·  0.07% var",                           "value": 8},
                            {"label": "PC-10 ·  0.07% var",                           "value": 9},
                            {"label": "PC-11 ·  0.05% var",                           "value": 10},
                            {"label": "PC-12 ·  0.05% var",                           "value": 11},
                            {"label": "PC-13 ·  0.04% var",                           "value": 12},
                            {"label": "PC-14 ·  0.03% var",                           "value": 13},
                            {"label": "PC-15 ·  0.03% var — near-noise residual",     "value": 14},
                        ],
                        value=0, clearable=False,
                        style={"fontFamily": "Inter, sans-serif"},
                    ),
                    html.Div([
                        html.Span("ℹ️ ", style={"fontSize": "0.8rem"}),
                        html.Span(
                            "Variance % computed across all 4.6 M pixels from 20 QCL cubes (223 raw MIR bands → 15 PCA). "
                            "PC-1–5 capture 99.1% of spectral variance. PC-6–15 encode subtle residual features.",
                            style={"fontSize": "0.72rem", "color": "#64748b"},
                        ),
                    ], style={"marginTop": "0.4rem", "lineHeight": "1.4"}),
                ], width=4),
            ], className="g-3 mb-3"),

            dbc.Button("🚀 Execute Advanced AI Analysis", id="run-btn", className="btn-primary-qcl"),
            dbc.Button("↩ Back to Overview", id="reset-btn", n_clicks=0,
                       style={"marginLeft": "0.75rem", "background": "transparent",
                              "border": "2px solid #003d82", "color": "#003d82",
                              "fontWeight": "600", "borderRadius": "8px",
                              "padding": "0.5rem 1.2rem", "display": "none"}),
            dcc.Loading(
                id="loading-status",
                type="circle",
                color="#003d82",
                children=html.Div(id="analysis-status", className="mt-3"),
            ),
        ], className="controls-panel", id="sec-analysis"),

        # ── Dynamic content area ──────────────────────────────────────────
        dcc.Loading(
            id="loading-main",
            type="default",
            color="#003d82",
            fullscreen=False,
            children=html.Div(id="main-content"),
        ),

        # ── Technical Specifications ──────────────────────────────────────
        html.Hr(className="qcl-divider"),
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("🧠 AI Model Architecture", className="card-title"),
                        html.Ul([
                            html.Li([html.Strong("Architecture: "), "HyperspectralUNet (custom 3-level encoder · no pretrained backbone)"]),
                            html.Li([html.Strong("Input: "),        "15-channel PCA-reduced hyperspectral data"]),
                            html.Li([html.Strong("Loss: "),         "Weighted CrossEntropyLoss (weights [0.5, 1.0, 1.0, 2.5]) + Soft Dice (50/50)"]),
                            html.Li([html.Strong("Optimiser: "),    "AdamW (lr=1×10⁻⁴, weight decay=1×10⁻⁴) · CosineAnnealingLR"]),
                            html.Li([html.Strong("Training set: "), "20 real QCL cubes (480\u00d7480\u00d715) \u00b7 Zenodo CC BY 4.0"]),
                            html.Li([html.Strong("Best epoch: "),   "Epoch 18 of 28 \u00b7 mean IoU 86.6% \u00b7 mean Dice 92.3%"]),
                        ]),
                    ], className="clinical-card"), width=6),
                    dbc.Col(html.Div([
                        html.Div("🔬 Spero-QT 340 — Daylight Solutions", className="card-title"),
                        html.Ul([
                            html.Li([html.Strong("Spectral range: "),      "950–1800 cm⁻¹ std · customisable 800–2300 cm⁻¹"]),
                            html.Li([html.Strong("Spectral resolution: "), "Variable · down to 2 cm⁻¹"]),
                            html.Li([html.Strong("High-res mode (0.7 NA):"), "1.3 μm pixel · < 5 μm spatial res · 650×650 μm FOV"]),
                            html.Li([html.Strong("Wide-field mode (0.3 NA):"), "4.3 μm pixel · < 12 μm spatial res · 2×2 mm FOV"]),
                            html.Li([html.Strong("Acquisition: "),         "< 40 s for 450 absorbance images · 150× faster than FTIR"]),
                            html.Li([html.Strong("Camera: "),              "480×480 uncooled microbolometer FPA (no LN₂ required)"]),
                            html.Li([html.Strong("Throughput: "),          "> 7 million spectral points per second"]),
                            html.Li([html.Strong("Software: "),            "ChemVision™ · exports to MATLAB / ENVI"]),
                        ]),
                    ], className="clinical-card"), width=6),
                ], className="g-3"),
                html.H4("📊 Training Performance Results", className="section-subtitle"),
                dash_table.DataTable(
                    data=[
                        {"Metric": "Pixel Accuracy",          "Value": "96.5%", "Notes": "4 val tiles (017–020), confusion matrix"},
                        {"Metric": "Mean IoU (4 classes)",    "Value": "86.6%", "Notes": "Pixel-weighted Jaccard"},
                        {"Metric": "Malignant Stroma IoU",    "Value": "91.9%", "Notes": "Primary clinical class"},
                        {"Metric": "Malignant Stroma Recall", "Value": "99.1%", "Notes": "Very few missed malignancies"},
                        {"Metric": "Benign Stroma IoU",       "Value": "96.1%", "Notes": "Highest-scoring class"},
                        {"Metric": "Checkpoint saved epoch",  "Value": "18 / 28", "Notes": "Best val_loss = 0.1699"},
                    ],
                    columns=[{"name": c, "id": c} for c in ["Metric", "Value", "Notes"]],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#f1f5f9", "fontWeight": "700", "color": "#0f172a"},
                    style_data={"color": "#0f172a"},
                    style_cell={"fontFamily": "Inter, sans-serif", "padding": "10px 14px"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
                ),
            ], title="⚙️ Technical Specifications & Model Details"),
        ], start_collapsed=True),

        # ── References ────────────────────────────────────────────────────
        html.Div([  # id="sec-refs" is set below
            html.Span("09  References", className="num-label"),
            html.H2("Citations & Data Provenance", className="section-title",
                    style={"marginTop": "0.4rem"}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("\U0001f4da Dataset & Paper", className="card-title"),
                    html.Ol([
                        html.Li([
                            html.Strong("Kröger-Lui, N. et al. (2017). "),
                            html.Em("\"Quantum cascade laser spectral histopathology: Breast cancer diagnostics using high throughput chemical imaging.\""),
                            " Analytical Chemistry. ",
                            html.A("Dataset DOI: 10.5281/zenodo.808456", href="https://doi.org/10.5281/zenodo.808456", target="_blank"),
                            " \u00b7 CC BY 4.0.",
                        ], style={"marginBottom": "0.75rem"}),
                        html.Li([
                            html.Strong("Dataset: "),
                            "QCL hyperspectral breast tissue microarray — Zenodo open-access repository. 207 patients, CC BY 4.0 licence. ",
                            html.A("DOI: 10.5281/zenodo.808456", href="https://doi.org/10.5281/zenodo.808456", target="_blank"),
                        ], style={"marginBottom": "0.75rem"}),
                    ], style={"fontSize": "0.85rem", "color": "#334155", "paddingLeft": "1.2rem"}),
                ], className="clinical-card"), width=6),
                dbc.Col(html.Div([
                    html.Div("\U0001f9e0 Model & Methods", className="card-title"),
                    html.Ol(start=3, children=[
                        html.Li([
                            html.Strong("Ronneberger, O., Fischer, P., & Brox, T. (2015). "),
                            html.Em("\"U-Net: Convolutional Networks for Biomedical Image Segmentation.\""),
                            " MICCAI 2015. ",
                            html.A("arXiv: 1505.04597", href="https://arxiv.org/abs/1505.04597", target="_blank"),
                        ], style={"marginBottom": "0.75rem"}),
                        html.Li([
                            html.Strong("Daylight Solutions / Leonardo DRS. "),
                            html.Em("Spero\u00ae QT 340 Wide-Field QCL-IR Microscope — Product Specification Sheet (2024)."),
                            " Spectral range 950–1800 cm\u207b\u00b9, 480\u00d7480 uncooled FPA, < 40 s per 450-image cube.",
                        ], style={"marginBottom": "0.75rem"}),
                        html.Li([
                            html.Strong("This work: "),
                            "HyperspectralUNet trained on 20 processed cubes (480\u00d7480\u00d715) from the Zenodo dataset. "
                            "Checkpoint epoch 18 of 28 \u00b7 val loss 0.1699 \u00b7 pixel accuracy 96.5% \u00b7 mean IoU 86.6%. "
                            "Source code available in this portfolio repository.",
                        ]),
                    ], style={"fontSize": "0.85rem", "color": "#334155", "paddingLeft": "1.2rem"}),
                ], className="clinical-card"), width=6),
            ], className="g-3"),
        ], id="sec-refs", style={"marginTop": "3rem", "marginBottom": "1rem"}),

        # ── Footer ───────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("SPECTRA", style={
                    "background": "linear-gradient(90deg,#93c5fd 0%,#ffffff 100%)",
                    "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent",
                    "background-clip": "text", "fontWeight": "900", "fontSize": "1.6rem",
                    "letterSpacing": "-0.02em", "marginRight": "0.6rem",
                }),
                html.Span("Spectral Pathology Engine for Cancer Tissue Recognition & Assessment",
                          style={"fontSize": "0.82rem", "color": "rgba(255,255,255,0.55)",
                                 "fontWeight": "400", "verticalAlign": "middle"}),
            ], style={"marginBottom": "0.8rem"}),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "margin": "0.7rem 0"}),
            dbc.Row([
                dbc.Col([
                    html.P([
                        html.Strong("Instrument: "),
                        "Spero-QT 340 wide-field QCL-IR microscope  \u00b7  Daylight Solutions (Leonardo DRS)",
                        html.Br(),
                        html.Strong("Data: "),
                        "Kr\u00f6ger-Lui et al. \u2014 Zenodo 10.5281/zenodo.808456 \u00b7 CC BY 4.0  \u00b7  ",
                        html.Strong("Model: "),
                        "HyperspectralUNet  \u00b7  20 cubes (480\u00d7480\u00d715)  \u00b7  checkpoint epoch 18 / 28 \u00b7 pixel acc 96.5%",
                    ], style={"fontSize": "0.8rem", "color": "rgba(255,255,255,0.65)", "margin": "0"}),
                ], width=8),
                dbc.Col([
                    html.P([
                        html.Strong("Alex Domingues Batista"), html.Br(),
                        html.A("linkedin.com/in/alexdbatista",
                               href="https://linkedin.com/in/alexdbatista", target="_blank",
                               style={"color": "#93c5fd"}),
                        "  ·  ",
                        html.A("alex.domin.batista@gmail.com",
                               href="mailto:alex.domin.batista@gmail.com",
                               style={"color": "#93c5fd"}),
                    ], style={"fontSize": "0.8rem", "color": "rgba(255,255,255,0.65)",
                              "margin": "0", "textAlign": "right"}),
                ], width=4),
            ], align="center"),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "margin": "0.7rem 0"}),
            html.P(
                "Research portfolio project · March 2026  ·  "
                "For demonstration purposes — not a certified medical device  ·  v1.0",
                style={"fontSize": "0.72rem", "color": "rgba(255,255,255,0.35)",
                       "textAlign": "center", "margin": "0"},
            ),
        ], className="qcl-footer"),

    ], fluid=True, style={"maxWidth": "1400px", "padding": "1.5rem"}),
        ], id="main-wrapper"),  # end main-wrapper
    ], id="outer-flex"),  # end outer-flex
], style={"minHeight": "100vh", "background": "#fafbfc"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("sample-info", "children"),
    Input("sample-selector", "value"),
)
def update_sample_info(sample_id):
    if not sample_id:
        return ""
    cube, _ = load_sample_data(sample_id)
    if cube is None:
        return dbc.Alert("❌ Could not load sample data.", color="danger")
    return dbc.Alert([
        html.Strong(f"📋 Sample {sample_id} — Clinical Metadata"), html.Br(),
        f"Hyperspectral cube: {cube.shape[0]}×{cube.shape[1]} px, {cube.shape[2]} PCA channels (from 223 raw MIR bands, 912–1800 cm⁻¹) | "
        f"4.3 μm/pixel (wide-field 0.3 NA) | Ground truth: Expert pathologist annotations available",
    ], color="info", style={"color": "#0f172a", "borderRadius": "12px"})


@app.callback(
    Output("analysis-store",  "data"),
    Output("analysis-status", "children"),
    Output("main-content",    "children"),
    Output("reset-btn",       "style"),
    Input("run-btn",           "n_clicks"),
    Input("reset-btn",         "n_clicks"),
    State("sample-selector",   "value"),
    State("analysis-store",    "data"),
    prevent_initial_call=False,
)
def run_or_show(run_clicks, reset_clicks, sample_id, store):
    _RESET_HIDDEN  = {"marginLeft": "0.75rem", "background": "transparent",
                      "border": "2px solid #003d82", "color": "#003d82",
                      "fontWeight": "600", "borderRadius": "8px",
                      "padding": "0.5rem 1.2rem", "display": "none"}
    _RESET_VISIBLE = {**_RESET_HIDDEN, "display": "inline-block"}

    from dash import ctx
    triggered = ctx.triggered_id

    # Reset or initial load → show pre-analysis
    if triggered == "reset-btn" or not run_clicks:
        return None, "", pre_analysis_content(), _RESET_HIDDEN

    # Run analysis
    if not sample_id:
        return store, dbc.Alert("Please select a sample first.", color="warning"), pre_analysis_content(), _RESET_HIDDEN

    cube, labels = load_sample_data(sample_id)
    if cube is None or labels is None:
        return store, dbc.Alert("❌ Could not load sample data.", color="danger"), pre_analysis_content(), _RESET_HIDDEN

    try:
        results = run_analysis(sample_id)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return store, dbc.Alert(f"❌ Analysis error: {exc}", color="danger"), pre_analysis_content(), _RESET_HIDDEN
    if results is None:
        return store, dbc.Alert("❌ Analysis failed — check server logs.", color="danger"), pre_analysis_content(), _RESET_HIDDEN

    store_data = {
        "sample_id":         sample_id,
        "prediction":        results["prediction"].tolist(),
        "confidence_scores": results["confidence_scores"],
        "processing_time":   results["processing_time"],
        "model_version":     results["model_version"],
    }

    status = dbc.Alert(
        f"✅ Analysis complete in {results['processing_time']:.1f} seconds — scroll down for full results",
        color="success", style={"borderRadius": "12px"},
    )

    results["prediction"] = np.array(results["prediction"])
    content = post_analysis_content(sample_id, cube, labels, results)
    return store_data, status, content, _RESET_VISIBLE


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
