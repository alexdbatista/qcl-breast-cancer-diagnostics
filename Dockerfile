# ── Hugging Face Spaces / Render compatible Docker image ─────────────────────
# Uses CPU-only PyTorch to keep image size < 1.5 GB.
# HF Spaces expects the app to listen on port 7860.

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by Pillow / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements_dash.txt .
RUN pip install --no-cache-dir -r requirements_dash.txt

# Copy application code and assets
COPY app_dash.py 03_spatial_cnn_segmentation.py ./
COPY models/ models/
COPY data/processed/ data/processed/
COPY data/labels/ data/labels/

# Render.com sets $PORT dynamically; HF Spaces uses 7860; fallback to 7860.
EXPOSE 7860

# Shell form so gunicorn picks up $PORT at runtime (required by Render.com).
CMD ["sh", "-c", "gunicorn app_dash:server --bind 0.0.0.0:${PORT:-7860} --timeout 300 --workers 1 --threads 2"]
