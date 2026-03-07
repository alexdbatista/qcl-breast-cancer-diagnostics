# 🔬 QCL Spatial Histopathology: Breast Cancer Diagnostics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-ee4c2c)
![Healthcare AI](https://img.shields.io/badge/Domain-Digital_Pathology-red)

**QCL Spatial Histopathology** is an advanced biomedical computer vision project that leverages high-throughput infrared hyperspectral imaging to automate breast cancer diagnostics. Unlike standard digital pathology which relies on chemical staining (H&E) and RGB images, this pipeline uses raw middle-infrared vibrational signatures obtained via a **Daylight Solutions Spero® QCL Microscope**.

This entirely label-free technique enables the segmentation of malignant stroma from benign tissue based entirely on physical chemistry and molecular vibrational contrasts.

---

## 🎯 Business & Clinical Impact (EU Market Focus)

- **Eliminates Staining Bottlenecks:** Replaces expensive, time-consuming, and variable wet-lab tissue staining with a rapid, optical label-free readout.
- **Augments Human Pathologists:** Automates standard screening, freeing highly specialized pathologists to focus on ambiguous and edge-case diagnoses.
- **DSGVO / GDPR Compliance by Design:** Features fully anonymized tissue microarray (TMA) data.
- **Deep Tech Intersections:** Bridges photonics hardware (Quantum Cascade Lasers), physical chemistry (IR spectroscopy), and spatial AI (Computer Vision).

---

## 🏗️ Technical Architecture

This project is structured iteratively, mimicking an industrial AI R&D pipeline for medical devices (ISO 13485 concepts):

1. **Ingestion & Data Quality (Module 01):**
   - Fetches and validates raw hyperspectral tensors (Data cubes).
   - Manages large out-of-core multidimensional `.mat` files.
2. **Spectral Dimensionality Reduction (Module 02):**
   - Applies Manifold Learning (UMAP, t-SNE) to identify underlying phenotype clusters inside the 'fingerprint' IR region (912 to 1800 cm⁻¹).
3. **Deep Spatial Segmentation (Module 03):**
   - Trains a convolutional architecture (e.g., U-Net or Vision Transformer) to classify pixels into histological classes (e.g., Malignant Stroma vs. Benign).

---

## 💾 Dataset Details

We utilize the open-access dataset: *"Quantum Cascade Laser Spectral Histopathology: Breast Cancer Diagnostics Using High Throughput Chemical Imaging"*.

- **DOI:** [10.5281/zenodo.808456](https://doi.org/10.5281/zenodo.808456)
- **Cohort:** Breast cancer tissue microarray (TMA) from 207 unique patients.
- **Source:** Originally published in *Analytical Chemistry* (2017).

> *Note: Due to the extreme size of hyperspectral data cubes (often gigabytes per core), the `data/` directory is strictly ignored by Git. You must fetch the data locally using the provided ingestion Python script.*

---

## 🚀 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Fetch the Dataset:**
   ```bash
   python 01_data_ingestion.py
   ```
   *Note: This script queries Zenodo API to download the primary TMA `.mat` files to `data/raw/`.*

---

📍 *Part of the Applied Data Science Architectures portfolio by Alex Domingues Batista.*
