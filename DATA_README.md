# Dataset Provenance

## Primary Dataset

**Title:** Quantum Cascade Laser Spectral Histopathology: Breast Cancer Diagnostics Using High Throughput Chemical Imaging

**DOI:** [10.5281/zenodo.808456](https://doi.org/10.5281/zenodo.808456)

**Repository:** Zenodo (open-access, CC-BY license)

**Original Publication:**
> Kröger-Lui, Norbert, et al. "Quantum cascade laser-based hyperspectral imaging of biological tissue."
> *Analytical Chemistry* 2017.

**Instrument:** Daylight Solutions Spero® Quantum Cascade Laser (QCL) Infrared Microscope

---

## Cohort Description

| Property | Value |
|----------|-------|
| **Sample Type** | Breast Cancer Tissue Microarray (TMA) |
| **Patient Cohort** | 207 unique patients |
| **Tissue Classes** | 4 histological categories (Malignant Stroma, Benign Epithelium, etc.) |
| **Spectral Range** | 912 – 1800 cm⁻¹ (Mid-IR fingerprint region) |
| **File Format** | MATLAB `.mat` hyperspectral data cubes |
| **Anonymization** | Patient identifiers fully removed — DSGVO/GDPR compliant |

---

## Data Management

- Raw `.mat` files are stored in `data/raw/` — **gitignored** due to file size (multi-GB).
- Processed PCA-reduced tensors are stored in `data/processed/` — **gitignored**.
- Run `python 01_data_ingestion.py` to automatically download all files from Zenodo.

## Licensing

This dataset is provided under the [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
Original authors must be cited in any derivative works or publications.
