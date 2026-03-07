import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The specific Zenodo ID for "Quantum Cascade Laser Spectral Histopathology: 
# Breast Cancer Diagnostics Using High Throughput Chemical Imaging"
# Original Publication: Analytical Chemistry (2017)
ZENODO_DOI = "808456"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_DOI}"

DATA_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Download Logic (Regulatory/Pipeline focus)
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: Path):
    """
    Download a file from an external URL with a progress bar.
    
    Parameters
    ----------
    url : str
        The direct download link (e.g., from the Zenodo API).
    dest_path : Path
        Local filesystem destination for the downloaded file.
    """
    if dest_path.exists():
        print(f"[INFO] File already exists at {dest_path}. Skipping download.")
        return

    print(f"[INFO] Downloading {dest_path.name} from Zenodo...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("[ERROR] Something went wrong during download.")
        if dest_path.exists():
            dest_path.unlink()
    else:
        print(f"[SUCCESS] Downloaded {dest_path.name}")


def get_zenodo_files() -> list:
    """Query the Zenodo API to fetch all file URLs associated with the DOI."""
    print(f"[INFO] Querying Zenodo API for record {ZENODO_DOI}...")
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()
    
    record = response.json()
    files = []
    
    if "files" in record:
        for f in record["files"]:
            files.append({
                "filename": f["key"],
                "url": f["links"]["self"],
                "size_mb": f["size"] / (1024 * 1024)
            })
    return files


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"--- QCL Breast Cancer Histopathology Ingestion ---")
    print(f"Target Directory: {DATA_DIR.absolute()}\n")
    
    try:
        zenodo_files = get_zenodo_files()
        total_size = sum([f["size_mb"] for f in zenodo_files])
        print(f"[INFO] Found {len(zenodo_files)} files. Total dataset size: {total_size:.2f} MB")
        
        for file_info in zenodo_files:
            file_dest = DATA_DIR / file_info["filename"]
            download_file(file_info["url"], file_dest)
            
        print("\n[DONE] Data ingestion pipeline complete. ")
        print("Note: Check the raw/ directory for .mat hyperspectral cubes.")
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Network error during Zenodo API call: {e}")
    except Exception as e:
        print(f"\n[ERROR] Pipeline aborted: {e}")
