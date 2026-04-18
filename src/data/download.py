"""
Download IEEE-CIS Fraud Detection dataset from Kaggle.
Requires KAGGLE_API_TOKEN env variable (set in .env).
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
DATASET = "lnasiri007/ieeecis-fraud-detection"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected = RAW_DIR / "train_transaction.csv"
    if expected.exists():
        print("Dataset already downloaded.")
        return

    print(f"Downloading {DATASET}...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Kaggle download failed. Check KAGGLE_API_TOKEN in .env")
    print("Download complete.")


if __name__ == "__main__":
    download()
