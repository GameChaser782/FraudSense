"""
End-to-end training pipeline for FraudSense.
Run this once after downloading the dataset.

Steps:
  1. Preprocess raw CSV → train/val/test parquet + meta.joblib
  2. Build heterogeneous transaction graph (.pt)
  3. Train LightGBM baseline
  4. Train GNN (GAT)
  5. Train stacking ensemble → save metrics.json
  6. Compute SHAP explanations
  7. Detect fraud rings → save fraud_rings.json
  8. Export all static JSON for the web dashboard

Usage:
  source venv/bin/activate
  python train_all.py [--skip-gnn]   # --skip-gnn for fast CPU run
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main(skip_gnn: bool = False):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    step("1 / 7  Preprocessing")
    from src.data.preprocess import run as preprocess
    preprocess()

    step("2 / 7  Building graph")
    from src.data.graph_builder import run as build_graph
    build_graph()

    step("3 / 7  Training LightGBM baseline")
    from src.models.tabular_baseline import train as train_lgbm
    train_lgbm()

    if not skip_gnn:
        step("4 / 7  Training GNN (GAT)  [this takes ~10–20 min on CPU]")
        from src.models.gnn_model import train as train_gnn
        train_gnn()
    else:
        print("\n[skipped] GNN training — will use LightGBM-only ensemble")
        import numpy as np
        import yaml
        with open("configs/config.yaml") as f:
            import yaml; cfg = yaml.safe_load(f)
        # Duplicate LightGBM preds as GNN stand-in so ensemble.py can run
        proc = cfg["data"]["processed_dir"]
        lgbm_val  = np.load(f"{proc}/lgbm_val_preds.npy")
        lgbm_test = np.load(f"{proc}/lgbm_test_preds.npy")
        np.save(f"{proc}/gnn_val_preds.npy",  lgbm_val)
        np.save(f"{proc}/gnn_test_preds.npy", lgbm_test)

    step("5 / 7  Stacking ensemble + final metrics")
    from src.models.ensemble import train as train_ensemble
    train_ensemble()

    step("6 / 7  SHAP explainability")
    from src.evaluation.explainability import run as run_shap
    run_shap()

    step("7 / 7  Fraud ring detection")
    from src.visualization.graph_viz import run as run_rings
    run_rings()

    step("Done!")
    print("\nAll artifacts saved. Start the dashboard with:")
    print("  source venv/bin/activate && uvicorn webapp.app:app --reload")
    print("\nFor GitHub Pages deployment, run:")
    print("  python export_static.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN training (uses LightGBM-only ensemble)")
    args = parser.parse_args()
    main(skip_gnn=args.skip_gnn)
