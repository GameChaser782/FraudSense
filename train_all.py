"""
End-to-end training pipeline for FraudSense.
Auto-resumes from the last completed step — re-run safely after any crash.

Steps:
  1. Preprocess raw CSV → train/val/test parquet + meta.joblib
  2. Build heterogeneous transaction graph (.pt)
  3. Train LightGBM baseline
  4. Train GNN (GAT)
  5. Train stacking ensemble → save metrics.json
  6. Compute SHAP explanations
  7. Detect fraud rings → save fraud_rings.json

Usage:
  source venv/bin/activate
  python train_all.py              # full run (auto-skips completed steps)
  python train_all.py --skip-gnn  # skip GNN, use LightGBM-only ensemble
  python train_all.py --force     # ignore checkpoints, redo everything
"""

import argparse
import os
import sys
import yaml
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC  = CFG["data"]["processed_dir"]
SAVED = CFG["model"]["saved_dir"]

# ── Checkpoint definitions ────────────────────────────────────────────────────
# Each step is considered done if ALL its output files exist.
CHECKPOINTS = {
    1: [f"{PROC}/train.parquet", f"{PROC}/val.parquet",
        f"{PROC}/test.parquet", f"{PROC}/meta.joblib"],
    2: [CFG["data"]["graph_file"]],
    3: [CFG["model"]["lgbm_path"],
        f"{PROC}/lgbm_val_preds.npy", f"{PROC}/lgbm_test_preds.npy"],
    4: [CFG["model"]["gnn_path"],
        f"{PROC}/gnn_val_preds.npy",  f"{PROC}/gnn_test_preds.npy"],
    5: [CFG["model"]["metrics_path"], CFG["model"]["meta_path"]],
    6: [f"{PROC}/shap_data.json"],
    7: [CFG["data"]["fraud_rings_file"]],
}


def is_done(step_num: int) -> bool:
    return all(Path(f).exists() for f in CHECKPOINTS[step_num])


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def skip(name: str):
    print(f"\n  [✓ already done — skipping]  {name}")


def main(skip_gnn: bool = False, force: bool = False):
    os.makedirs(PROC,  exist_ok=True)
    os.makedirs(SAVED, exist_ok=True)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    if force or not is_done(1):
        step("1 / 7  Preprocessing")
        from src.data.preprocess import run as preprocess
        preprocess()
    else:
        skip("1 / 7  Preprocessing")

    # ── Step 2: Build graph ───────────────────────────────────────────────────
    if force or not is_done(2):
        step("2 / 7  Building graph")
        from src.data.graph_builder import run as build_graph
        build_graph()
    else:
        skip("2 / 7  Building graph")

    # ── Step 3: LightGBM ─────────────────────────────────────────────────────
    if force or not is_done(3):
        step("3 / 7  Training LightGBM baseline")
        from src.models.tabular_baseline import train as train_lgbm
        train_lgbm()
    else:
        skip("3 / 7  Training LightGBM baseline")

    # ── Step 4: GNN ───────────────────────────────────────────────────────────
    if skip_gnn:
        if not is_done(4):
            print("\n  [--skip-gnn] Copying LightGBM preds as GNN stand-in")
            lgbm_val  = np.load(f"{PROC}/lgbm_val_preds.npy")
            lgbm_test = np.load(f"{PROC}/lgbm_test_preds.npy")
            np.save(f"{PROC}/gnn_val_preds.npy",  lgbm_val)
            np.save(f"{PROC}/gnn_test_preds.npy", lgbm_test)
            # Create a dummy gnn_model.pt so checkpoint is satisfied
            import torch
            torch.save({"skip_gnn": True}, CFG["model"]["gnn_path"])
        else:
            skip("4 / 7  GNN (--skip-gnn, already have preds)")
    elif force or not is_done(4):
        step("4 / 7  Training GNN (GAT)  [~10–20 min on GPU]")
        from src.models.gnn_model import train as train_gnn
        train_gnn()
    else:
        skip("4 / 7  Training GNN (GAT)")

    # ── Step 5: Ensemble ──────────────────────────────────────────────────────
    if force or not is_done(5):
        step("5 / 7  Stacking ensemble + final metrics")
        from src.models.ensemble import train as train_ensemble
        train_ensemble()
    else:
        skip("5 / 7  Stacking ensemble + final metrics")

    # ── Step 6: SHAP ──────────────────────────────────────────────────────────
    if force or not is_done(6):
        step("6 / 7  SHAP explainability")
        from src.evaluation.explainability import run as run_shap
        run_shap()
    else:
        skip("6 / 7  SHAP explainability")

    # ── Step 7: Fraud rings ───────────────────────────────────────────────────
    if force or not is_done(7):
        step("7 / 7  Fraud ring detection")
        from src.visualization.graph_viz import run as run_rings
        run_rings()
    else:
        skip("7 / 7  Fraud ring detection")

    step("Done!")
    print("\nAll artifacts saved. Start the dashboard with:")
    print("  uvicorn webapp.app:app --reload")
    print("\nFor GitHub Pages deployment, run:")
    print("  python export_static.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN training (uses LightGBM-only ensemble)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoints and redo all steps")
    args = parser.parse_args()
    main(skip_gnn=args.skip_gnn, force=args.force)
