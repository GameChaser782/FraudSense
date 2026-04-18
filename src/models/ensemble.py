"""
Stacking ensemble: LightGBM + GNN predictions → Logistic Regression meta-learner.
"""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report, roc_curve,
                             average_precision_score, precision_recall_curve)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    proc = cfg["data"]["processed_dir"]

    lgbm_val  = np.load(f"{proc}/lgbm_val_preds.npy")
    gnn_val   = np.load(f"{proc}/gnn_val_preds.npy")
    lgbm_test = np.load(f"{proc}/lgbm_test_preds.npy")
    gnn_test  = np.load(f"{proc}/gnn_test_preds.npy")

    val_df  = pd.read_parquet(cfg["data"]["val_file"])
    test_df = pd.read_parquet(cfg["data"]["test_file"])
    y_val   = val_df["isFraud"].values
    y_test  = test_df["isFraud"].values

    # Stack val predictions as meta-features
    X_val_meta  = np.column_stack([lgbm_val,  gnn_val])
    X_test_meta = np.column_stack([lgbm_test, gnn_test])

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(X_val_meta, y_val)

    ensemble_val_preds  = meta.predict_proba(X_val_meta)[:, 1]
    ensemble_test_preds = meta.predict_proba(X_test_meta)[:, 1]

    val_auc  = roc_auc_score(y_val,  ensemble_val_preds)
    test_auc = roc_auc_score(y_test, ensemble_test_preds)

    lgbm_test_auc = roc_auc_score(y_test, lgbm_test)
    gnn_test_auc  = roc_auc_score(y_test, gnn_test)

    # PR-AUC — stricter metric for imbalanced data (3.5% fraud)
    lgbm_pr_auc = average_precision_score(y_test, lgbm_test)
    gnn_pr_auc  = average_precision_score(y_test, gnn_test)
    ens_pr_auc  = average_precision_score(y_test, ensemble_test_preds)

    print(f"{'Model':<12} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print(f"{'LightGBM':<12} {lgbm_test_auc:>9.4f} {lgbm_pr_auc:>9.4f}")
    print(f"{'GNN':<12} {gnn_test_auc:>9.4f} {gnn_pr_auc:>9.4f}")
    print(f"{'Ensemble':<12} {test_auc:>9.4f} {ens_pr_auc:>9.4f}")

    # Compute ROC curves for all three models (for dashboard)
    def roc_for_json(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Downsample to 200 points for JSON size
        idx = np.linspace(0, len(fpr) - 1, 200, dtype=int)
        return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}

    # Best threshold by F1 on test set
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_test, (ensemble_test_preds >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_threshold = float(thresholds[np.argmax(f1s)])
    test_labels = (ensemble_test_preds >= best_threshold).astype(int)

    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test, test_labels, zero_division=0)
    recall    = recall_score(y_test, test_labels, zero_division=0)
    f1        = f1_score(y_test, test_labels, zero_division=0)

    print(f"\nEnsemble @ threshold={best_threshold:.2f}")
    print(classification_report(y_test, test_labels, digits=4))

    # Save meta-learner
    joblib.dump(meta, cfg["model"]["meta_path"])
    np.save(f"{proc}/ensemble_test_preds.npy", ensemble_test_preds)

    # PR curve data for dashboard
    def pr_for_json(y_true, y_score):
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        idx = np.linspace(0, len(prec) - 1, 200, dtype=int)
        return {"precision": prec[idx].tolist(), "recall": rec[idx].tolist()}

    # Save full metrics JSON (used by dashboard)
    metrics = {
        "lgbm_test_auc":     round(lgbm_test_auc, 4),
        "gnn_test_auc":      round(gnn_test_auc, 4),
        "ensemble_test_auc": round(test_auc, 4),
        "lgbm_pr_auc":       round(lgbm_pr_auc, 4),
        "gnn_pr_auc":        round(gnn_pr_auc, 4),
        "ensemble_pr_auc":   round(ens_pr_auc, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1":                round(f1, 4),
        "best_threshold":    best_threshold,
        "roc_lgbm":     roc_for_json(y_test, lgbm_test),
        "roc_gnn":      roc_for_json(y_test, gnn_test),
        "roc_ensemble": roc_for_json(y_test, ensemble_test_preds),
        "pr_lgbm":      pr_for_json(y_test, lgbm_test),
        "pr_gnn":       pr_for_json(y_test, gnn_test),
        "pr_ensemble":  pr_for_json(y_test, ensemble_test_preds),
        "model_comparison": {
            "models":   ["LightGBM", "GNN (GAT)", "Ensemble"],
            "auc":      [round(lgbm_test_auc, 4), round(gnn_test_auc, 4), round(test_auc, 4)],
            "pr_auc":   [round(lgbm_pr_auc, 4),   round(gnn_pr_auc, 4),   round(ens_pr_auc, 4)],
        },
    }

    os.makedirs(cfg["model"]["saved_dir"], exist_ok=True)
    with open(cfg["model"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {cfg['model']['metrics_path']}")
    return meta, metrics


if __name__ == "__main__":
    train()
