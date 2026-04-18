"""
SHAP-based explainability for the LightGBM model.
Generates:
  - Feature importance (mean |SHAP|)
  - SHAP waterfall data for sample transactions
  - Saved to data/processed/shap_*.json for dashboard consumption
"""

import json
import yaml
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
N_BACKGROUND = 500   # background samples for TreeExplainer
N_EXPLAIN    = 100   # transactions to explain (for dashboard waterfall examples)
TOP_FEATURES = 20    # how many features to show in importance plot


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run():
    cfg = load_config()
    meta     = joblib.load(cfg["data"]["processed_dir"] + "/meta.joblib")
    feat_cols = meta["feature_cols"]

    model = joblib.load(cfg["model"]["lgbm_path"])
    test_df = pd.read_parquet(cfg["data"]["test_file"])
    X_test = test_df[feat_cols].fillna(-999)

    print(f"Computing SHAP values on {N_EXPLAIN} test transactions...")
    explainer = shap.TreeExplainer(model)

    # Full SHAP for feature importance (use a sample for speed)
    sample = X_test.sample(n=min(N_BACKGROUND, len(X_test)), random_state=42)
    shap_vals = explainer.shap_values(sample)
    # LightGBM binary classifier returns list [neg_class, pos_class]; take pos class
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Feature importance: mean absolute SHAP
    mean_shap = np.abs(shap_vals).mean(axis=0)
    feat_imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": mean_shap,
    }).sort_values("importance", ascending=False).head(TOP_FEATURES)

    feat_imp = {
        "features": feat_imp_df["feature"].tolist(),
        "importance": feat_imp_df["importance"].round(4).tolist(),
    }

    # Waterfall data for a fraud sample and a legit sample
    fraud_idx  = test_df[test_df["isFraud"] == 1].index[:1]
    legit_idx  = test_df[test_df["isFraud"] == 0].index[:1]

    def waterfall_data(idx):
        row = X_test.loc[idx]
        sv_raw = explainer.shap_values(row)
        # LightGBM returns list [neg, pos]; take pos class, then first row
        if isinstance(sv_raw, list):
            sv = sv_raw[1][0]
        else:
            sv = sv_raw[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:10]
        return {
            "features":   [feat_cols[i] for i in top_idx],
            "shap_values": [round(float(sv[i]), 4) for i in top_idx],
            "feature_values": [round(float(row.iloc[0, i]), 4) for i in top_idx],
            "base_value": round(float(explainer.expected_value), 4),
        }

    waterfall_fraud = waterfall_data(fraud_idx)
    waterfall_legit = waterfall_data(legit_idx)

    out = {
        "feature_importance": feat_imp,
        "waterfall_fraud":    waterfall_fraud,
        "waterfall_legit":    waterfall_legit,
    }

    out_path = cfg["data"]["processed_dir"] + "/shap_data.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"SHAP data saved to {out_path}")
    print(f"Top 5 features: {feat_imp['features'][:5]}")
    return out


if __name__ == "__main__":
    run()
