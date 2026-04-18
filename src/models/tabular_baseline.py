"""
LightGBM tabular baseline for IEEE-CIS fraud detection.
Uses temporal split (no leakage), class_weight handling for imbalance,
early stopping on validation AUC.
"""

import json
import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    os.makedirs(cfg["model"]["saved_dir"], exist_ok=True)

    meta = joblib.load(cfg["data"]["processed_dir"] + "/meta.joblib")
    feature_cols = meta["feature_cols"]

    train_df = pd.read_parquet(cfg["data"]["train_file"])
    val_df   = pd.read_parquet(cfg["data"]["val_file"])
    test_df  = pd.read_parquet(cfg["data"]["test_file"])

    X_train, y_train = train_df[feature_cols].fillna(-999), train_df["isFraud"]
    X_val,   y_val   = val_df[feature_cols].fillna(-999),   val_df["isFraud"]
    X_test,  y_test  = test_df[feature_cols].fillna(-999),  test_df["isFraud"]

    lgb_cfg = cfg["lgbm"]
    params = {
        "objective":          "binary",
        "metric":             "auc",
        "learning_rate":      lgb_cfg["learning_rate"],
        "num_leaves":         lgb_cfg["num_leaves"],
        "max_depth":          lgb_cfg["max_depth"],
        "min_child_samples":  lgb_cfg["min_child_samples"],
        "feature_fraction":   lgb_cfg["feature_fraction"],
        "bagging_fraction":   lgb_cfg["bagging_fraction"],
        "bagging_freq":       lgb_cfg["bagging_freq"],
        "reg_alpha":          lgb_cfg["reg_alpha"],
        "reg_lambda":         lgb_cfg["reg_lambda"],
        "is_unbalance":       True,
        "verbosity":          -1,
        "n_jobs":             -1,
        "seed":               42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=True),
        lgb.log_evaluation(50),
    ]

    print("Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=lgb_cfg["n_estimators"],
        valid_sets=[dval],
        callbacks=callbacks,
    )

    val_preds  = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_auc  = roc_auc_score(y_val,  val_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    print(f"\nVal  AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    threshold = 0.5
    test_labels = (test_preds >= threshold).astype(int)
    print("\nTest classification report:")
    print(classification_report(y_test, test_labels, digits=4))

    # Save model and predictions
    joblib.dump(model, cfg["model"]["lgbm_path"])
    np.save(cfg["data"]["processed_dir"] + "/lgbm_val_preds.npy",  val_preds)
    np.save(cfg["data"]["processed_dir"] + "/lgbm_test_preds.npy", test_preds)

    metrics = {
        "lgbm_val_auc":  round(val_auc,  4),
        "lgbm_test_auc": round(test_auc, 4),
    }
    print(f"\nSaved model to {cfg['model']['lgbm_path']}")
    return model, metrics


if __name__ == "__main__":
    import os
    train()
