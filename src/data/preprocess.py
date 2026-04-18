"""
Feature engineering and train/val/test splitting for IEEE-CIS Fraud Detection.
Temporal split is used to prevent data leakage (no random shuffle).
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Columns used for graph construction (entity linkage)
GRAPH_COLS = ["card1", "card2", "card3", "card4", "card5", "card6",
              "addr1", "addr2", "P_emaildomain", "R_emaildomain",
              "DeviceType", "DeviceInfo"]

# High-cardinality categoricals to frequency-encode
HIGH_CARD_CATS = ["P_emaildomain", "R_emaildomain", "DeviceInfo",
                  "id_31", "id_33"]

# Low-cardinality categoricals to label-encode
LOW_CARD_CATS = ["ProductCD", "card4", "card6", "DeviceType",
                 "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def load_raw(cfg: dict) -> pd.DataFrame:
    """Merge transaction and identity tables on TransactionID."""
    txn = pd.read_csv(cfg["data"]["raw_dir"] + "/train_transaction.csv")
    idf = pd.read_csv(cfg["data"]["raw_dir"] + "/train_identity.csv")
    df = txn.merge(idf, on="TransactionID", how="left")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} cols")
    return df


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, amount, and aggregation features."""
    new_cols = {}

    # Time features (TransactionDT is seconds since some reference point)
    new_cols["hour"] = (df["TransactionDT"] // 3600) % 24
    new_cols["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7
    new_cols["week"] = df["TransactionDT"] // (3600 * 24 * 7)

    # Amount features
    new_cols["log_amount"] = np.log1p(df["TransactionAmt"])
    new_cols["amount_cents"] = (df["TransactionAmt"] * 100 % 100).round(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Card-level aggregations (mean, std, count of amount per card1)
    card_stats = df.groupby("card1")["TransactionAmt"].agg(
        card1_mean_amt="mean", card1_std_amt="std", card1_count="count"
    ).reset_index()
    df = df.merge(card_stats, on="card1", how="left")

    # Addr-level aggregation
    addr_stats = df.groupby("addr1")["TransactionAmt"].agg(
        addr1_mean_amt="mean", addr1_count="count"
    ).reset_index()
    df = df.merge(addr_stats, on="addr1", how="left")

    # Frequency encoding for high-cardinality categoricals
    freq_cols = {}
    for col in HIGH_CARD_CATS:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            freq_cols[f"{col}_freq"] = df[col].map(freq).fillna(0).astype("float32")
    if freq_cols:
        df = pd.concat([df, pd.DataFrame(freq_cols, index=df.index)], axis=1)

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """Label-encode low-cardinality categoricals. Returns df + encoders dict."""
    if encoders is None:
        encoders = {}
    for col in LOW_CARD_CATS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("nan")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype("int16")
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col]).astype("int16")
    return df, encoders


def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Split by TransactionDT order — no shuffling to avoid leakage."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    print(f"Split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    print(f"Fraud rate → train: {train.isFraud.mean():.4f} | val: {val.isFraud.mean():.4f} | test: {test.isFraud.mean():.4f}")
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return model feature columns (exclude IDs, target, raw categoricals)."""
    exclude = {"TransactionID", "isFraud", "TransactionDT"} | set(HIGH_CARD_CATS)
    # Exclude raw string columns (object dtype) that weren't encoded
    obj_cols = set(df.select_dtypes(include=["object", "string"]).columns)
    return [c for c in df.columns if c not in exclude and c not in obj_cols]


def run():
    cfg = load_config()
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)

    df = load_raw(cfg)
    df = reduce_mem(df)
    df = engineer_features(df)

    train, val, test = temporal_split(
        df,
        cfg["split"]["train_ratio"],
        cfg["split"]["val_ratio"],
    )

    # Fit encoders on train only, apply to val/test
    train, encoders = encode_categoricals(train, fit=True)
    val, _ = encode_categoricals(val, encoders=encoders, fit=False)
    test, _ = encode_categoricals(test, encoders=encoders, fit=False)

    feature_cols = get_feature_cols(train)
    print(f"Feature count: {len(feature_cols)}")

    # Save
    train.to_parquet(cfg["data"]["train_file"], index=False)
    val.to_parquet(cfg["data"]["val_file"], index=False)
    test.to_parquet(cfg["data"]["test_file"], index=False)
    joblib.dump({"encoders": encoders, "feature_cols": feature_cols},
                cfg["data"]["processed_dir"] + "/meta.joblib")

    print("Preprocessing complete.")
    return train, val, test, feature_cols


if __name__ == "__main__":
    run()
