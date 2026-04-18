"""
Builds a heterogeneous transaction graph for PyTorch Geometric.

Graph structure:
  - Node types: transaction, card, addr, email, device
  - Edge types:
      (transaction, uses_card,  card)
      (transaction, uses_addr,  addr)
      (transaction, uses_email, email)
      (transaction, uses_device, device)

Transaction node features = the tabular feature vector (float32).
Entity node features = fraud stats from TRAINING data only (4-D):
  [log_tx_count, fraud_rate, mean_log_amt, std_log_amt]
Using training stats for all splits prevents leakage.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

ENTITY_COLS = {
    "card1":         ("card",   "uses_card"),
    "addr1":         ("addr",   "uses_addr"),
    "P_emaildomain": ("email",  "uses_email"),
    "DeviceType":    ("device", "uses_device"),
}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_entity_index(series: pd.Series):
    """Map unique non-null values to integer IDs."""
    vals = series.dropna().unique()
    return {v: i for i, v in enumerate(vals)}


def compute_entity_stats(train_df: pd.DataFrame) -> dict:
    """
    Compute per-entity fraud statistics from TRAINING data only.
    Returns {col: DataFrame indexed by entity value}.
    Called once; results passed to build_graph for all splits.
    """
    stats = {}
    for col in ENTITY_COLS:
        if col not in train_df.columns:
            continue
        sub = train_df[["isFraud", "log_amount", col]].dropna(subset=[col])
        agg = sub.groupby(col).agg(
            fraud_rate   =("isFraud",    "mean"),
            mean_log_amt =("log_amount", "mean"),
            std_log_amt  =("log_amount", "std"),
            tx_count     =("isFraud",    "count"),
        )
        agg["log_tx_count"] = np.log1p(agg["tx_count"])
        agg["std_log_amt"]  = agg["std_log_amt"].fillna(0.0)
        stats[col] = agg[["log_tx_count", "fraud_rate", "mean_log_amt", "std_log_amt"]]
    return stats


def _standardize_frame(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """Z-score columns with train-only stats; zero out constant columns."""
    safe_std = std.replace(0, 1).fillna(1)
    standardized = (df - mean) / safe_std
    return standardized.fillna(0.0)


def build_graph(df: pd.DataFrame, feature_cols: list,
                entity_stats: dict,
                tx_feature_stats: dict,
                entity_feature_stats: dict) -> HeteroData:
    """
    Build HeteroData from a DataFrame.
    entity_stats must be computed from TRAINING data only to avoid leakage.
    """
    data = HeteroData()
    df = df.reset_index(drop=True)

    # ── Transaction node features & labels ──────────────────────────────────
    tx_features = df[feature_cols].fillna(0.0)
    tx_features = _standardize_frame(
        tx_features,
        tx_feature_stats["mean"],
        tx_feature_stats["std"],
    )
    feat_matrix = tx_features.values.astype("float32")
    data["transaction"].x = torch.tensor(feat_matrix, dtype=torch.float)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)
    data["transaction"].transaction_id = torch.tensor(
        df["TransactionID"].values, dtype=torch.long
    )

    # ── Entity nodes + edges ─────────────────────────────────────────────────
    for col, (node_type, edge_type) in ENTITY_COLS.items():
        if col not in df.columns:
            continue
        idx_map = _build_entity_index(df[col])
        if not idx_map:
            continue

        stat_df = entity_stats.get(col)
        n_entities = len(idx_map)

        if stat_df is not None:
            # 4-D features: log_tx_count, fraud_rate, mean_log_amt, std_log_amt
            feature_mean = entity_feature_stats[col]["mean"]
            feature_std = entity_feature_stats[col]["std"]
            global_defaults = feature_mean.values.astype("float32")  # fallback for unseen
            entity_feats = np.tile(global_defaults, (n_entities, 1))
            for v, i in idx_map.items():
                if v in stat_df.index:
                    entity_feats[i] = stat_df.loc[v].values.astype("float32")
            entity_feats = _standardize_frame(
                pd.DataFrame(entity_feats, columns=stat_df.columns),
                feature_mean,
                feature_std,
            ).values.astype("float32")
        else:
            # Fallback: 1-D log_count (should not happen)
            entity_feats = np.zeros((n_entities, 1), dtype="float32")
            vc = df[col].value_counts()
            for v, i in idx_map.items():
                entity_feats[i, 0] = np.log1p(vc.get(v, 1))

        data[node_type].x = torch.tensor(entity_feats, dtype=torch.float)

        # Edges: transaction → entity (non-null rows only)
        mask = df[col].notna()
        src = torch.tensor(df.index[mask].values, dtype=torch.long)
        dst = torch.tensor(df[col][mask].map(idx_map).values, dtype=torch.long)
        data["transaction", edge_type, node_type].edge_index = torch.stack([src, dst])
        data[node_type, f"rev_{edge_type}", "transaction"].edge_index = torch.stack([dst, src])

    return data


def run():
    cfg = load_config()
    meta = joblib.load(cfg["data"]["processed_dir"] + "/meta.joblib")
    feature_cols = meta["feature_cols"]

    train = pd.read_parquet(cfg["data"]["train_file"])
    val   = pd.read_parquet(cfg["data"]["val_file"])
    test  = pd.read_parquet(cfg["data"]["test_file"])
    tx_feature_stats = {
        "mean": train[feature_cols].fillna(0.0).mean(),
        "std": train[feature_cols].fillna(0.0).std().replace(0, 1).fillna(1),
    }

    # Compute entity stats from training data ONLY
    print("Computing entity statistics from training set...")
    entity_stats = compute_entity_stats(train)
    entity_feature_stats = {
        col: {
            "mean": st.mean(),
            "std": st.std().replace(0, 1).fillna(1),
        }
        for col, st in entity_stats.items()
    }
    for col, st in entity_stats.items():
        print(f"  {col}: {len(st)} entities, "
              f"fraud rate range {st['fraud_rate'].min():.3f}–{st['fraud_rate'].max():.3f}")

    print("Building train graph...")
    train_graph = build_graph(
        train, feature_cols, entity_stats, tx_feature_stats, entity_feature_stats
    )
    print("Building val graph...")
    val_graph   = build_graph(
        val, feature_cols, entity_stats, tx_feature_stats, entity_feature_stats
    )
    print("Building test graph...")
    test_graph  = build_graph(
        test, feature_cols, entity_stats, tx_feature_stats, entity_feature_stats
    )

    graphs = {"train": train_graph, "val": val_graph, "test": test_graph}
    torch.save(graphs, cfg["data"]["graph_file"])
    print(f"Saved graphs to {cfg['data']['graph_file']}")
    print(f"Train graph: {train_graph}")
    return graphs


if __name__ == "__main__":
    run()
