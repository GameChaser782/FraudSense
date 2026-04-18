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
Entity node features = frequency embeddings (1-D: log-count).
"""

import yaml
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_entity_index(series: pd.Series):
    """Map unique non-null values to integer IDs."""
    vals = series.dropna().unique()
    return {v: i for i, v in enumerate(vals)}


def build_graph(df: pd.DataFrame, feature_cols: list) -> HeteroData:
    """
    Build HeteroData from a DataFrame.
    Expects isFraud column for labels.
    Returns a HeteroData object.
    """
    data = HeteroData()
    n = len(df)
    df = df.reset_index(drop=True)

    # ── Transaction node features & labels ──────────────────────────────────
    feat_matrix = df[feature_cols].fillna(0).values.astype("float32")
    data["transaction"].x = torch.tensor(feat_matrix, dtype=torch.float)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)
    data["transaction"].transaction_id = torch.tensor(
        df["TransactionID"].values, dtype=torch.long
    )

    # ── Helper: build entity nodes + edges ──────────────────────────────────
    def add_entity(col: str, node_type: str, edge_type: str):
        if col not in df.columns:
            return
        idx_map = _build_entity_index(df[col])
        if not idx_map:
            return

        # Node features: log(count) of each entity value
        counts = df[col].map(df[col].value_counts()).fillna(1)
        log_counts = np.log1p(df[col].map(df[col].value_counts()).fillna(1).values)
        n_entities = len(idx_map)
        entity_feats = np.zeros((n_entities, 1), dtype="float32")
        for v, i in idx_map.items():
            entity_feats[i, 0] = np.log1p(df[col].value_counts().get(v, 1))
        data[node_type].x = torch.tensor(entity_feats, dtype=torch.float)

        # Edges: transaction → entity (only for non-null rows)
        mask = df[col].notna()
        src = torch.tensor(df.index[mask].values, dtype=torch.long)
        dst = torch.tensor(
            df[col][mask].map(idx_map).values, dtype=torch.long
        )
        data["transaction", edge_type, node_type].edge_index = torch.stack([src, dst])
        # Reverse edges too (for message passing in both directions)
        data[node_type, f"rev_{edge_type}", "transaction"].edge_index = torch.stack([dst, src])

    add_entity("card1",          "card",   "uses_card")
    add_entity("addr1",          "addr",   "uses_addr")
    add_entity("P_emaildomain",  "email",  "uses_email")
    add_entity("DeviceType",     "device", "uses_device")

    return data


def run():
    cfg = load_config()
    meta = joblib.load(cfg["data"]["processed_dir"] + "/meta.joblib")
    feature_cols = meta["feature_cols"]

    # Build on training set only (to avoid test leakage in graph structure)
    train = pd.read_parquet(cfg["data"]["train_file"])
    val   = pd.read_parquet(cfg["data"]["val_file"])
    test  = pd.read_parquet(cfg["data"]["test_file"])

    # Build separate graphs per split (node IDs are local to each split)
    print("Building train graph...")
    train_graph = build_graph(train, feature_cols)
    print("Building val graph...")
    val_graph   = build_graph(val,   feature_cols)
    print("Building test graph...")
    test_graph  = build_graph(test,  feature_cols)

    graphs = {"train": train_graph, "val": val_graph, "test": test_graph}
    torch.save(graphs, cfg["data"]["graph_file"])
    print(f"Saved graphs to {cfg['data']['graph_file']}")
    print(f"Train graph: {train_graph}")
    return graphs


if __name__ == "__main__":
    run()
