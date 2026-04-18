"""
Graph Attention Network (GAT) on the heterogeneous transaction graph.

Architecture:
  - HeteroData with node types: transaction, card, addr, email, device
  - 3-layer HeteroConv wrapping GATConv per edge type
  - Transaction node classification (isFraud)
  - Focal loss for class imbalance
  - Mini-batch training via NeighborLoader
"""

import os
import yaml
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.loader import NeighborLoader

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ── HeteroGAT Model ───────────────────────────────────────────────────────────
class HeteroGAT(nn.Module):
    def __init__(self, metadata, hidden_channels: int, num_layers: int,
                 heads: int, dropout: float, in_channels_dict: dict):
        super().__init__()

        # Input projection for each node type to same hidden dimension
        self.input_proj = nn.ModuleDict({
            ntype: Linear(in_ch, hidden_channels)
            for ntype, in_ch in in_channels_dict.items()
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GATConv(
                        hidden_channels,
                        hidden_channels // heads,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                    for edge_type in metadata[1]
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        # Project all node types to hidden_channels
        h = {ntype: F.elu(self.input_proj[ntype](x))
             for ntype, x in x_dict.items()}

        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: F.elu(v) for k, v in h.items()}
            h = {k: F.dropout(v, p=self.dropout, training=self.training)
                 for k, v in h.items()}

        # Only classify transaction nodes
        return self.classifier(h["transaction"]).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    cfg = load_config()
    os.makedirs(cfg["model"]["saved_dir"], exist_ok=True)
    gnn_cfg = cfg["gnn"]

    graphs = torch.load(cfg["data"]["graph_file"], weights_only=False)
    train_graph = graphs["train"]
    val_graph   = graphs["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build in_channels_dict from graph
    in_channels_dict = {
        ntype: train_graph[ntype].x.shape[1]
        for ntype in train_graph.node_types
    }

    model = HeteroGAT(
        metadata=train_graph.metadata(),
        hidden_channels=gnn_cfg["hidden_channels"],
        num_layers=gnn_cfg["num_layers"],
        heads=gnn_cfg["heads"],
        dropout=gnn_cfg["dropout"],
        in_channels_dict=in_channels_dict,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_cfg["lr"])
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # NeighborLoader for scalable mini-batch training
    train_loader = NeighborLoader(
        train_graph,
        num_neighbors={key: gnn_cfg["num_neighbors"] for key in train_graph.edge_types},
        batch_size=gnn_cfg["batch_size"],
        input_nodes=("transaction", None),
        shuffle=True,
    )

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, gnn_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x_dict, batch.edge_index_dict)
            # batch_size nodes are the "seed" nodes with labels
            seed_logits = logits[:batch["transaction"].batch_size]
            seed_labels = batch["transaction"].y[:batch["transaction"].batch_size]
            loss = criterion(seed_logits, seed_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation (full-batch on CPU for simplicity since no GPU)
        model.eval()
        with torch.no_grad():
            val_logits = model(
                {k: v.to(device) for k, v in val_graph.x_dict.items()},
                {k: v.to(device) for k, v in val_graph.edge_index_dict.items()},
            )
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_labels = val_graph["transaction"].y.numpy()
            val_auc = roc_auc_score(val_labels, val_probs)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest Val AUC: {best_val_auc:.4f}")

    # Save best model
    torch.save({
        "state_dict": best_state,
        "metadata": train_graph.metadata(),
        "in_channels_dict": in_channels_dict,
        "config": gnn_cfg,
    }, cfg["model"]["gnn_path"])

    # Save val predictions for ensembling
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(
            val_graph.x_dict, val_graph.edge_index_dict
        )
        val_probs = torch.sigmoid(val_logits).numpy()
        test_graph = graphs["test"]
        test_logits = model(
            test_graph.x_dict, test_graph.edge_index_dict
        )
        test_probs = torch.sigmoid(test_logits).numpy()

    np.save(cfg["data"]["processed_dir"] + "/gnn_val_preds.npy",  val_probs)
    np.save(cfg["data"]["processed_dir"] + "/gnn_test_preds.npy", test_probs)

    test_labels = graphs["test"]["transaction"].y.numpy()
    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"Test AUC: {test_auc:.4f}")

    return model, {"gnn_val_auc": round(best_val_auc, 4), "gnn_test_auc": round(test_auc, 4)}


if __name__ == "__main__":
    train()
