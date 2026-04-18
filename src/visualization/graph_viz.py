"""
Fraud ring detection via Louvain community detection on the card-transaction graph.

Steps:
1. Build a card-to-card co-occurrence graph (two cards share an edge if they
   share a terminal/device/email in the transaction data).
2. Run Louvain community detection.
3. Identify communities with high fraud rates (fraud rings).
4. Export top fraud rings as D3.js-compatible JSON.
"""

import json
import yaml
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain   # python-louvain
from pathlib import Path
from collections import defaultdict

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_card_cooccurrence_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build a card co-occurrence graph:
      nodes = card1 values (unique cards)
      edges = two cards share a P_emaildomain or DeviceInfo value
    Edge weight = number of shared entity connections.
    """
    G = nx.Graph()

    # Add all cards as nodes with fraud rate attribute
    card_stats = df.groupby("card1").agg(
        fraud_rate=("isFraud", "mean"),
        tx_count=("isFraud", "count"),
    ).reset_index()

    for _, row in card_stats.iterrows():
        G.add_node(int(row["card1"]),
                   fraud_rate=float(row["fraud_rate"]),
                   tx_count=int(row["tx_count"]))

    # Connect cards that share P_emaildomain
    for entity_col in ["P_emaildomain", "DeviceType"]:
        if entity_col not in df.columns:
            continue
        grp = df.dropna(subset=[entity_col, "card1"]).groupby(entity_col)["card1"].apply(list)
        for cards in grp:
            cards = list(set(cards))
            for i in range(len(cards)):
                for j in range(i + 1, len(cards)):
                    c1, c2 = int(cards[i]), int(cards[j])
                    if G.has_edge(c1, c2):
                        G[c1][c2]["weight"] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)

    return G


def detect_fraud_rings(G: nx.Graph, min_size: int, min_fraud_rate: float) -> list:
    """Run Louvain community detection and return high-fraud communities."""
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)

    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    rings = []
    for comm_id, nodes in communities.items():
        if len(nodes) < min_size:
            continue
        fraud_rates = [G.nodes[n].get("fraud_rate", 0) for n in nodes]
        avg_fraud_rate = float(np.mean(fraud_rates))
        if avg_fraud_rate < min_fraud_rate:
            continue

        subgraph = G.subgraph(nodes)
        rings.append({
            "community_id":   comm_id,
            "size":           len(nodes),
            "avg_fraud_rate": round(avg_fraud_rate, 4),
            "nodes":          nodes,
            "edges":          list(subgraph.edges(data=True)),
        })

    rings.sort(key=lambda r: r["avg_fraud_rate"], reverse=True)
    return rings


def export_for_d3(rings: list, top_k: int = 5) -> dict:
    """Convert top fraud rings to D3.js force-directed graph format."""
    all_nodes = []
    all_links = []
    node_set  = set()

    for ring in rings[:top_k]:
        group = ring["community_id"]
        for n in ring["nodes"]:
            if n not in node_set:
                fraud_rate = ring["avg_fraud_rate"]
                all_nodes.append({
                    "id":         n,
                    "group":      group,
                    "fraud_rate": fraud_rate,
                    "label":      f"Card {n}",
                })
                node_set.add(n)
        for src, dst, data in ring["edges"]:
            all_links.append({
                "source": src,
                "target": dst,
                "weight": data.get("weight", 1),
            })

    return {
        "nodes": all_nodes,
        "links": all_links,
        "summary": [
            {
                "rank":           i + 1,
                "community_id":   r["community_id"],
                "size":           r["size"],
                "avg_fraud_rate": r["avg_fraud_rate"],
            }
            for i, r in enumerate(rings[:top_k])
        ],
    }


def run():
    cfg = load_config()
    ring_cfg = cfg["fraud_ring"]

    train_df = pd.read_parquet(cfg["data"]["train_file"])

    print("Building card co-occurrence graph...")
    G = build_card_cooccurrence_graph(train_df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Running Louvain community detection...")
    rings = detect_fraud_rings(
        G,
        min_size=ring_cfg["min_community_size"],
        min_fraud_rate=ring_cfg["min_fraud_rate"],
    )
    print(f"Found {len(rings)} fraud ring communities")

    if rings:
        print(f"Top ring: {rings[0]['size']} cards, {rings[0]['avg_fraud_rate']:.1%} fraud rate")

    d3_data = export_for_d3(rings, top_k=5)

    out_path = cfg["data"]["fraud_rings_file"]
    with open(out_path, "w") as f:
        json.dump(d3_data, f, indent=2)

    print(f"Fraud ring data saved to {out_path}")
    return d3_data


if __name__ == "__main__":
    run()
