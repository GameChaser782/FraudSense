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

    # Only keep cards with at least 2 transactions to avoid noise
    card_stats = card_stats[card_stats["tx_count"] >= 2]
    valid_cards = set(card_stats["card1"].tolist())
    df = df[df["card1"].isin(valid_cards)]

    for _, row in card_stats.iterrows():
        G.add_node(int(row["card1"]),
                   fraud_rate=float(row["fraud_rate"]),
                   tx_count=int(row["tx_count"]))

    # Connect cards sharing the same entity value (email/device/addr).
    # Skip groups larger than MAX_GROUP_SIZE — large groups (e.g. all gmail.com
    # users) are not indicative of fraud rings and produce O(n²) edges.
    MAX_GROUP_SIZE = 20

    for entity_col in ["P_emaildomain", "addr1", "DeviceType"]:
        if entity_col not in df.columns:
            continue
        grp = df.dropna(subset=[entity_col, "card1"]).groupby(entity_col)["card1"].apply(
            lambda x: list(set(x.tolist()))
        )
        skipped = 0
        for cards in grp:
            if len(cards) > MAX_GROUP_SIZE:
                skipped += 1
                continue
            for i in range(len(cards)):
                for j in range(i + 1, len(cards)):
                    c1, c2 = int(cards[i]), int(cards[j])
                    if G.has_edge(c1, c2):
                        G[c1][c2]["weight"] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)
        if skipped:
            print(f"  {entity_col}: skipped {skipped} large groups (>{MAX_GROUP_SIZE} cards)")

    return G


def detect_fraud_rings(G: nx.Graph, min_size: int, min_fraud_rate: float) -> list:
    """Run Louvain community detection and return high-fraud communities."""
    # Only run on the connected subgraph — isolated nodes form trivial communities
    connected_nodes = [n for n, d in G.degree() if d > 0]
    G_conn = G.subgraph(connected_nodes)
    print(f"  Connected subgraph: {G_conn.number_of_nodes()} nodes, {G_conn.number_of_edges()} edges")

    if G_conn.number_of_nodes() == 0:
        print("  No connected nodes — graph too sparse.")
        return []

    partition = community_louvain.best_partition(G_conn, weight="weight", random_state=42)

    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    # Debug: show distribution of community fraud rates
    all_rates = []
    for comm_id, nodes in communities.items():
        if len(nodes) >= min_size:
            rates = [G.nodes[n].get("fraud_rate", 0) for n in nodes]
            all_rates.append((len(nodes), float(np.mean(rates))))
    if all_rates:
        rates_only = [r for _, r in all_rates]
        print(f"  Communities (size>={min_size}): {len(all_rates)} total | "
              f"fraud rate range: {min(rates_only):.3f}–{max(rates_only):.3f} | "
              f"median: {float(np.median(rates_only)):.3f}")

    candidates = []
    for comm_id, nodes in communities.items():
        if len(nodes) < min_size:
            continue
        fraud_rates = [G.nodes[n].get("fraud_rate", 0) for n in nodes]
        avg_fraud_rate = float(np.mean(fraud_rates))
        subgraph = G.subgraph(nodes)
        candidates.append({
            "community_id":   comm_id,
            "size":           len(nodes),
            "avg_fraud_rate": round(avg_fraud_rate, 4),
            "nodes":          nodes,
            "edges":          list(subgraph.edges(data=True)),
        })

    # Sort by fraud rate descending, return top 5 above min_fraud_rate
    # If fewer than 3 pass the threshold, fall back to top 5 by fraud rate
    candidates.sort(key=lambda r: r["avg_fraud_rate"], reverse=True)
    rings = [r for r in candidates if r["avg_fraud_rate"] >= min_fraud_rate]
    if len(rings) < 3:
        rings = candidates[:5]  # fallback: top 5 regardless of threshold

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
