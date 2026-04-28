"""
CognitiveSync — Edge Signal Extractor
T-039: Derive collaboration relationships between employees
       from cross-platform interaction signals
"""

from __future__ import annotations

import pandas as pd

EPSILON = 1e-8


def load_edges(interactions_path: str) -> pd.DataFrame:
    """
    Load interaction signals and produce normalised undirected edges.

    Supported interaction types:
        MEETING           — shared calendar events
        SLACK_DM          — direct message / thread exchange
        GITHUB_PR         — PR review relationship
        GITHUB_CO_COMMIT  — co-commits to same repo

    Each interaction type is normalised independently using the
    org-level max weight for that type. Edges are then made
    undirected by merging (A→B) and (B→A) pairs, keeping the
    max weight for each unordered pair.

    Args:
        interactions_path: Path to interactions.csv

    Returns:
        pd.DataFrame with columns [source, target, weight]
        All weights in [0, 1]. One row per unique employee pair.
    """
    df = pd.read_csv(interactions_path)
    df = df.rename(columns={
        "source_pseudo_id": "source",
        "target_pseudo_id": "target",
    })

    if df.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    # Normalise weight per interaction type by org-level max
    for itype in df["interaction_type"].unique():
        mask = df["interaction_type"] == itype
        type_max = df.loc[mask, "weight"].max()
        df.loc[mask, "weight"] = df.loc[mask, "weight"] / (type_max + EPSILON)

    # Make undirected — add reverse direction then take max per pair
    reverse = df[["target", "source", "weight"]].rename(
        columns={"target": "source", "source": "target"}
    )
    combined = pd.concat([df[["source", "target", "weight"]], reverse])

    # Sort each pair so (A, B) and (B, A) collapse to the same key
    combined["pair"] = combined.apply(
        lambda r: tuple(sorted([r["source"], r["target"]])), axis=1
    )

    edges = (
        combined.groupby("pair")["weight"]
        .max()
        .reset_index()
    )

    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])

    unpacked = pd.DataFrame(edges["pair"].tolist(), index=edges.index, columns=["source", "target"])
    edges = edges.drop(columns=["pair"])
    edges["source"] = unpacked["source"]
    edges["target"] = unpacked["target"]
    edges = edges[["source", "target", "weight"]].copy()
    edges["weight"] = edges["weight"].clip(0.0, 1.0)

    return edges