"""End-to-end demo: CSV → graph → GNN → cascade → printed results.

Reproducible seeded run using data/features.csv + data/interactions.csv.
Uses the latest model checkpoint (models/final-v1 via 'latest' symlink).

Usage:
    uv run python scripts/run_demo.py
    uv run python scripts/run_demo.py --top 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA = Path(__file__).parent.parent / "data"
MODELS = Path(__file__).parent.parent / "models"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"


def main(top_n: int = 10) -> int:
    from intelligence.cascade import CascadePropagator
    from intelligence.graph_builder import GraphBuilder
    from intelligence.inference import InferencePipeline

    features_path = str(DATA / "features.csv")
    interactions_path = str(DATA / "interactions.csv")

    print("\n── CognitiveSync Demo ──────────────────────────────────────────")

    # Step 1: Build graph
    print("Step 1/3  Building org graph from CSVs …", end="", flush=True)
    builder = GraphBuilder()
    built = builder.build_from_csv(features_path, interactions_path)
    print(f"  {built.nx_graph.number_of_nodes()} nodes, {built.nx_graph.number_of_edges()} edges")

    # Step 2: GNN inference
    print("Step 2/3  Running BurnoutGAT inference …", end="", flush=True)
    run_id = uuid4()
    pipeline = InferencePipeline(model_registry_path=MODELS, device="cuda")
    pipeline.load_model(version="latest")
    scored = pipeline.score(pyg_data=built.pyg_data, node_ids=built.node_ids, run_id=run_id)
    high_risk = [pid for pid, ns in scored.node_scores.items() if ns.burnout_score >= 0.70]
    print(f"  {len(scored.node_scores)} nodes scored, {len(high_risk)} high-risk (≥0.70)")

    # Step 3: Cascade propagation
    print("Step 3/3  Running cascade propagation …", end="", flush=True)
    propagator = CascadePropagator()
    burnout_scores = {pid: ns.burnout_score for pid, ns in scored.node_scores.items()}
    cascade = propagator.propagate(built.nx_graph, burnout_scores)
    affected = sum(1 for cr in cascade.values() if cr.cascade_risk > 0.0)
    print(f"  {affected} nodes have non-zero cascade risk\n")

    # Print top-N high risk nodes
    sorted_nodes = sorted(
        scored.node_scores.values(), key=lambda ns: ns.burnout_score, reverse=True
    )
    print(f"{'Rank':<5} {'pseudo_id':<38} {'score':>6} {'CI':>14} {'cascade':>8}  top feature")
    print("─" * 90)
    for rank, ns in enumerate(sorted_nodes[:top_n], start=1):
        pid = ns.pseudo_id
        cr = cascade.get(pid)
        cascade_risk = cr.cascade_risk if cr else 0.0
        top_feat = max(ns.top_features, key=ns.top_features.get) if ns.top_features else "—"  # type: ignore[arg-type]
        print(
            f"{rank:<5} {str(pid):<38} {ns.burnout_score:>6.3f} "
            f"[{ns.confidence_low:.2f},{ns.confidence_high:.2f}] "
            f"{cascade_risk:>8.3f}  {top_feat}"
        )

    # Save JSON artifact
    ARTIFACTS.mkdir(exist_ok=True)
    artifact = {
        "run_id": str(run_id),
        "node_count": len(scored.node_scores),
        "high_risk_count": len(high_risk),
        "cascade_affected_count": affected,
        "top_nodes": [
            {
                "pseudo_id": str(ns.pseudo_id),
                "burnout_score": ns.burnout_score,
                "confidence_low": ns.confidence_low,
                "confidence_high": ns.confidence_high,
                "cascade_risk": cascade.get(ns.pseudo_id).cascade_risk
                if cascade.get(ns.pseudo_id)
                else 0.0,
                "top_features": ns.top_features,
            }
            for ns in sorted_nodes[:top_n]
        ],
    }
    out_path = ARTIFACTS / "demo_results.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nArtifact saved → {out_path}")
    print("────────────────────────────────────────────────────────────────\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CognitiveSync end-to-end demo")
    parser.add_argument("--top", type=int, default=10, help="Top N nodes to display")
    args = parser.parse_args()
    sys.exit(main(top_n=args.top))
