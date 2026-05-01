"""T-026: Validate synthetic data output.

Spot-checks:
- Node count matches requested N
- Label balance: burnout fraction within ±5% of configured rate
- Feature distributions: mean/std per class (burnout vs normal)
- Edge density: within-team ~40%, cross-team ~5% (±10% tolerance)
- Identifier pseudonymity: all IDs are UUID4, no collisions
Saves validation notes and sample metrics to artifacts/synthetic_validation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import UUID

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import SyntheticOrgGenerator, FEATURE_NAMES


def validate(n_employees: int = 200, seed: int = 42) -> dict[str, object]:
    gen = SyntheticOrgGenerator(n_employees=n_employees, seed=seed)
    org = gen.generate()

    results: dict[str, object] = {}
    failures: list[str] = []

    # --- Node count ---
    actual_nodes = len(org.employees)
    results["node_count"] = {"expected": n_employees, "actual": actual_nodes, "pass": actual_nodes == n_employees}
    if actual_nodes != n_employees:
        failures.append(f"Node count: expected {n_employees}, got {actual_nodes}")

    # --- Label balance ---
    burnout_rate = float(org.labels.mean())
    expected_rate = gen.burnout_fraction
    tol = 0.05
    label_pass = abs(burnout_rate - expected_rate) <= tol
    results["label_balance"] = {
        "expected_rate": expected_rate,
        "actual_rate": round(burnout_rate, 4),
        "tolerance": tol,
        "n_burnout": int(org.labels.sum()),
        "n_normal": int((org.labels == 0).sum()),
        "pass": label_pass,
    }
    if not label_pass:
        failures.append(
            f"Label balance: expected ~{expected_rate:.2f}, got {burnout_rate:.4f} (tol={tol})"
        )

    # --- Feature distributions ---
    burnout_mask = org.labels == 1
    normal_mask = org.labels == 0
    feat_stats: list[dict[str, object]] = []
    for j, fname in enumerate(FEATURE_NAMES):
        col = org.feature_matrix[:, j]
        b_mean = float(col[burnout_mask].mean()) if burnout_mask.any() else float("nan")
        n_mean = float(col[normal_mask].mean()) if normal_mask.any() else float("nan")
        feat_stats.append({
            "feature": fname,
            "burnout_mean": round(b_mean, 4),
            "normal_mean": round(n_mean, 4),
            "global_min": round(float(col.min()), 4),
            "global_max": round(float(col.max()), 4),
        })
        # All values must be in [0, 1]
        if not (0.0 <= col.min() and col.max() <= 1.0):
            failures.append(f"Feature {fname} out of [0,1]: min={col.min():.4f}, max={col.max():.4f}")

    results["feature_distributions"] = feat_stats

    # --- Edge density ---
    id_to_team = {emp.pseudo_id: emp.team_id for emp in org.employees}
    within_possible = 0
    cross_possible = 0
    within_actual = 0
    cross_actual = 0

    pseudo_ids = [emp.pseudo_id for emp in org.employees]
    n = len(pseudo_ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same = id_to_team[pseudo_ids[i]] == id_to_team[pseudo_ids[j]]
            if same:
                within_possible += 1
            else:
                cross_possible += 1

    edge_set = {(s, d) for s, d, _ in org.edges}
    for src, dst, _ in org.edges:
        same = id_to_team[src] == id_to_team[dst]
        if same:
            within_actual += 1
        else:
            cross_actual += 1

    within_density = within_actual / max(within_possible, 1)
    cross_density = cross_actual / max(cross_possible, 1)
    edge_tol = 0.10
    within_pass = abs(within_density - 0.40) <= edge_tol
    cross_pass = abs(cross_density - 0.05) <= edge_tol
    results["edge_density"] = {
        "within_team": {"expected": 0.40, "actual": round(within_density, 4), "pass": within_pass},
        "cross_team": {"expected": 0.05, "actual": round(cross_density, 4), "pass": cross_pass},
        "total_edges": len(org.edges),
    }
    if not within_pass:
        failures.append(f"Within-team density: expected ~0.40, got {within_density:.4f}")
    if not cross_pass:
        failures.append(f"Cross-team density: expected ~0.05, got {cross_density:.4f}")

    # --- Pseudonymity ---
    all_ids = [str(emp.pseudo_id) for emp in org.employees]
    unique_ids = set(all_ids)
    uuid_valid = all(len(uid) == 36 for uid in all_ids)  # UUID string format
    id_collision_free = len(unique_ids) == len(all_ids)
    results["pseudonymity"] = {
        "all_uuid_format": uuid_valid,
        "no_collisions": id_collision_free,
        "pass": uuid_valid and id_collision_free,
    }
    if not uuid_valid:
        failures.append("Some IDs are not valid UUID strings")
    if not id_collision_free:
        failures.append(f"ID collisions detected: {len(all_ids) - len(unique_ids)} duplicates")

    # --- Raw signals keyed by pseudo_id ---
    raw_key_match = set(org.raw_signals.keys()) == set(str(e.pseudo_id) for e in org.employees)
    results["raw_signals_coverage"] = {"all_employees_covered": raw_key_match, "pass": raw_key_match}
    if not raw_key_match:
        failures.append("raw_signals keys do not match employee IDs")

    results["overall"] = {
        "pass": len(failures) == 0,
        "failures": failures,
        "n_employees": n_employees,
        "seed": seed,
    }

    return results


def main() -> int:
    print("T-026: Validating synthetic data...")
    results = validate(n_employees=200, seed=42)

    out_path = Path(__file__).parent.parent / "artifacts" / "synthetic_validation.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")

    overall = results["overall"]  # type: ignore[index]
    if overall["pass"]:  # type: ignore[index]
        print("PASS: All synthetic data checks passed.")
        return 0
    else:
        print("FAIL:")
        for f in overall["failures"]:  # type: ignore[index]
            print(f"  - {f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
