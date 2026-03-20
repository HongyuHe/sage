"""
Derive clean-trace percentile thresholds for Sage shield feature predicates.

Example usage:
python scripts/gen_sage_shield_thresholds.py \
  --dataset attacks/output/shield-dataset/rl-constrained-300k/sage_shield_dataset.csv \
  --out attacks/output/shield-dataset/rl-constrained-300k/clean_feature_thresholds.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import repo_root_from_script, resolve_repo_path
else:
    from ._trace_attack_common import repo_root_from_script, resolve_repo_path

from sage_rl.shield.features import FEATURE_COLUMNS


def _percentiles(values: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    clean_values = np.asarray(values, dtype=np.float64)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return {f"p{int(p)}": float("nan") for p in percentiles}
    return {
        f"p{int(percentile)}": float(np.percentile(clean_values, int(percentile)))
        for percentile in percentiles
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clean-trace percentile thresholds for Sage shield features.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--percentiles", type=str, default="90,95")
    args = parser.parse_args()

    repo_root = os.path.abspath(str(args.repo_root))
    dataset_path = resolve_repo_path(repo_root, str(args.dataset))
    out_path = resolve_repo_path(repo_root, str(args.out))
    percentiles = [int(item.strip()) for item in str(args.percentiles).split(",") if item.strip()]

    df = pd.read_csv(dataset_path)
    trace_type_counts = (
        df["trace_type"].value_counts(dropna=False).to_dict() if "trace_type" in df.columns else {}
    )
    df = df[(df["trace_type"] == "clean") & (df.get("has_env_error", 0) == 0)]
    if df.empty:
        raise RuntimeError(
            "no clean, error-free rows found in shield dataset; "
            f"trace_type counts: {trace_type_counts}. "
            "Regenerate the dataset without `--adv-only-rollout` before deriving clean thresholds."
        )

    rows: list[dict[str, Any]] = []
    for feature_name in FEATURE_COLUMNS:
        if feature_name not in df.columns:
            continue
        row = {"feature": str(feature_name)}
        row.update(_percentiles(df[feature_name].to_numpy(dtype=np.float64, copy=False), percentiles))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
