"""
Derive clean-trace percentile thresholds for the legacy two-stage Sage shield pipeline.

Example usage:
python scripts/gen_sage_shield_thresholds.py \
  --dataset attacks/shield/shield-dataset/gap-constrained-all-loss_50ms_300k/sage_shield_dataset.csv \
  --out attacks/shield/shield-dataset/gap-constrained-all-loss_50ms_300k/clean_feature_thresholds.csv

python scripts/gen_sage_shield_thresholds.py \
  --dataset attacks/output/shield-dataset/hotnets19-loss_50ms_300k/sage_shield_dataset.csv \
  --out attacks/output/shield-dataset/hotnets19-loss_50ms_300k/clean_feature_thresholds.csv
"""

from __future__ import annotations

import argparse
import os
import sys


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import repo_root_from_script, resolve_repo_path
    from scripts.gen_sage_shield_dataset import write_clean_feature_thresholds_from_dataset
else:
    from ._trace_attack_common import repo_root_from_script, resolve_repo_path
    from .gen_sage_shield_dataset import write_clean_feature_thresholds_from_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clean-trace percentile thresholds for the legacy two-stage Sage shield pipeline.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--percentiles", type=str, default="10,25,90,95")
    args = parser.parse_args()

    repo_root = os.path.abspath(str(args.repo_root))
    dataset_path = resolve_repo_path(repo_root, str(args.dataset))
    out_path = resolve_repo_path(repo_root, str(args.out))
    percentiles = [int(item.strip()) for item in str(args.percentiles).split(",") if item.strip()]
    write_clean_feature_thresholds_from_dataset(
        dataset_path=dataset_path,
        out_path=out_path,
        percentiles=percentiles,
    )
    print(out_path)


if __name__ == "__main__":
    main()
