"""
Example usage:
PYTHON_BIN=.venv/bin/python
$PYTHON_BIN scripts/prepare_trace_splits.py

Run from the repo root. This copies the source traces into:
  attacks/train
  attacks/test
and writes the corresponding manifest files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import materialize_trace_splits, repo_root_from_script, resolve_repo_path
else:
    from ._trace_attack_common import materialize_trace_splits, repo_root_from_script, resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--train-dir", type=str, default="attacks/train")
    parser.add_argument("--test-dir", type=str, default="attacks/test")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    train_dir = resolve_repo_path(repo_root, str(args.train_dir))
    test_dir = resolve_repo_path(repo_root, str(args.test_dir))
    payload = materialize_trace_splits(
        repo_root=repo_root,
        train_root=train_dir,
        test_root=test_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
