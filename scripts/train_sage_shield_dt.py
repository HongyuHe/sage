"""
Train threshold-predicate decision trees for a Sage risk classifier + directional shield.

Example usage:
time python scripts/train_sage_shield_dt.py \
  --dataset attacks/output/shield-dataset/gap-constrained-3baselines_300k/sage_shield_dataset.csv \
  --thresholds attacks/output/shield-dataset/gap-constrained-3baselines_300k/clean_feature_thresholds.csv \
  --out-dir attacks/output/shield-rules/gap-constrained-3baselines_300k
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import repo_root_from_script, resolve_repo_path, save_json, utc_now_iso
else:
    from ._trace_attack_common import repo_root_from_script, resolve_repo_path, save_json, utc_now_iso

from sage_rl.shield.features import FEATURE_COLUMNS
from sage_rl.shield.labels import ACTIVE_LABEL, INACTIVE_LABEL, RISKY_LABEL, SAFE_LABEL, weak_direction_labels

_RAW_CATEGORICAL_FEATURES: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Predicate:
    name: str
    feature: str
    threshold_col: str
    threshold_value: float


@dataclass(frozen=True)
class _TreeTrainingConfig:
    backend: str
    seed: int
    max_depth: int | None
    max_leaf_nodes: int | None
    min_samples_leaf: int
    h2o_ntrees: int
    h2o_max_depth: int
    h2o_min_rows: int
    h2o_sample_rate: float
    h2o_mtries: int | None


def _threshold_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if str(column).startswith("p")]


def _load_threshold_predicates(
    thresholds_csv: str,
    *,
    threshold_cols: list[str] | None,
) -> tuple[list[_Predicate], dict[str, dict[str, float]]]:
    df = pd.read_csv(thresholds_csv)
    threshold_columns = _threshold_columns(df)
    if not threshold_columns:
        raise RuntimeError("no percentile columns found in threshold CSV")
    if threshold_cols is None:
        use_cols = list(threshold_columns)
    else:
        missing = [column for column in threshold_cols if column not in threshold_columns]
        if missing:
            raise RuntimeError(f"unknown threshold columns: {missing}")
        use_cols = list(threshold_cols)

    predicates: list[_Predicate] = []
    threshold_map: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        feature = str(row["feature"])
        if feature not in FEATURE_COLUMNS:
            continue
        threshold_map.setdefault(feature, {})
        for column in use_cols:
            raw_value = row.get(column)
            if not np.isfinite(float(raw_value)):
                continue
            threshold_value = float(raw_value)
            threshold_map[feature][str(column)] = threshold_value
            predicates.append(
                _Predicate(
                    name=f"{feature}__gt__{column}",
                    feature=feature,
                    threshold_col=str(column),
                    threshold_value=threshold_value,
                )
            )
    return predicates, threshold_map


def _build_predicate_matrix(df: pd.DataFrame, *, predicates: list[_Predicate]) -> tuple[pd.DataFrame, dict[str, _Predicate]]:
    out_data: dict[str, Any] = {}
    pred_map: dict[str, _Predicate] = {}
    for predicate in predicates:
        if predicate.feature not in df.columns:
            continue
        values = df[predicate.feature].astype(float).to_numpy(dtype=np.float64, copy=False)
        mask = np.where(np.isfinite(values), values > float(predicate.threshold_value), 0.0)
        out_data[str(predicate.name)] = mask.astype(np.int8)
        pred_map[str(predicate.name)] = predicate
    if not out_data:
        raise RuntimeError("no predicate features were created")
    return pd.DataFrame(out_data), pred_map


def _build_raw_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out_data: dict[str, Any] = {}
    for feature_name in FEATURE_COLUMNS:
        if feature_name not in df.columns:
            continue
        values = pd.to_numeric(df[feature_name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        out_data[str(feature_name)] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not out_data:
        raise RuntimeError("no raw feature columns were found in the dataset")
    return pd.DataFrame(out_data)


def _positive_rules_for_constant_label(*, num_rows: int, positive_active: bool) -> list[dict[str, Any]]:
    if not bool(positive_active):
        return []
    return [{"atoms": [], "purity": 1.0, "support": float(max(int(num_rows), 0))}]


def _label_risk(df: pd.DataFrame, *, risk_gap_pct: float, baseline_score_floor: float) -> np.ndarray:
    gap_pct = df["hard_gap_percent"].astype(float).to_numpy(dtype=np.float64, copy=False)
    baseline_score = df["hard_baseline_score"].astype(float).to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(gap_pct) & np.isfinite(baseline_score) & (baseline_score >= float(baseline_score_floor))
    return np.where(valid & (gap_pct >= float(risk_gap_pct)), RISKY_LABEL, SAFE_LABEL).astype(np.int32)


def _label_direction(df: pd.DataFrame, *, risky_mask: np.ndarray, action_margin: float) -> tuple[np.ndarray, np.ndarray]:
    sage_previous_action = df["sage_previous_action"].astype(float).to_numpy(dtype=np.float64, copy=False)
    best_previous_action = df["best_baseline_previous_action"].astype(float).to_numpy(dtype=np.float64, copy=False)
    backoff = np.zeros_like(risky_mask, dtype=np.int32)
    push = np.zeros_like(risky_mask, dtype=np.int32)
    for index, risky in enumerate(risky_mask.tolist()):
        backoff_label, push_label = weak_direction_labels(
            risky=bool(risky),
            sage_previous_action=float(sage_previous_action[index]),
            best_baseline_previous_action=float(best_previous_action[index]),
            action_margin=float(action_margin),
        )
        backoff[index] = int(backoff_label)
        push[index] = int(push_label)
    return backoff, push


def _train_sklearn_tree(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    max_depth: int | None,
    max_leaf_nodes: int | None,
    min_samples_leaf: int,
):
    try:
        from sklearn.tree import DecisionTreeClassifier  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for shield rule training. "
            "Install it in the active environment, e.g. `.venv/bin/pip install scikit-learn`."
        ) from exc

    model = DecisionTreeClassifier(
        random_state=int(seed),
        max_depth=int(max_depth) if max_depth is not None else None,
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes is not None else None,
        min_samples_leaf=int(min_samples_leaf),
    )
    model.fit(x, y)
    return model


def _train_h2o_tree(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    label_col: str,
    categorical_feature_names: set[str] | None,
    enforce_binary_categorical: bool,
    seed: int,
    ntrees: int,
    max_depth: int,
    min_rows: int,
    sample_rate: float,
    mtries: int | None,
):
    try:
        import h2o  # type: ignore
        from h2o.estimators import H2ORandomForestEstimator  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "h2o is required for `--backend h2o`. "
            "Install it in the active environment, e.g. `.venv/bin/pip install h2o`."
        ) from exc

    feature_cols = list(x.columns)
    categorical_cols = {str(item) for item in (categorical_feature_names or set())}
    for column in feature_cols:
        if str(column) not in categorical_cols:
            continue
        series = x[str(column)]
        unique_values = {int(value) for value in pd.Series(series).dropna().unique().tolist()}
        if bool(enforce_binary_categorical) and not unique_values.issubset({0, 1}):
            raise RuntimeError(
                f"H2O backend expects binary categorical features, but column {column!r} "
                f"has non-binary values: {sorted(unique_values)}"
            )

    df_xy = x.copy()
    for column in feature_cols:
        if str(column) in categorical_cols:
            df_xy[str(column)] = df_xy[str(column)].astype("category")
    df_xy[str(label_col)] = pd.Series(np.asarray(y, dtype=np.int32), index=df_xy.index).astype("category")

    frame = h2o.H2OFrame(df_xy)
    for column in feature_cols:
        if str(column) in categorical_cols:
            frame[str(column)] = frame[str(column)].asfactor()
    frame[str(label_col)] = frame[str(label_col)].asfactor()

    params: dict[str, Any] = {
        "ntrees": int(ntrees),
        "max_depth": int(max_depth),
        "min_rows": int(min_rows),
        "sample_rate": float(sample_rate),
        "seed": int(seed),
    }
    if mtries is not None:
        params["mtries"] = int(mtries)

    model = H2ORandomForestEstimator(**params)
    model.train(x=list(x.columns), y=str(label_col), training_frame=frame)
    return model


def _rule_atoms_from_predicate(predicate: _Predicate, *, is_true_branch: bool) -> dict[str, Any]:
    if is_true_branch:
        return {"feature": str(predicate.feature), "op": "gt", "value": float(predicate.threshold_value)}
    return {"feature": str(predicate.feature), "op": "le", "value": float(predicate.threshold_value)}


def _rule_atom_from_raw_feature(*, feature_name: str, threshold_value: float, is_true_branch: bool) -> dict[str, Any]:
    if bool(is_true_branch):
        return {"feature": str(feature_name), "op": "gt", "value": float(threshold_value)}
    return {"feature": str(feature_name), "op": "le", "value": float(threshold_value)}


def _extract_positive_rules(
    model: Any,
    *,
    feature_cols: list[str],
    pred_map: dict[str, _Predicate],
    positive_class: int,
    leaf_purity: float,
) -> list[dict[str, Any]]:
    tree = model.tree_
    feature_names = list(feature_cols)
    rules: list[dict[str, Any]] = []

    def recurse(node_id: int, path_atoms: list[dict[str, Any]]) -> None:
        if int(tree.feature[node_id]) < 0:
            counts = np.asarray(tree.value[node_id][0], dtype=np.float64)
            total = float(np.sum(counts))
            if total <= 0.0:
                return
            probs = counts / total
            pred_index = int(np.argmax(probs))
            klass = int(model.classes_[pred_index])
            if klass != int(positive_class):
                return
            if float(np.max(probs)) < float(leaf_purity):
                return
            normalized_atoms = sorted(path_atoms, key=lambda atom: (str(atom["feature"]), str(atom["op"]), float(atom["value"])))
            rules.append({"atoms": normalized_atoms, "purity": float(np.max(probs)), "support": float(total)})
            return

        feature_index = int(tree.feature[node_id])
        predicate = pred_map[str(feature_names[feature_index])]
        recurse(
            int(tree.children_left[node_id]),
            path_atoms + [_rule_atoms_from_predicate(predicate, is_true_branch=False)],
        )
        recurse(
            int(tree.children_right[node_id]),
            path_atoms + [_rule_atoms_from_predicate(predicate, is_true_branch=True)],
        )

    recurse(0, [])

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        signature = json.dumps(rule["atoms"], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(rule)
    return deduped


def _extract_positive_rules_raw(
    model: Any,
    *,
    feature_cols: list[str],
    positive_class: int,
    leaf_purity: float,
) -> list[dict[str, Any]]:
    tree = model.tree_
    feature_names = list(feature_cols)
    rules: list[dict[str, Any]] = []

    def recurse(node_id: int, path_atoms: list[dict[str, Any]]) -> None:
        if int(tree.feature[node_id]) < 0:
            counts = np.asarray(tree.value[node_id][0], dtype=np.float64)
            total = float(np.sum(counts))
            if total <= 0.0:
                return
            probs = counts / total
            pred_index = int(np.argmax(probs))
            klass = int(model.classes_[pred_index])
            if klass != int(positive_class):
                return
            if float(np.max(probs)) < float(leaf_purity):
                return
            normalized_atoms = sorted(path_atoms, key=lambda atom: (str(atom["feature"]), str(atom["op"]), float(atom["value"])))
            rules.append({"atoms": normalized_atoms, "purity": float(np.max(probs)), "support": float(total)})
            return

        feature_index = int(tree.feature[node_id])
        feature_name = str(feature_names[feature_index])
        threshold_value = float(tree.threshold[node_id])
        recurse(
            int(tree.children_left[node_id]),
            path_atoms + [_rule_atom_from_raw_feature(feature_name=feature_name, threshold_value=threshold_value, is_true_branch=False)],
        )
        recurse(
            int(tree.children_right[node_id]),
            path_atoms + [_rule_atom_from_raw_feature(feature_name=feature_name, threshold_value=threshold_value, is_true_branch=True)],
        )

    recurse(0, [])

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        signature = json.dumps(rule["atoms"], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(rule)
    return deduped


def _extract_positive_rules_from_h2o(
    model: Any,
    *,
    pred_map: dict[str, _Predicate],
    label_name: str,
    positive_class: int,
    leaf_purity: float,
) -> list[dict[str, Any]]:
    try:
        from h2o.tree import H2OTree  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "h2o is required for extracting rules from an H2O model. "
            "Install it in the active environment, e.g. `.venv/bin/pip install h2o`."
        ) from exc

    names = list(model._model_json["output"]["names"])
    domains = list(model._model_json["output"]["domains"])
    if str(label_name) not in names:
        raise RuntimeError(f"H2O model does not expose label column name: {label_name}")

    domain_by_name = {str(name): domain for name, domain in zip(names, domains)}
    label_domain = domain_by_name[str(label_name)]
    if label_domain is None:
        raise RuntimeError(f"H2O model does not expose a categorical domain for {label_name}")

    positive_str = str(int(positive_class))
    label_domain = [str(item) for item in list(label_domain)]
    if positive_str not in label_domain:
        raise RuntimeError(f"positive class {positive_str} not found in H2O label domain: {label_domain}")

    is_binomial = len(label_domain) == 2
    domain0 = str(label_domain[0]) if is_binomial else None
    rules: list[dict[str, Any]] = []

    def _normalized_levels(varname: str, levels: list[Any]) -> set[str]:
        domain = domain_by_name.get(str(varname))
        values: set[str] = set()
        for raw in levels:
            try:
                index = int(raw)
            except Exception:
                values.add(str(raw))
                continue
            if domain is not None and 0 <= index < len(domain):
                values.add(str(domain[index]))
            else:
                values.add(str(index))
        return values

    def _branch_atom(predicate: _Predicate, *, selected_levels: set[str]) -> dict[str, Any]:
        if selected_levels == {"1"}:
            return _rule_atoms_from_predicate(predicate, is_true_branch=True)
        if selected_levels == {"0"}:
            return _rule_atoms_from_predicate(predicate, is_true_branch=False)
        raise RuntimeError(
            f"unexpected categorical levels for predicate split {predicate.name}: {sorted(selected_levels)}"
        )

    def recurse(node: Any, path_atoms: list[dict[str, Any]]) -> None:
        if node.__class__.__name__ == "H2OLeafNode":
            purity = float(node.prediction)
            if is_binomial and domain0 is not None and positive_str != domain0:
                purity = 1.0 - purity
            if float(purity) < float(leaf_purity):
                return
            normalized_atoms = sorted(path_atoms, key=lambda atom: (str(atom["feature"]), str(atom["op"]), float(atom["value"])))
            rules.append({"atoms": normalized_atoms, "purity": float(purity), "support": 0.0})
            return

        predicate_name = str(node.split_feature)
        if predicate_name not in pred_map:
            raise RuntimeError(f"unexpected H2O split feature not in predicate map: {predicate_name}")
        predicate = pred_map[predicate_name]

        left_levels = _normalized_levels(predicate_name, list(node.left_levels or []))
        right_levels = _normalized_levels(predicate_name, list(node.right_levels or []))
        if left_levels:
            left_atom = _branch_atom(predicate, selected_levels=left_levels)
            right_atom = _branch_atom(predicate, selected_levels={"0", "1"} - left_levels)
        elif right_levels:
            right_atom = _branch_atom(predicate, selected_levels=right_levels)
            left_atom = _branch_atom(predicate, selected_levels={"0", "1"} - right_levels)
        else:
            threshold = float(node.threshold)
            if threshold < 0.0 or threshold > 1.0:
                raise RuntimeError(f"unexpected numeric threshold for predicate feature {predicate_name}: {threshold}")
            left_atom = _rule_atoms_from_predicate(predicate, is_true_branch=False)
            right_atom = _rule_atoms_from_predicate(predicate, is_true_branch=True)

        recurse(node.left_child, path_atoms + [left_atom])
        recurse(node.right_child, path_atoms + [right_atom])

    if is_binomial:
        tree_class = "0"
        tree = H2OTree(model=model, tree_number=0, tree_class=tree_class)
        recurse(tree.root_node, [])
    else:
        tree = H2OTree(model=model, tree_number=0, tree_class=positive_str)
        recurse(tree.root_node, [])

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        signature = json.dumps(rule["atoms"], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(rule)
    return deduped


def _extract_positive_rules_from_h2o_raw(
    model: Any,
    *,
    label_name: str,
    positive_class: int,
    leaf_purity: float,
) -> list[dict[str, Any]]:
    try:
        from h2o.tree import H2OTree  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "h2o is required for extracting rules from an H2O model. "
            "Install it in the active environment, e.g. `.venv/bin/pip install h2o`."
        ) from exc

    names = list(model._model_json["output"]["names"])
    domains = list(model._model_json["output"]["domains"])
    if str(label_name) not in names:
        raise RuntimeError(f"H2O model does not expose label column name: {label_name}")

    domain_by_name = {str(name): domain for name, domain in zip(names, domains)}
    label_domain = domain_by_name[str(label_name)]
    if label_domain is None:
        raise RuntimeError(f"H2O model does not expose a categorical domain for {label_name}")

    positive_str = str(int(positive_class))
    label_domain = [str(item) for item in list(label_domain)]
    if positive_str not in label_domain:
        raise RuntimeError(f"positive class {positive_str} not found in H2O label domain: {label_domain}")

    is_binomial = len(label_domain) == 2
    domain0 = str(label_domain[0]) if is_binomial else None
    rules: list[dict[str, Any]] = []

    def recurse(node: Any, path_atoms: list[dict[str, Any]]) -> None:
        if node.__class__.__name__ == "H2OLeafNode":
            purity = float(node.prediction)
            if is_binomial and domain0 is not None and positive_str != domain0:
                purity = 1.0 - purity
            if float(purity) < float(leaf_purity):
                return
            normalized_atoms = sorted(path_atoms, key=lambda atom: (str(atom["feature"]), str(atom["op"]), float(atom["value"])))
            rules.append({"atoms": normalized_atoms, "purity": float(purity), "support": 0.0})
            return

        if node.left_levels or node.right_levels:
            raise RuntimeError(
                "raw-feature H2O rule extraction encountered a categorical split. "
                "Current Sage raw shield features are expected to be continuous."
            )
        feature_name = str(node.split_feature)
        threshold_value = float(node.threshold)
        recurse(
            node.left_child,
            path_atoms + [_rule_atom_from_raw_feature(feature_name=feature_name, threshold_value=threshold_value, is_true_branch=False)],
        )
        recurse(
            node.right_child,
            path_atoms + [_rule_atom_from_raw_feature(feature_name=feature_name, threshold_value=threshold_value, is_true_branch=True)],
        )

    if is_binomial:
        tree = H2OTree(model=model, tree_number=0, tree_class="0")
    else:
        tree = H2OTree(model=model, tree_number=0, tree_class=positive_str)
    recurse(tree.root_node, [])

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        signature = json.dumps(rule["atoms"], sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(rule)
    return deduped


def _human_rule_text(label: str, rules: list[dict[str, Any]]) -> str:
    lines: list[str] = [f"{label} rules ({len(rules)}):"]
    for index, rule in enumerate(rules, start=1):
        atoms = rule.get("atoms", [])
        antecedent = " and ".join(
            f"{atom['feature']} {atom['op']} {atom['value']:.6g}" for atom in atoms
        ) or "true"
        lines.append(f"  {index:02d}. if {antecedent} then {label}")
    return "\n".join(lines) + "\n"


def _train_tree_model(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    label_name: str,
    train_cfg: _TreeTrainingConfig,
    categorical_feature_names: set[str] | None = None,
    enforce_binary_categorical: bool = False,
):
    if str(train_cfg.backend) == "sklearn":
        return _train_sklearn_tree(
            x,
            y,
            seed=int(train_cfg.seed),
            max_depth=train_cfg.max_depth,
            max_leaf_nodes=train_cfg.max_leaf_nodes,
            min_samples_leaf=int(train_cfg.min_samples_leaf),
        )
    if str(train_cfg.backend) == "h2o":
        return _train_h2o_tree(
            x,
            y,
            label_col=str(label_name),
            categorical_feature_names={str(item) for item in (categorical_feature_names or set())},
            enforce_binary_categorical=bool(enforce_binary_categorical),
            seed=int(train_cfg.seed),
            ntrees=int(train_cfg.h2o_ntrees),
            max_depth=int(train_cfg.h2o_max_depth),
            min_rows=int(train_cfg.h2o_min_rows),
            sample_rate=float(train_cfg.h2o_sample_rate),
            mtries=int(train_cfg.h2o_mtries) if train_cfg.h2o_mtries is not None else None,
        )
    raise RuntimeError(f"unsupported backend: {train_cfg.backend}")


def _extract_positive_rules_from_model(
    model: Any | None,
    *,
    backend: str,
    feature_mode: str,
    feature_cols: list[str],
    pred_map: dict[str, _Predicate],
    positive_class: int,
    leaf_purity: float,
    num_rows: int,
    positive_count: int,
    label_name: str,
) -> list[dict[str, Any]]:
    if int(positive_count) <= 0:
        return []
    if int(positive_count) >= int(num_rows):
        return _positive_rules_for_constant_label(num_rows=int(num_rows), positive_active=True)
    if model is None:
        return []
    if str(backend) == "sklearn":
        if str(feature_mode) == "raw":
            return _extract_positive_rules_raw(
                model,
                feature_cols=feature_cols,
                positive_class=int(positive_class),
                leaf_purity=float(leaf_purity),
            )
        return _extract_positive_rules(
            model,
            feature_cols=feature_cols,
            pred_map=pred_map,
            positive_class=int(positive_class),
            leaf_purity=float(leaf_purity),
        )
    if str(backend) == "h2o":
        if str(feature_mode) == "raw":
            return _extract_positive_rules_from_h2o_raw(
                model,
                label_name=str(label_name),
                positive_class=int(positive_class),
                leaf_purity=float(leaf_purity),
            )
        return _extract_positive_rules_from_h2o(
            model,
            pred_map=pred_map,
            label_name=str(label_name),
            positive_class=int(positive_class),
            leaf_purity=float(leaf_purity),
        )
    raise RuntimeError(f"unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decision-tree rules for Sage risk + directional shielding.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=["sklearn", "h2o"], default="h2o")
    parser.add_argument("--risk-feature-mode", choices=["raw", "predicate"], default="raw")
    parser.add_argument("--risk-gap-pct", type=float, default=20.0)
    parser.add_argument("--baseline-score-floor", type=float, default=0.3)
    parser.add_argument("--action-margin", type=float, default=0.15)
    parser.add_argument("--threshold-cols", type=str, default="p90,p95")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--max-leaf-nodes", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--leaf-purity", type=float, default=0.99)
    parser.add_argument("--h2o-ntrees", type=int, default=1)
    parser.add_argument("--h2o-max-depth", type=int, default=200)
    parser.add_argument("--h2o-min-rows", type=int, default=1)
    parser.add_argument("--h2o-sample-rate", type=float, default=1.0)
    parser.add_argument("--h2o-mtries", type=int, default=None)
    parser.add_argument("--history-len", type=int, default=4)
    args = parser.parse_args()

    repo_root = os.path.abspath(str(args.repo_root))
    dataset_path = resolve_repo_path(repo_root, str(args.dataset))
    thresholds_path = resolve_repo_path(repo_root, str(args.thresholds))
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    df = df[df.get("has_env_error", 0) == 0].copy()
    if df.empty:
        raise RuntimeError("dataset is empty after filtering invalid rows")

    predicates, threshold_map = _load_threshold_predicates(
        thresholds_path,
        threshold_cols=[item.strip() for item in str(args.threshold_cols).split(",") if item.strip()] or None,
    )
    x_pred, pred_map = _build_predicate_matrix(df, predicates=predicates)
    x_risk = _build_raw_feature_matrix(df) if str(args.risk_feature_mode) == "raw" else x_pred
    train_cfg = _TreeTrainingConfig(
        backend=str(args.backend),
        seed=int(args.seed),
        max_depth=int(args.max_depth),
        max_leaf_nodes=int(args.max_leaf_nodes),
        min_samples_leaf=int(args.min_samples_leaf),
        h2o_ntrees=int(args.h2o_ntrees),
        h2o_max_depth=int(args.h2o_max_depth),
        h2o_min_rows=int(args.h2o_min_rows),
        h2o_sample_rate=float(args.h2o_sample_rate),
        h2o_mtries=int(args.h2o_mtries) if args.h2o_mtries is not None else None,
    )
    risk_y = _label_risk(
        df,
        risk_gap_pct=float(args.risk_gap_pct),
        baseline_score_floor=float(args.baseline_score_floor),
    )
    risky_mask = risk_y == int(RISKY_LABEL)
    backoff_y, push_y = _label_direction(df, risky_mask=risky_mask, action_margin=float(args.action_margin))
    h2o = None
    if str(args.backend) == "h2o":
        try:
            import h2o  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "h2o is required for `--backend h2o`. Install it in the active environment, "
                "e.g. `.venv/bin/pip install h2o`."
            ) from exc
        try:
            from h2o.exceptions import H2ODependencyWarning  # type: ignore

            warnings.filterwarnings("ignore", category=H2ODependencyWarning)
        except Exception:
            pass
        try:
            h2o.init()
        except Exception as exc:
            raise RuntimeError(
                "failed to initialize H2O for `--backend h2o`. "
                "Make sure the active environment has both the `h2o` Python package "
                "and a working Java runtime (JRE/JDK)."
            ) from exc

    try:
        risk_model = None
        num_risk_positive = int(np.sum(risk_y == int(RISKY_LABEL)))
        if 0 < num_risk_positive < int(x_pred.shape[0]):
            risk_model = _train_tree_model(
                x_risk,
                risk_y,
                label_name="risk_label",
                train_cfg=train_cfg,
                categorical_feature_names=(
                    {name for name in _RAW_CATEGORICAL_FEATURES if name in x_risk.columns}
                    if str(args.risk_feature_mode) == "raw"
                    else set(x_risk.columns)
                ),
                enforce_binary_categorical=bool(str(args.risk_feature_mode) == "predicate"),
            )

        risky_subset = x_pred.loc[risky_mask].reset_index(drop=True)
        backoff_subset = backoff_y[risky_mask]
        push_subset = push_y[risky_mask]
        backoff_model = None
        push_model = None
        if risky_subset.shape[0] > 0:
            num_backoff_positive = int(np.sum(backoff_subset == ACTIVE_LABEL))
            num_push_positive = int(np.sum(push_subset == ACTIVE_LABEL))
            if 0 < num_backoff_positive < int(risky_subset.shape[0]):
                backoff_model = _train_tree_model(
                    risky_subset,
                    backoff_subset,
                    label_name="backoff_label",
                    train_cfg=train_cfg,
                    categorical_feature_names=set(risky_subset.columns),
                    enforce_binary_categorical=True,
                )
            if 0 < num_push_positive < int(risky_subset.shape[0]):
                push_model = _train_tree_model(
                    risky_subset,
                    push_subset,
                    label_name="push_label",
                    train_cfg=train_cfg,
                    categorical_feature_names=set(risky_subset.columns),
                    enforce_binary_categorical=True,
                )
        else:
            num_backoff_positive = 0
            num_push_positive = 0

        risk_rules = _extract_positive_rules_from_model(
            risk_model,
            backend=str(args.backend),
            feature_mode=str(args.risk_feature_mode),
            feature_cols=list(x_risk.columns),
            pred_map=pred_map,
            positive_class=RISKY_LABEL,
            leaf_purity=float(args.leaf_purity),
            num_rows=int(x_risk.shape[0]),
            positive_count=num_risk_positive,
            label_name="risk_label",
        )
        backoff_rules = _extract_positive_rules_from_model(
            backoff_model,
            backend=str(args.backend),
            feature_mode="predicate",
            feature_cols=list(risky_subset.columns),
            pred_map=pred_map,
            positive_class=ACTIVE_LABEL,
            leaf_purity=float(args.leaf_purity),
            num_rows=int(risky_subset.shape[0]),
            positive_count=int(num_backoff_positive),
            label_name="backoff_label",
        )
        push_rules = _extract_positive_rules_from_model(
            push_model,
            backend=str(args.backend),
            feature_mode="predicate",
            feature_cols=list(risky_subset.columns),
            pred_map=pred_map,
            positive_class=ACTIVE_LABEL,
            leaf_purity=float(args.leaf_purity),
            num_rows=int(risky_subset.shape[0]),
            positive_count=int(num_push_positive),
            label_name="push_label",
        )
    finally:
        if h2o is not None:
            h2o.shutdown(prompt=False)

    bundle = {
        "version": 1,
        "feature_names": list(FEATURE_COLUMNS),
        "history_len": int(args.history_len),
        "risk": {"rules": risk_rules},
        "backoff": {"rules": backoff_rules},
        "push": {"rules": push_rules},
        "metadata": {
            "created_at_utc": utc_now_iso(),
            "backend": str(args.backend),
            "risk_feature_mode": str(args.risk_feature_mode),
            "direction_feature_mode": "predicate",
            "dataset_path": dataset_path,
            "thresholds_path": thresholds_path,
            "threshold_map": threshold_map,
            "risk_gap_pct": float(args.risk_gap_pct),
            "baseline_score_floor": float(args.baseline_score_floor),
            "action_margin": float(args.action_margin),
            "num_rows": int(df.shape[0]),
            "num_risky_rows": int(np.sum(risky_mask)),
            "num_backoff_positive": int(np.sum(backoff_y == ACTIVE_LABEL)),
            "num_push_positive": int(np.sum(push_y == ACTIVE_LABEL)),
            "h2o_ntrees": int(args.h2o_ntrees),
            "h2o_max_depth": int(args.h2o_max_depth),
            "h2o_min_rows": int(args.h2o_min_rows),
            "h2o_sample_rate": float(args.h2o_sample_rate),
            "h2o_mtries": int(args.h2o_mtries) if args.h2o_mtries is not None else None,
        },
    }

    rules_json_path = os.path.join(out_dir, "sage_directional_shield_rules.json")
    save_json(rules_json_path, bundle)
    with open(os.path.join(out_dir, "sage_directional_shield_rules.txt"), "w", encoding="utf-8") as file_obj:
        file_obj.write(_human_rule_text("risky", risk_rules))
        file_obj.write("\n")
        file_obj.write(_human_rule_text("back_off", backoff_rules))
        file_obj.write("\n")
        file_obj.write(_human_rule_text("push_harder", push_rules))
    print(rules_json_path)


if __name__ == "__main__":
    main()
