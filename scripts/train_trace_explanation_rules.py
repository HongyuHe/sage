"""
Train trace-level explanation rules for difference, challenge, mechanism, and prototype tasks.

Example usage:
time python scripts/train_trace_explanation_rules.py \
  --dataset attacks/output/trace-explanations/gap-constrained-3baselines_300k/trace_explanation_dataset.csv \
  --out-dir attacks/output/trace-explanations/gap-constrained-3baselines_300k/rules
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

from attacks.analysis import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    NUMERIC_FEATURE_COLUMNS,
    challenge_label,
    difference_label,
    mechanism_label_map,
    mechanism_shares,
)


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


@dataclass(frozen=True)
class _EncodedFeatureSpec:
    encoded_name: str
    feature_name: str
    kind: str
    category_value: str | None = None


@dataclass(frozen=True)
class _SplitSpec:
    train_indices: np.ndarray
    val_indices: np.ndarray


def _require_sklearn_component() -> dict[str, Any]:
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.tree import DecisionTreeClassifier  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for trace explanation rule training. "
            "Install it in the active environment, e.g. `.venv/bin/pip install scikit-learn`."
        ) from exc
    return {
        "KMeans": KMeans,
        "StandardScaler": StandardScaler,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
    }


def _coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna("missing").astype(str)
    return out


def _prepare_sklearn_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[_EncodedFeatureSpec]]:
    numeric = pd.DataFrame(index=df.index)
    specs: list[_EncodedFeatureSpec] = []
    for column in NUMERIC_FEATURE_COLUMNS:
        if column in df.columns:
            numeric[str(column)] = pd.to_numeric(df[str(column)], errors="coerce").fillna(0.0).astype(float)
            specs.append(_EncodedFeatureSpec(encoded_name=str(column), feature_name=str(column), kind="numeric"))

    categorical = df[[column for column in CATEGORICAL_FEATURE_COLUMNS if column in df.columns]].copy()
    categorical = categorical.fillna("missing").astype(str)
    categorical_blocks: list[pd.DataFrame] = []
    for column in categorical.columns:
        series = categorical[str(column)]
        for category_value in sorted(series.unique().tolist()):
            encoded_name = f"{column}__is__{category_value}"
            categorical_blocks.append(
                pd.DataFrame({encoded_name: (series == str(category_value)).astype(np.int8)}, index=df.index)
            )
            specs.append(
                _EncodedFeatureSpec(
                    encoded_name=encoded_name,
                    feature_name=str(column),
                    kind="categorical_dummy",
                    category_value=str(category_value),
                )
            )
    if categorical_blocks:
        matrix = pd.concat([numeric, *categorical_blocks], axis=1)
    else:
        matrix = numeric
    if matrix.empty:
        raise RuntimeError("no explanation features were prepared for sklearn training")
    return matrix, specs


def _feature_frame_for_h2o(df: pd.DataFrame) -> pd.DataFrame:
    data: dict[str, Any] = {}
    for column in NUMERIC_FEATURE_COLUMNS:
        if column in df.columns:
            data[str(column)] = pd.to_numeric(df[str(column)], errors="coerce").fillna(0.0).astype(float)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column in df.columns:
            data[str(column)] = df[str(column)].fillna("missing").astype(str)
    out = pd.DataFrame(data, index=df.index)
    if out.empty:
        raise RuntimeError("no explanation features were prepared for h2o training")
    return out


def _stratified_split(y: np.ndarray, *, validation_fraction: float, seed: int) -> _SplitSpec:
    indices = np.arange(y.shape[0], dtype=np.int64)
    if y.shape[0] <= 1 or validation_fraction <= 0.0:
        return _SplitSpec(train_indices=indices, val_indices=np.asarray([], dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for klass in sorted(set(int(item) for item in y.tolist())):
        klass_indices = indices[y == int(klass)]
        shuffled = klass_indices.copy()
        rng.shuffle(shuffled)
        val_count = int(round(float(validation_fraction) * float(len(shuffled))))
        if val_count <= 0 and len(shuffled) > 1:
            val_count = 1
        if val_count >= len(shuffled):
            val_count = max(len(shuffled) - 1, 0)
        if val_count > 0:
            val_parts.append(shuffled[:val_count])
            train_parts.append(shuffled[val_count:])
        else:
            train_parts.append(shuffled)
    train_indices = np.sort(np.concatenate(train_parts)) if train_parts else np.asarray([], dtype=np.int64)
    val_indices = np.sort(np.concatenate(val_parts)) if val_parts else np.asarray([], dtype=np.int64)
    return _SplitSpec(train_indices=train_indices, val_indices=val_indices)


def _random_split(num_rows: int, *, validation_fraction: float, seed: int) -> _SplitSpec:
    indices = np.arange(int(num_rows), dtype=np.int64)
    if int(num_rows) <= 1 or validation_fraction <= 0.0:
        return _SplitSpec(train_indices=indices, val_indices=np.asarray([], dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    val_count = int(round(float(validation_fraction) * float(len(shuffled))))
    if val_count <= 0 and len(shuffled) > 1:
        val_count = 1
    if val_count >= len(shuffled):
        val_count = max(len(shuffled) - 1, 0)
    val_indices = np.sort(shuffled[:val_count]) if val_count > 0 else np.asarray([], dtype=np.int64)
    train_indices = np.sort(shuffled[val_count:]) if val_count > 0 else indices
    return _SplitSpec(train_indices=train_indices, val_indices=val_indices)


def _fit_sklearn_tree(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    max_depth: int | None,
    max_leaf_nodes: int | None,
    min_samples_leaf: int,
):
    components = _require_sklearn_component()
    model = components["DecisionTreeClassifier"](
        random_state=int(seed),
        max_depth=int(max_depth) if max_depth is not None else None,
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes is not None else None,
        min_samples_leaf=int(min_samples_leaf),
    )
    model.fit(x, y)
    return model


def _h2o_frame_from_features(
    feature_df: pd.DataFrame,
    *,
    label_name: str | None,
    labels: np.ndarray | None,
    categorical_feature_names: set[str],
):
    try:
        import h2o  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "h2o is required for `--backend h2o`. Install it in the active environment, e.g. `.venv/bin/pip install h2o`."
        ) from exc

    frame_df = feature_df.copy()
    if label_name is not None and labels is not None:
        frame_df[str(label_name)] = pd.Series(np.asarray(labels, dtype=np.int32), index=frame_df.index).astype(str)
    frame = h2o.H2OFrame(frame_df)
    for column in feature_df.columns:
        if str(column) in categorical_feature_names:
            frame[str(column)] = frame[str(column)].asfactor()
    if label_name is not None and labels is not None:
        frame[str(label_name)] = frame[str(label_name)].asfactor()
    return frame


def _fit_h2o_tree(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    label_name: str,
    categorical_feature_names: set[str],
    train_cfg: _TreeTrainingConfig,
):
    try:
        import h2o  # type: ignore
        from h2o.estimators import H2ORandomForestEstimator  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "h2o is required for `--backend h2o`. Install it in the active environment, e.g. `.venv/bin/pip install h2o`."
        ) from exc

    try:
        from h2o.exceptions import H2ODependencyWarning  # type: ignore

        warnings.filterwarnings("ignore", category=H2ODependencyWarning)
    except Exception:
        pass

    if not h2o.connection():
        try:
            h2o.init()
        except Exception as exc:
            raise RuntimeError(
                "failed to initialize H2O. Make sure the active environment has both the `h2o` package and a working Java runtime."
            ) from exc

    frame = _h2o_frame_from_features(
        x,
        label_name=str(label_name),
        labels=y,
        categorical_feature_names={str(item) for item in categorical_feature_names},
    )
    params: dict[str, Any] = {
        "ntrees": int(train_cfg.h2o_ntrees),
        "max_depth": int(train_cfg.h2o_max_depth),
        "min_rows": int(train_cfg.h2o_min_rows),
        "sample_rate": float(train_cfg.h2o_sample_rate),
        "seed": int(train_cfg.seed),
    }
    if train_cfg.h2o_mtries is not None:
        params["mtries"] = int(train_cfg.h2o_mtries)
    model = H2ORandomForestEstimator(**params)
    model.train(x=list(x.columns), y=str(label_name), training_frame=frame)
    return model


def _fit_tree_model(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    label_name: str,
    categorical_feature_names: set[str],
    train_cfg: _TreeTrainingConfig,
):
    if str(train_cfg.backend) == "sklearn":
        return _fit_sklearn_tree(
            x,
            y,
            seed=int(train_cfg.seed),
            max_depth=train_cfg.max_depth,
            max_leaf_nodes=train_cfg.max_leaf_nodes,
            min_samples_leaf=int(train_cfg.min_samples_leaf),
        )
    if str(train_cfg.backend) == "h2o":
        return _fit_h2o_tree(
            x,
            y,
            label_name=str(label_name),
            categorical_feature_names=categorical_feature_names,
            train_cfg=train_cfg,
        )
    raise RuntimeError(f"unsupported backend: {train_cfg.backend}")


def _predict_sklearn(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=np.int32)


def _predict_h2o(model: Any, x: pd.DataFrame, *, categorical_feature_names: set[str]) -> np.ndarray:
    frame = _h2o_frame_from_features(
        x,
        label_name=None,
        labels=None,
        categorical_feature_names=categorical_feature_names,
    )
    prediction_df = model.predict(frame).as_data_frame()
    if "predict" in prediction_df.columns:
        return prediction_df["predict"].astype(int).to_numpy(dtype=np.int32, copy=False)
    first_column = prediction_df.columns[0]
    return prediction_df[first_column].astype(int).to_numpy(dtype=np.int32, copy=False)


def _predict_tree_model(model: Any, x: pd.DataFrame, *, backend: str, categorical_feature_names: set[str]) -> np.ndarray:
    if str(backend) == "sklearn":
        return _predict_sklearn(model, x)
    if str(backend) == "h2o":
        return _predict_h2o(model, x, categorical_feature_names=categorical_feature_names)
    raise RuntimeError(f"unsupported backend: {backend}")


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    components = _require_sklearn_component()
    if y_true.size == 0:
        return {
            "num_rows": 0.0,
            "positive_fraction": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    return {
        "num_rows": float(y_true.size),
        "positive_fraction": float(np.mean(y_true == 1)),
        "accuracy": float(components["accuracy_score"](y_true, y_pred)),
        "precision": float(components["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(components["recall_score"](y_true, y_pred, zero_division=0)),
        "f1": float(components["f1_score"](y_true, y_pred, zero_division=0)),
    }


def _constant_rules(*, positive_active: bool, num_rows: int) -> list[dict[str, Any]]:
    if not bool(positive_active):
        return []
    return [{"atoms": [], "purity": 1.0, "support": float(max(int(num_rows), 0))}]


def _numeric_atom(*, feature_name: str, threshold_value: float, is_true_branch: bool) -> dict[str, Any]:
    if bool(is_true_branch):
        return {"feature": str(feature_name), "op": "gt", "value": float(threshold_value)}
    return {"feature": str(feature_name), "op": "le", "value": float(threshold_value)}


def _categorical_dummy_atom(*, feature_name: str, category_value: str, is_true_branch: bool) -> dict[str, Any]:
    if bool(is_true_branch):
        return {"feature": str(feature_name), "op": "eq", "value": str(category_value)}
    return {"feature": str(feature_name), "op": "ne", "value": str(category_value)}


def _atom_sort_key(atom: dict[str, Any]) -> tuple[str, str, str]:
    value = atom.get("value")
    if isinstance(value, list):
        value_str = json.dumps(value)
    else:
        value_str = str(value)
    return str(atom.get("feature")), str(atom.get("op")), value_str


def _extract_rules_from_sklearn(
    model: Any,
    *,
    encoded_specs: list[_EncodedFeatureSpec],
    positive_class: int,
    leaf_purity: float,
) -> list[dict[str, Any]]:
    tree = model.tree_
    feature_names = [spec.encoded_name for spec in encoded_specs]
    spec_by_name = {spec.encoded_name: spec for spec in encoded_specs}
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
            if not klass == int(positive_class):
                return
            purity = float(np.max(probs))
            if purity < float(leaf_purity):
                return
            normalized_atoms = sorted(path_atoms, key=_atom_sort_key)
            rules.append({"atoms": normalized_atoms, "purity": purity, "support": float(total)})
            return

        feature_index = int(tree.feature[node_id])
        encoded_name = str(feature_names[feature_index])
        spec = spec_by_name[encoded_name]
        threshold_value = float(tree.threshold[node_id])
        if str(spec.kind) == "categorical_dummy":
            left_atom = _categorical_dummy_atom(
                feature_name=str(spec.feature_name),
                category_value=str(spec.category_value),
                is_true_branch=False,
            )
            right_atom = _categorical_dummy_atom(
                feature_name=str(spec.feature_name),
                category_value=str(spec.category_value),
                is_true_branch=True,
            )
        else:
            left_atom = _numeric_atom(feature_name=str(spec.feature_name), threshold_value=threshold_value, is_true_branch=False)
            right_atom = _numeric_atom(feature_name=str(spec.feature_name), threshold_value=threshold_value, is_true_branch=True)
        recurse(int(tree.children_left[node_id]), path_atoms + [left_atom])
        recurse(int(tree.children_right[node_id]), path_atoms + [right_atom])

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


def _h2o_probability_for_positive(node_prediction: float, *, positive_class: int, label_domain: list[str]) -> float:
    if len(label_domain) == 2:
        domain0 = str(label_domain[0])
        if str(int(positive_class)) == domain0:
            return float(node_prediction)
        return float(1.0 - node_prediction)
    return float(node_prediction)


def _extract_rules_from_h2o(
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
            "h2o is required for extracting rules from an H2O model. Install it in the active environment, e.g. `.venv/bin/pip install h2o`."
        ) from exc

    names = list(model._model_json["output"]["names"])
    domains = list(model._model_json["output"]["domains"])
    domain_by_name = {str(name): domain for name, domain in zip(names, domains)}
    label_domain_raw = domain_by_name.get(str(label_name))
    if label_domain_raw is None:
        raise RuntimeError(f"H2O model does not expose a categorical label domain for {label_name}")
    label_domain = [str(item) for item in list(label_domain_raw)]
    positive_str = str(int(positive_class))
    if positive_str not in label_domain:
        raise RuntimeError(f"positive class {positive_str} is not present in H2O label domain {label_domain}")

    def decode_levels(varname: str, raw_levels: list[Any]) -> list[str]:
        domain = domain_by_name.get(str(varname))
        values: list[str] = []
        for raw in raw_levels:
            try:
                index = int(raw)
            except Exception:
                values.append(str(raw))
                continue
            if domain is not None and 0 <= index < len(domain):
                values.append(str(domain[index]))
            else:
                values.append(str(index))
        return values

    def categorical_atom(*, feature_name: str, values: list[str], negate: bool) -> dict[str, Any]:
        unique_values = sorted({str(item) for item in values})
        if len(unique_values) == 1:
            return {
                "feature": str(feature_name),
                "op": "ne" if bool(negate) else "eq",
                "value": str(unique_values[0]),
            }
        return {
            "feature": str(feature_name),
            "op": "not_in" if bool(negate) else "in",
            "value": list(unique_values),
        }

    rules: list[dict[str, Any]] = []

    def recurse(node: Any, path_atoms: list[dict[str, Any]]) -> None:
        if node.__class__.__name__ == "H2OLeafNode":
            purity = _h2o_probability_for_positive(float(node.prediction), positive_class=int(positive_class), label_domain=label_domain)
            if purity < float(leaf_purity):
                return
            normalized_atoms = sorted(path_atoms, key=_atom_sort_key)
            rules.append({"atoms": normalized_atoms, "purity": float(purity), "support": 0.0})
            return

        feature_name = str(node.split_feature)
        left_levels = list(node.left_levels or [])
        right_levels = list(node.right_levels or [])
        if left_levels or right_levels:
            if left_levels:
                decoded_left = decode_levels(feature_name, left_levels)
                left_atom = categorical_atom(feature_name=feature_name, values=decoded_left, negate=False)
                right_atom = categorical_atom(feature_name=feature_name, values=decoded_left, negate=True)
            else:
                decoded_right = decode_levels(feature_name, right_levels)
                left_atom = categorical_atom(feature_name=feature_name, values=decoded_right, negate=True)
                right_atom = categorical_atom(feature_name=feature_name, values=decoded_right, negate=False)
        else:
            threshold_value = float(node.threshold)
            left_atom = _numeric_atom(feature_name=feature_name, threshold_value=threshold_value, is_true_branch=False)
            right_atom = _numeric_atom(feature_name=feature_name, threshold_value=threshold_value, is_true_branch=True)
        recurse(node.left_child, path_atoms + [left_atom])
        recurse(node.right_child, path_atoms + [right_atom])

    if len(label_domain) == 2:
        tree = H2OTree(model=model, tree_number=0)
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


def _extract_rules(
    model: Any | None,
    *,
    backend: str,
    encoded_specs: list[_EncodedFeatureSpec],
    label_name: str,
    positive_class: int,
    leaf_purity: float,
    num_rows: int,
    positive_count: int,
) -> list[dict[str, Any]]:
    if int(positive_count) <= 0:
        return []
    if int(positive_count) >= int(num_rows):
        return _constant_rules(positive_active=True, num_rows=int(num_rows))
    if model is None:
        return []
    if str(backend) == "sklearn":
        return _extract_rules_from_sklearn(
            model,
            encoded_specs=encoded_specs,
            positive_class=int(positive_class),
            leaf_purity=float(leaf_purity),
        )
    if str(backend) == "h2o":
        return _extract_rules_from_h2o(
            model,
            label_name=str(label_name),
            positive_class=int(positive_class),
            leaf_purity=float(leaf_purity),
        )
    raise RuntimeError(f"unsupported backend: {backend}")


def _format_atom(atom: dict[str, Any]) -> str:
    op = str(atom.get("op"))
    feature_name = str(atom.get("feature"))
    value = atom.get("value")
    if op in {"in", "not_in"}:
        rendered = ", ".join(str(item) for item in list(value))
        return f"{feature_name} {op} {{{rendered}}}"
    if op in {"eq", "ne"}:
        return f"{feature_name} {op} {value}"
    return f"{feature_name} {op} {float(value):.6g}"


def _rules_to_text(*, heading: str, rules: list[dict[str, Any]]) -> str:
    lines = [f"{heading} ({len(rules)}):"]
    for index, rule in enumerate(rules, start=1):
        atoms = list(rule.get("atoms", []))
        antecedent = " and ".join(_format_atom(atom) for atom in atoms) or "true"
        lines.append(f"  {index:02d}. if {antecedent}")
    return "\n".join(lines) + "\n"


def _binary_bundle(
    *,
    task_name: str,
    feature_df: pd.DataFrame,
    y: np.ndarray,
    split: _SplitSpec,
    backend: str,
    encoded_specs: list[_EncodedFeatureSpec],
    categorical_feature_names: set[str],
    train_cfg: _TreeTrainingConfig,
    leaf_purity: float,
) -> dict[str, Any]:
    train_x = feature_df.iloc[split.train_indices].reset_index(drop=True)
    train_y = np.asarray(y[split.train_indices], dtype=np.int32)
    val_x = feature_df.iloc[split.val_indices].reset_index(drop=True)
    val_y = np.asarray(y[split.val_indices], dtype=np.int32)

    model = None
    if train_y.size > 0 and np.unique(train_y).size > 1:
        model = _fit_tree_model(
            train_x,
            train_y,
            label_name=f"{task_name}_label",
            categorical_feature_names=categorical_feature_names,
            train_cfg=train_cfg,
        )

    if model is not None:
        train_pred = _predict_tree_model(model, train_x, backend=str(backend), categorical_feature_names=categorical_feature_names)
        val_pred = _predict_tree_model(model, val_x, backend=str(backend), categorical_feature_names=categorical_feature_names) if val_y.size > 0 else np.asarray([], dtype=np.int32)
    else:
        positive_active = bool(train_y.size > 0 and np.all(train_y == 1))
        train_pred = np.ones_like(train_y) if positive_active else np.zeros_like(train_y)
        val_pred = np.ones_like(val_y) if positive_active else np.zeros_like(val_y)

    rules = _extract_rules(
        model,
        backend=str(backend),
        encoded_specs=encoded_specs,
        label_name=f"{task_name}_label",
        positive_class=1,
        leaf_purity=float(leaf_purity),
        num_rows=int(train_y.size),
        positive_count=int(np.sum(train_y == 1)),
    )
    return {
        "task_name": str(task_name),
        "positive_fraction": float(np.mean(y == 1)) if y.size > 0 else 0.0,
        "train_metrics": _binary_metrics(train_y, train_pred),
        "val_metrics": _binary_metrics(val_y, val_pred),
        "num_rules": len(rules),
        "rules": rules,
    }


def _cluster_prototypes(
    feature_df: pd.DataFrame,
    *,
    split: _SplitSpec,
    prototype_clusters: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    components = _require_sklearn_component()
    scaler = components["StandardScaler"]()
    kmeans_cls = components["KMeans"]
    train_x = feature_df.iloc[split.train_indices].reset_index(drop=True)
    val_x = feature_df.iloc[split.val_indices].reset_index(drop=True)
    if train_x.empty:
        return np.asarray([], dtype=np.int32), {"num_clusters": 0}
    train_scaled = scaler.fit_transform(train_x)
    num_clusters = min(max(int(prototype_clusters), 1), train_scaled.shape[0])
    kmeans = kmeans_cls(n_clusters=int(num_clusters), random_state=int(seed), n_init=10)
    train_labels = np.asarray(kmeans.fit_predict(train_scaled), dtype=np.int32)
    full_labels = np.zeros(feature_df.shape[0], dtype=np.int32)
    full_labels[split.train_indices] = train_labels
    if not val_x.empty:
        val_scaled = scaler.transform(val_x)
        full_labels[split.val_indices] = np.asarray(kmeans.predict(val_scaled), dtype=np.int32)
    return full_labels, {"num_clusters": int(num_clusters), "cluster_centers": kmeans.cluster_centers_.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train trace explanation rules for difference, challenge, mechanism, and prototype tasks.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=["sklearn", "h2o"], default="h2o")
    parser.add_argument("--tasks", type=str, default="difference,challenge,mechanism,prototype")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--challenge-gap-pct", type=float, default=25.0)
    parser.add_argument("--baseline-score-floor", type=float, default=0.3)
    parser.add_argument("--mechanism-share-threshold", type=float, default=0.45)
    parser.add_argument("--mechanism-min-strength", type=float, default=0.02)
    parser.add_argument("--prototype-clusters", type=int, default=4)
    parser.add_argument("--prototype-min-train-support", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-leaf-nodes", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--leaf-purity", type=float, default=0.95)
    parser.add_argument("--h2o-ntrees", type=int, default=1)
    parser.add_argument("--h2o-max-depth", type=int, default=20)
    parser.add_argument("--h2o-min-rows", type=int, default=2)
    parser.add_argument("--h2o-sample-rate", type=float, default=1.0)
    parser.add_argument("--h2o-mtries", type=int, default=None)
    parser.add_argument("--max-env-error-steps", type=int, default=0)
    args = parser.parse_args()

    requested_tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
    allowed_tasks = {"difference", "challenge", "mechanism", "prototype"}
    unknown_tasks = [item for item in requested_tasks if item not in allowed_tasks]
    if unknown_tasks:
        raise RuntimeError(f"unknown tasks requested: {unknown_tasks}")

    repo_root = os.path.abspath(str(args.repo_root))
    dataset_path = resolve_repo_path(repo_root, str(args.dataset))
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if "env_error_steps" in df.columns:
        df = df[pd.to_numeric(df["env_error_steps"], errors="coerce").fillna(0.0) <= float(args.max_env_error_steps)].copy()
    if df.empty:
        raise RuntimeError("dataset is empty after filtering invalid rows")
    df = _coerce_feature_frame(df)

    sklearn_matrix, encoded_specs = _prepare_sklearn_matrix(df)
    h2o_matrix = _feature_frame_for_h2o(df)
    feature_matrix = sklearn_matrix if str(args.backend) == "sklearn" else h2o_matrix
    categorical_feature_names = set(CATEGORICAL_FEATURE_COLUMNS if str(args.backend) == "h2o" else ())
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

    outputs: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "dataset_path": dataset_path,
        "backend": str(args.backend),
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "tasks": {},
    }
    text_sections: list[str] = []

    if "difference" in requested_tasks:
        diff_y = np.asarray([difference_label(str(value)) for value in df["trace_type"].tolist()], dtype=np.int32)
        diff_split = _stratified_split(diff_y, validation_fraction=float(args.validation_fraction), seed=int(args.seed))
        diff_bundle = _binary_bundle(
            task_name="difference_adv_like",
            feature_df=feature_matrix,
            y=diff_y,
            split=diff_split,
            backend=str(args.backend),
            encoded_specs=encoded_specs,
            categorical_feature_names=categorical_feature_names,
            train_cfg=train_cfg,
            leaf_purity=float(args.leaf_purity),
        )
        outputs["tasks"]["difference"] = diff_bundle
        text_sections.append(_rules_to_text(heading="difference / adv_like", rules=list(diff_bundle["rules"])))

    adv_mask = np.asarray(df["trace_type"].astype(str).str.lower() == "adv", dtype=bool)
    if np.any(adv_mask) and any(task in requested_tasks for task in ("challenge", "mechanism", "prototype")):
        adv_df = df.loc[adv_mask].reset_index(drop=True)
        adv_feature_matrix = feature_matrix.loc[adv_mask].reset_index(drop=True)
        adv_sklearn_matrix = sklearn_matrix.loc[adv_mask].reset_index(drop=True)

        challenge_y = np.asarray(
            [
                challenge_label(
                    dict(row),
                    gap_pct_threshold=float(args.challenge_gap_pct),
                    baseline_score_floor=float(args.baseline_score_floor),
                )
                for row in adv_df.to_dict(orient="records")
            ],
            dtype=np.int32,
        )
        adv_split = _stratified_split(challenge_y, validation_fraction=float(args.validation_fraction), seed=int(args.seed))

        if "challenge" in requested_tasks:
            challenge_bundle = _binary_bundle(
                task_name="challenge_high_gap",
                feature_df=adv_feature_matrix,
                y=challenge_y,
                split=adv_split,
                backend=str(args.backend),
                encoded_specs=encoded_specs,
                categorical_feature_names=categorical_feature_names,
                train_cfg=train_cfg,
                leaf_purity=float(args.leaf_purity),
            )
            outputs["tasks"]["challenge"] = challenge_bundle
            text_sections.append(_rules_to_text(heading="challenge / high_gap", rules=list(challenge_bundle["rules"])))

        if "mechanism" in requested_tasks:
            mechanism_outputs: dict[str, Any] = {}
            adv_rows = adv_df.to_dict(orient="records")
            mechanism_labels = {
                label: np.asarray(
                    [
                        mechanism_label_map(
                            row,
                            challenge_gap_pct_threshold=float(args.challenge_gap_pct),
                            baseline_score_floor=float(args.baseline_score_floor),
                            share_threshold=float(args.mechanism_share_threshold),
                            min_strength=float(args.mechanism_min_strength),
                        )[label]
                        for row in adv_rows
                    ],
                    dtype=np.int32,
                )
                for label in ("throughput_harm", "rtt_harm", "loss_harm")
            }
            for label_name, label_values in mechanism_labels.items():
                mechanism_split = _stratified_split(
                    label_values,
                    validation_fraction=float(args.validation_fraction),
                    seed=int(args.seed),
                )
                bundle = _binary_bundle(
                    task_name=str(label_name),
                    feature_df=adv_feature_matrix,
                    y=label_values,
                    split=mechanism_split,
                    backend=str(args.backend),
                    encoded_specs=encoded_specs,
                    categorical_feature_names=categorical_feature_names,
                    train_cfg=train_cfg,
                    leaf_purity=float(args.leaf_purity),
                )
                mechanism_outputs[str(label_name)] = bundle
                text_sections.append(_rules_to_text(heading=f"mechanism / {label_name}", rules=list(bundle["rules"])))
            outputs["tasks"]["mechanism"] = mechanism_outputs
            outputs["mechanism_share_examples"] = {
                str(label): float(np.mean([mechanism_shares(row).get(label, 0.0) for row in adv_rows])) for label in ("throughput_harm", "rtt_harm", "loss_harm")
            }

        if "prototype" in requested_tasks:
            prototype_split = _random_split(
                int(adv_df.shape[0]),
                validation_fraction=float(args.validation_fraction),
                seed=int(args.seed),
            )
            prototype_labels, cluster_info = _cluster_prototypes(
                adv_sklearn_matrix,
                split=prototype_split,
                prototype_clusters=int(args.prototype_clusters),
                seed=int(args.seed),
            )
            prototype_outputs: dict[str, Any] = {"cluster_info": cluster_info, "clusters": {}}
            for cluster_id in sorted(set(int(item) for item in prototype_labels.tolist())):
                train_support = int(np.sum(prototype_labels[prototype_split.train_indices] == int(cluster_id)))
                if train_support < int(args.prototype_min_train_support):
                    continue
                cluster_y = np.asarray(prototype_labels == int(cluster_id), dtype=np.int32)
                bundle = _binary_bundle(
                    task_name=f"prototype_{cluster_id:02d}",
                    feature_df=adv_feature_matrix,
                    y=cluster_y,
                    split=prototype_split,
                    backend=str(args.backend),
                    encoded_specs=encoded_specs,
                    categorical_feature_names=categorical_feature_names,
                    train_cfg=train_cfg,
                    leaf_purity=float(args.leaf_purity),
                )
                cluster_rows = adv_df.loc[cluster_y == 1].copy()
                prototype_outputs["clusters"][f"prototype_{cluster_id:02d}"] = {
                    **bundle,
                    "train_support": int(train_support),
                    "total_support": int(np.sum(cluster_y == 1)),
                    "mean_hard_gap_percent": float(pd.to_numeric(cluster_rows.get("hard_gap_percent_mean", 0.0), errors="coerce").fillna(0.0).mean()),
                    "mean_hard_gap_value": float(pd.to_numeric(cluster_rows.get("hard_gap_value_mean", 0.0), errors="coerce").fillna(0.0).mean()),
                }
                text_sections.append(_rules_to_text(heading=f"prototype / prototype_{cluster_id:02d}", rules=list(bundle["rules"])))
            outputs["tasks"]["prototype"] = prototype_outputs

    backend_tag = str(args.backend).strip().lower()
    json_path = os.path.join(out_dir, f"trace_explanation_rules_{backend_tag}.json")
    txt_path = os.path.join(out_dir, f"trace_explanation_rules_{backend_tag}.txt")
    save_json(json_path, outputs)
    with open(txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(text_sections).strip() + "\n")
    print(json_path)


if __name__ == "__main__":
    main()
