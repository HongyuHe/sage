from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

from attacks.online.shm import DEFAULT_OBS_COLS
from sage_rl.shield.features import ShieldFeatureTracker


@dataclass(frozen=True)
class RuleAtom:
    feature: str
    op: str
    value: float


class RuleSet:
    def __init__(self, rules: Sequence[Sequence[RuleAtom]] | None = None) -> None:
        self.rules = [list(rule) for rule in (rules or [])]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "RuleSet":
        if payload is None:
            return cls([])
        parsed_rules: list[list[RuleAtom]] = []
        for rule in payload.get("rules", []):
            atoms = [
                RuleAtom(
                    feature=str(atom["feature"]),
                    op=str(atom["op"]),
                    value=float(atom["value"]),
                )
                for atom in rule.get("atoms", [])
            ]
            parsed_rules.append(atoms)
        return cls(parsed_rules)

    def match_count(self, values: Mapping[str, float]) -> int:
        matches = 0
        for atoms in self.rules:
            if all(_rule_atom_satisfied(values=values, atom=atom) for atom in atoms):
                matches += 1
        return int(matches)


@dataclass(frozen=True)
class RuleBundle:
    feature_names: tuple[str, ...]
    history_len: int
    risk: RuleSet
    backoff: RuleSet
    push: RuleSet
    metadata: dict[str, Any]


def _rule_atom_satisfied(*, values: Mapping[str, float], atom: RuleAtom) -> bool:
    value = float(values.get(str(atom.feature), 0.0))
    threshold = float(atom.value)
    if str(atom.op) == "gt":
        return value > threshold
    if str(atom.op) == "ge":
        return value >= threshold
    if str(atom.op) == "lt":
        return value < threshold
    if str(atom.op) == "le":
        return value <= threshold
    if str(atom.op) == "eq":
        return abs(value - threshold) <= 1e-9
    raise ValueError(f"unknown shield rule op: {atom.op}")


def load_rule_bundle(path: str) -> RuleBundle:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = dict(json.load(file_obj))
    return RuleBundle(
        feature_names=tuple(str(item) for item in payload.get("feature_names", [])),
        history_len=max(int(payload.get("history_len", 4)), 1),
        risk=RuleSet.from_payload(payload.get("risk")),
        backoff=RuleSet.from_payload(payload.get("backoff")),
        push=RuleSet.from_payload(payload.get("push")),
        metadata=dict(payload.get("metadata", {})),
    )


class DirectionalShield:
    def __init__(
        self,
        *,
        rule_bundle: RuleBundle,
        action_low: float,
        action_high: float,
        action_delta: float = 0.15,
        consecutive_risk: int = 2,
        cooldown_steps: int = 2,
        obs_cols: Sequence[int] = DEFAULT_OBS_COLS,
        log_path: str | None = None,
    ) -> None:
        self._rule_bundle = rule_bundle
        self._tracker = ShieldFeatureTracker(history_len=int(rule_bundle.history_len))
        self._action_low = float(action_low)
        self._action_high = float(action_high)
        self._action_delta = max(float(action_delta), 0.0)
        self._consecutive_risk = max(int(consecutive_risk), 1)
        self._cooldown_steps = max(int(cooldown_steps), 0)
        self._obs_cols = tuple(int(item) for item in obs_cols)
        self._log_path = str(log_path) if log_path else None
        self.reset()

    def reset(self) -> None:
        self._tracker.reset()
        self._risk_streak = 0
        self._cooldown_remaining = 0
        self._decision_index = 0
        if self._log_path:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            with open(self._log_path, "w", encoding="utf-8") as file_obj:
                file_obj.write("")

    def _append_log(self, payload: Mapping[str, Any]) -> None:
        if not self._log_path:
            return
        with open(self._log_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def adjust_action(
        self,
        *,
        observation: np.ndarray | Sequence[float],
        proposed_action: np.ndarray | Sequence[float],
    ) -> tuple[np.ndarray, dict[str, float]]:
        features = self._tracker.update_from_observation(observation, obs_cols=self._obs_cols)
        risk_matches = int(self._rule_bundle.risk.match_count(features))
        risky = risk_matches > 0
        if risky:
            self._risk_streak += 1
        else:
            self._risk_streak = 0
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        backoff_matches = int(self._rule_bundle.backoff.match_count(features))
        push_matches = int(self._rule_bundle.push.match_count(features))
        direction = "hold"
        if risky and self._risk_streak >= self._consecutive_risk and self._cooldown_remaining <= 0:
            if backoff_matches > 0 and push_matches == 0:
                direction = "back_off"
            elif push_matches > 0 and backoff_matches == 0:
                direction = "push_harder"

        action_array = np.asarray(proposed_action, dtype=np.float32).reshape(-1).astype(np.float32, copy=True)
        before = float(action_array[0]) if action_array.size > 0 else 0.0
        after = before
        if action_array.size > 0:
            if direction == "back_off":
                after = float(np.clip(before - self._action_delta, self._action_low, self._action_high))
                self._cooldown_remaining = self._cooldown_steps
            elif direction == "push_harder":
                after = float(np.clip(before + self._action_delta, self._action_low, self._action_high))
                self._cooldown_remaining = self._cooldown_steps
            action_array[0] = float(after)

        stats = {
            "shield/applied": 0.0 if direction == "hold" else 1.0,
            "shield/risky": 1.0 if risky else 0.0,
            "shield/risk_matches": float(risk_matches),
            "shield/backoff_matches": float(backoff_matches),
            "shield/push_matches": float(push_matches),
            "shield/risk_streak": float(self._risk_streak),
            "shield/cooldown_remaining": float(self._cooldown_remaining),
            "shield/action_before": float(before),
            "shield/action_after": float(after),
            "shield/direction_backoff": 1.0 if direction == "back_off" else 0.0,
            "shield/direction_push": 1.0 if direction == "push_harder" else 0.0,
            "shield/direction_hold": 1.0 if direction == "hold" else 0.0,
        }
        self._append_log(
            {
                "decision_index": int(self._decision_index),
                "direction": str(direction),
                "features": {key: float(value) for key, value in features.items()},
                **{key: float(value) for key, value in stats.items()},
            }
        )
        self._decision_index += 1
        return action_array.astype(np.float32, copy=False), stats


def maybe_build_shield_from_env(*, tcpspec: Mapping[str, Any]) -> DirectionalShield | None:
    rules_file = os.environ.get("SAGE_SHIELD_RULES_FILE", "").strip()
    if not rules_file:
        return None
    bundle = load_rule_bundle(rules_file)
    action_version = int(tcpspec.get("action_version", 9))
    action_max = float(tcpspec.get("action_max", 2.0))
    action_low = -action_max if action_version == 9 else 0.0
    action_high = float(action_max)
    return DirectionalShield(
        rule_bundle=bundle,
        action_low=action_low,
        action_high=action_high,
        action_delta=float(os.environ.get("SAGE_SHIELD_ACTION_DELTA", "0.15")),
        consecutive_risk=int(os.environ.get("SAGE_SHIELD_CONSECUTIVE_RISK", "2")),
        cooldown_steps=int(os.environ.get("SAGE_SHIELD_COOLDOWN_STEPS", "2")),
        obs_cols=tuple(int(item) for item in tcpspec.get("obs_cols", DEFAULT_OBS_COLS)),
        log_path=os.environ.get("SAGE_SHIELD_LOG_PATH", "").strip() or None,
    )
