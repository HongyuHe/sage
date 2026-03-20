"""
Example usage:
Default gap training now uses all available baselines: reno,bbr,cubic.
python scripts/train_online_attacker.py \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --ppo-n-steps 256 \
  --ppo-batch-size 64 \
  --ppo-n-epochs 4 \
  --out-dir attacks/output/models \
  --wandb --wandb-project sage-online-train --wandb-name v1-fast

time python scripts/train_online_attacker.py \
  --attack-mode independent_gap \
  --baseline-methods reno,bbr,cubic \
  --smooth-penalty-scale 0.00 \
  --attack-shared-bw-min-mbps 0.5 --attack-shared-bw-max-mbps 2000 \
  --effective-bw-cap-mbps 2000 \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --ppo-ent-coef 0.005 \
  --wandb --wandb-tags 300k --wandb-project sage-gap-train-v3 --wandb-name gap-unconstrained
  
time python scripts/train_online_attacker.py \
  --attack-mode independent_gap \
  --baseline-methods reno,bbr,cubic \
  --smooth-penalty-scale 0.05 \
  --attack-shared-bw-min-mbps 5 --attack-shared-bw-max-mbps 150 \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --ppo-ent-coef 0.01 \
  --wandb --wandb-tags 300k --wandb-project sage-gap-train-v3 --wandb-name gap-constrained-3b

time python scripts/train_online_attacker.py \
  --attack-mode independent_gap \
  --baseline-methods reno,bbr,cubic \
  --smooth-penalty-scale 0.05 \
  --attack-shared-bw-min-mbps 5 --attack-shared-bw-max-mbps 150 \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --ppo-ent-coef 0.01 \
  --baseline-hard-max \
  --wandb --wandb-tags 300k --wandb-project sage-gap-train-v3 --wandb-name gap-constrained-3b-hard
  
time python scripts/train_online_attacker.py \
  --attack-mode independent_gap \
  --baseline-methods bbr \
  --smooth-penalty-scale 0.05 \
  --attack-shared-bw-min-mbps 5 --attack-shared-bw-max-mbps 150 \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --ppo-ent-coef 0.01 \
  --wandb --wandb-tags 300k --wandb-project sage-gap-train-v3 --wandb-name gap-constrained-bbr-only

time python scripts/train_online_attacker.py \
  --attack-mode independent \
  --attack-interval-ms 100 \
  --attack-shared-bw-min-mbps 6 --attack-shared-bw-max-mbps 24 \
  --attack-shared-loss-min 0.0 --attack-shared-loss-max 0.0 \
  --attack-shared-delay-min-ms 25 --attack-shared-delay-max-ms 25 \
  --total-steps 300000 \
  --out-dir attacks/output/models \
  --wandb --wandb-tags 300k --wandb-project sage-gap-train-v3 --wandb-name hotnets19-100ms

Legacy two-baseline reproduction:
time python scripts/train_online_attacker.py \
  --attack-mode independent_gap \
  --baseline-methods cubic,bbr \
  --smooth-penalty-scale 0.05 \
  --attack-shared-bw-min-mbps 5 --attack-shared-bw-max-mbps 150 \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --ppo-ent-coef 0.01 \
  --wandb --wandb-tags 300k --wandb-project sage-gap-train-v3 --wandb-name gap-constrained-legacy
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys
import time

import numpy as np

from attacks.envs import (
    AVAILABLE_BASELINE_METHODS,
    AttackBounds,
    ParallelGapAttackEnv,
    normalize_baseline_methods,
)
from attacks.online import SageLaunchConfig, acquire_run_namespace


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        IndependentAttackEnv,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        try_import_wandb,
    )
else:
    from ._trace_attack_common import (
        IndependentAttackEnv,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        try_import_wandb,
    )


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
        from stable_baselines3.common.monitor import Monitor  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore

        return PPO, BaseCallback, Monitor, DummyVecEnv, VecNormalize
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for online attacker training") from exc


def _numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "avg": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _aggregate_selected_metrics(
    records: list[dict[str, float]],
    metric_map: dict[str, str],
    *,
    prefix: str,
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for source_key, target_key in metric_map.items():
        values = [
            float(record[source_key])
            for record in records
            if isinstance(record.get(source_key), (int, float, np.floating, np.integer))
        ]
        stats = _numeric_summary(values)
        for stat_name, stat_value in stats.items():
            payload[f"{prefix}/{target_key}-{stat_name}"] = float(stat_value)
    return payload


_WANDB_AGGREGATE_INFO_KEYS: dict[str, str] = {
    "attacker/reward": "attacker_reward",
    "attacker/shared_bw_mbps": "attacker_shared_bw_mbps",
    "attacker/shared_bw_fraction": "attacker_shared_bw_fraction",
    "sage/reward": "sage_reward",
    "sage/score": "sage_score",
    "sage/score_rate_norm": "sage_score_rate_norm",
    "sage/score_rtt_norm": "sage_score_rtt_norm",
    "sage/score_loss_norm": "sage_score_loss_norm",
    "sage/score_rate_contrib": "sage_score_rate_contrib",
    "sage/score_rtt_contrib": "sage_score_rtt_contrib",
    "sage/score_loss_penalty": "sage_score_loss_penalty",
    "sage/current_delivery_rate_mbps": "sage_current_delivery_rate_mbps",
    "sage/windowed_delivery_rate_mbps": "sage_windowed_rate_mbps",
    "sage/current_rtt_ms": "sage_rtt_ms",
    "sage/current_loss_mbps": "sage_loss_mbps",
    "gap/score_sage": "gap_score_sage",
    "gap/score_cubic": "gap_score_cubic",
    "gap/score_bbr": "gap_score_bbr",
    "gap/score_sage_rate_norm": "gap_score_sage_rate_norm",
    "gap/score_sage_rtt_norm": "gap_score_sage_rtt_norm",
    "gap/score_sage_loss_norm": "gap_score_sage_loss_norm",
    "gap/score_sage_rate_contrib": "gap_score_sage_rate_contrib",
    "gap/score_sage_rtt_contrib": "gap_score_sage_rtt_contrib",
    "gap/score_sage_loss_penalty": "gap_score_sage_loss_penalty",
    "gap/score_cubic_rate_norm": "gap_score_cubic_rate_norm",
    "gap/score_cubic_rtt_norm": "gap_score_cubic_rtt_norm",
    "gap/score_cubic_loss_norm": "gap_score_cubic_loss_norm",
    "gap/score_cubic_rate_contrib": "gap_score_cubic_rate_contrib",
    "gap/score_cubic_rtt_contrib": "gap_score_cubic_rtt_contrib",
    "gap/score_cubic_loss_penalty": "gap_score_cubic_loss_penalty",
    "gap/score_bbr_rate_norm": "gap_score_bbr_rate_norm",
    "gap/score_bbr_rtt_norm": "gap_score_bbr_rtt_norm",
    "gap/score_bbr_loss_norm": "gap_score_bbr_loss_norm",
    "gap/score_bbr_rate_contrib": "gap_score_bbr_rate_contrib",
    "gap/score_bbr_rtt_contrib": "gap_score_bbr_rtt_contrib",
    "gap/score_bbr_loss_penalty": "gap_score_bbr_loss_penalty",
    "gap/score_reno": "gap_score_reno",
    "gap/score_reno_rate_norm": "gap_score_reno_rate_norm",
    "gap/score_reno_rtt_norm": "gap_score_reno_rtt_norm",
    "gap/score_reno_loss_norm": "gap_score_reno_loss_norm",
    "gap/score_reno_rate_contrib": "gap_score_reno_rate_contrib",
    "gap/score_reno_rtt_contrib": "gap_score_reno_rtt_contrib",
    "gap/score_reno_loss_penalty": "gap_score_reno_loss_penalty",
    "gap/baseline_score": "gap_baseline_score",
    "gap/best_baseline_score": "gap_best_baseline_score",
    "gap/best_baseline_gap": "gap_best_baseline_gap",
    "gap/best_baseline_wins": "gap_best_baseline_gap_positive_fraction",
    "gap/baseline_weight_cubic": "gap_baseline_weight_cubic",
    "gap/baseline_weight_bbr": "gap_baseline_weight_bbr",
    "gap/baseline_weight_reno": "gap_baseline_weight_reno",
    "gap/value": "gap_value",
    "gap/reward": "gap_reward",
    "baseline/reno_rtt_ms": "baseline_reno_rtt_ms",
    "baseline/reno_rate_mbps": "baseline_reno_rate_mbps",
    "baseline/cubic_rtt_ms": "baseline_cubic_rtt_ms",
    "baseline/cubic_rate_mbps": "baseline_cubic_rate_mbps",
    "baseline/bbr_rtt_ms": "baseline_bbr_rtt_ms",
    "baseline/bbr_rate_mbps": "baseline_bbr_rate_mbps",
    "attacker/uplink_bw_mbps": "attacker_uplink_bw_mbps",
    "attacker/downlink_bw_mbps": "attacker_downlink_bw_mbps",
    "mm/up_applied_bw_mbps": "mm_up_applied_bw_mbps",
    "mm/down_applied_bw_mbps": "mm_down_applied_bw_mbps",
    "mm/up_departure_rate_mbps": "mm_up_departure_rate_mbps",
    "mm/down_departure_rate_mbps": "mm_down_departure_rate_mbps",
}

_WANDB_STEP_INFO_KEYS: dict[str, str] = {
    "attacker/shared_bw_mbps": "attacker_shared_bw_mbps",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument(
        "--attack-mode",
        type=str,
        default="independent_gap",
        choices=["independent", "independent_gap"],
    )
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="attacks/output/models")
    parser.add_argument("--checkpoint-dir", type=str, default="attacks/output/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--obs-history-len", type=int, default=4)
    parser.add_argument("--attack-interval-ms", type=float, default=100.0)
    parser.add_argument("--episode-steps", type=int, default=6000)
    parser.add_argument("--launch-timeout-s", type=float, default=90.0)
    parser.add_argument("--step-timeout-s", type=float, default=10.0)
    parser.add_argument("--smooth-penalty-scale", type=float, default=None)
    parser.add_argument("--baseline-gap-alpha", type=float, default=2.0)
    parser.add_argument(
        "--baseline-hard-max",
        action="store_true",
        help="Use the hard max over enabled baseline scores instead of the softmax-smoothed baseline score.",
    )
    parser.add_argument(
        "--baseline-methods",
        type=str,
        default=",".join(AVAILABLE_BASELINE_METHODS),
        help="Comma-separated gap baselines to launch. Default enables all available baselines: reno,bbr,cubic.",
    )
    parser.add_argument("--sync-guard-ms", type=float, default=25.0)
    parser.add_argument("--gap-launch-retries", type=int, default=6)
    parser.add_argument("--bw-scale-min", type=float, default=0.1)
    parser.add_argument("--bw-scale-max", type=float, default=2.0)
    parser.add_argument("--loss-max", type=float, default=0.15)
    parser.add_argument("--delay-max-ms", type=float, default=150.0)
    parser.add_argument("--effective-bw-cap-mbps", type=float, default=2000.0)
    parser.add_argument("--attack-shared-bw-min-mbps", type=float, default=None)
    parser.add_argument("--attack-shared-bw-max-mbps", type=float, default=None)
    parser.add_argument("--attack-uplink-bw-min-mbps", type=float, default=None)
    parser.add_argument("--attack-uplink-bw-max-mbps", type=float, default=None)
    parser.add_argument("--attack-downlink-bw-min-mbps", type=float, default=None)
    parser.add_argument("--attack-downlink-bw-max-mbps", type=float, default=None)
    parser.add_argument("--attack-uplink-loss-min", type=float, default=None)
    parser.add_argument("--attack-uplink-loss-max", type=float, default=None)
    parser.add_argument("--attack-downlink-loss-min", type=float, default=None)
    parser.add_argument("--attack-downlink-loss-max", type=float, default=None)
    parser.add_argument("--attack-shared-loss-min", type=float, default=None)
    parser.add_argument("--attack-shared-loss-max", type=float, default=None)
    parser.add_argument("--attack-uplink-delay-min-ms", type=float, default=None)
    parser.add_argument("--attack-uplink-delay-max-ms", type=float, default=None)
    parser.add_argument("--attack-downlink-delay-min-ms", type=float, default=None)
    parser.add_argument("--attack-downlink-delay-max-ms", type=float, default=None)
    parser.add_argument("--attack-shared-delay-min-ms", type=float, default=None)
    parser.add_argument("--attack-shared-delay-max-ms", type=float, default=None)
    parser.add_argument("--policy-width", type=int, default=128)
    parser.add_argument("--policy-depth", type=int, default=2)
    parser.add_argument("--ppo-n-steps", type=int, default=1024)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.0)

    parser.add_argument("--latency-ms", type=int, default=25)
    parser.add_argument("--port", type=int, default=5101)
    parser.add_argument("--iteration-id", type=int, default=0)
    parser.add_argument("--qsize-packets", type=int, default=128)
    parser.add_argument("--env-bw-mbps", type=int, default=48)
    parser.add_argument("--bw2-mbps", type=int, default=48)
    parser.add_argument("--trace-period-s", type=int, default=7)
    parser.add_argument("--sage-mode", type=int, default=0)
    parser.add_argument("--log-prefix", type=str, default="adv-train")
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--actor-id", type=int, default=900)
    parser.add_argument("--duration-steps", type=int, default=6000)
    parser.add_argument("--num-flows", type=int, default=1)
    parser.add_argument("--save-logs", type=int, default=0)
    parser.add_argument("--analyze-logs", type=int, default=0)
    parser.add_argument("--mm-adv-bin", type=str, default=None)
    parser.add_argument("--init-uplink-loss", type=float, default=0.0)
    parser.add_argument("--init-downlink-loss", type=float, default=0.0)
    parser.add_argument("--init-uplink-delay-ms", type=float, default=None)
    parser.add_argument("--init-downlink-delay-ms", type=float, default=None)
    parser.add_argument("--init-uplink-queue-packets", type=int, default=None)
    parser.add_argument("--init-downlink-queue-packets", type=int, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sage-online-train")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-log-every", type=int, default=100)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    PPO, BaseCallback, Monitor, DummyVecEnv, VecNormalize = _require_sb3()
    use_gap_objective = str(args.attack_mode) == "independent_gap"
    baseline_methods = normalize_baseline_methods(args.baseline_methods)
    shared_bandwidth_action_requested = (
        args.attack_shared_bw_min_mbps is not None and args.attack_shared_bw_max_mbps is not None
    )
    resolved_smooth_penalty_scale = (
        float(args.smooth_penalty_scale)
        if args.smooth_penalty_scale is not None
        else (0.0 if use_gap_objective else 0.01)
    )
    gap_ports_per_run = len(baseline_methods) + 1
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(args.actor_id),
        port=int(args.port),
        label=str(args.wandb_name or args.log_prefix or "train-online-attacker"),
        ports_per_run=gap_ports_per_run if use_gap_objective else 1,
    )
    resolved_runtime_dir = run_namespace.runtime_dir
    resolved_launch_config = {
        "run_id": run_namespace.run_id,
        "runtime_dir_resolved": resolved_runtime_dir,
        "launch_port_base_resolved": int(run_namespace.port_base),
        "launch_actor_id_base_resolved": int(run_namespace.actor_id_base),
        "runtime_slot": int(run_namespace.slot),
        "ports_per_run": gap_ports_per_run if use_gap_objective else 1,
    }

    wandb = None
    wandb_run = None
    env = None
    venv = None
    model = None
    model_path = ""
    vecnormalize_path = None
    checkpoint_dir = resolve_repo_path(repo_root, str(args.checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        model_prefix = "gap_adv" if use_gap_objective else "online_adv"
        model_stem = f"{model_prefix}_{stamp}_{run_namespace.run_id}"
        checkpoint_config_path = os.path.join(checkpoint_dir, f"{model_stem}.config.json")
    except Exception:
        model_stem = ""
        checkpoint_config_path = ""
    try:
        if bool(args.wandb):
            wandb = try_import_wandb()
            if wandb is None:
                raise RuntimeError("--wandb was set but the wandb package is unavailable")
            wandb_run = wandb.init(
                project=str(args.wandb_project),
                entity=args.wandb_entity,
                name=args.wandb_name,
                mode=str(args.wandb_mode),
                tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
                config={
                    **vars(args),
                    **resolved_launch_config,
                    "baseline_methods_resolved": list(baseline_methods),
                    "smooth_penalty_scale": float(resolved_smooth_penalty_scale),
                    "policy_action_transform": (
                        "log_unit_interval_shared_bandwidth"
                        if (use_gap_objective or shared_bandwidth_action_requested)
                        else "identity"
                    ),
                    "vecnormalize_enabled": bool(use_gap_objective or shared_bandwidth_action_requested),
                },
            )

        launch_config = replace(
            SageLaunchConfig(
                sage_script="sage_rl/sage.sh",
                latency_ms=int(args.latency_ms),
                port=int(args.port),
                downlink_trace="wired48",
                uplink_trace="wired48",
                iteration_id=int(args.iteration_id),
                qsize_packets=int(args.qsize_packets),
                env_bw_mbps=int(args.env_bw_mbps),
                bw2_mbps=int(args.bw2_mbps),
                trace_period_s=int(args.trace_period_s),
                first_time_mode=int(args.sage_mode),
                log_prefix=str(args.log_prefix),
                duration_seconds=int(args.duration_seconds),
                actor_id=int(args.actor_id),
                duration_steps=int(args.duration_steps),
                num_flows=int(args.num_flows),
                save_logs=int(args.save_logs),
                analyze_logs=int(args.analyze_logs),
                mm_adv_bin=args.mm_adv_bin,
                initial_uplink_loss=float(args.init_uplink_loss),
                initial_downlink_loss=float(args.init_downlink_loss),
                initial_uplink_delay_ms=args.init_uplink_delay_ms,
                initial_downlink_delay_ms=args.init_downlink_delay_ms,
                initial_uplink_queue_packets=args.init_uplink_queue_packets,
                initial_downlink_queue_packets=args.init_downlink_queue_packets,
            ),
            port=int(run_namespace.port_base),
            actor_id=int(run_namespace.actor_id_base),
        )

        #* Expose OnlineSageAttackEnv action bounds from this training entrypoint.
        attack_uplink_bw_min = (
            float(args.attack_uplink_bw_min_mbps) if args.attack_uplink_bw_min_mbps is not None else 0.0
        )
        attack_uplink_bw_max = (
            float(args.attack_uplink_bw_max_mbps)
            if args.attack_uplink_bw_max_mbps is not None
            else float(args.effective_bw_cap_mbps)
        )
        attack_downlink_bw_min = (
            float(args.attack_downlink_bw_min_mbps) if args.attack_downlink_bw_min_mbps is not None else 0.0
        )
        attack_downlink_bw_max = (
            float(args.attack_downlink_bw_max_mbps)
            if args.attack_downlink_bw_max_mbps is not None
            else float(args.effective_bw_cap_mbps)
        )
        default_loss_max = 0.0 if use_gap_objective else float(args.loss_max)
        attack_uplink_loss_min = (
            float(args.attack_uplink_loss_min) if args.attack_uplink_loss_min is not None else 0.0
        )
        attack_uplink_loss_max = (
            float(args.attack_uplink_loss_max) if args.attack_uplink_loss_max is not None else default_loss_max
        )
        attack_downlink_loss_min = (
            float(args.attack_downlink_loss_min) if args.attack_downlink_loss_min is not None else 0.0
        )
        attack_downlink_loss_max = (
            float(args.attack_downlink_loss_max) if args.attack_downlink_loss_max is not None else default_loss_max
        )
        attack_uplink_delay_min = (
            float(args.attack_uplink_delay_min_ms) if args.attack_uplink_delay_min_ms is not None else 0.0
        )
        attack_uplink_delay_max = (
            float(args.attack_uplink_delay_max_ms)
            if args.attack_uplink_delay_max_ms is not None
            else float(args.delay_max_ms)
        )
        attack_downlink_delay_min = (
            float(args.attack_downlink_delay_min_ms) if args.attack_downlink_delay_min_ms is not None else 0.0
        )
        attack_downlink_delay_max = (
            float(args.attack_downlink_delay_max_ms)
            if args.attack_downlink_delay_max_ms is not None
            else float(args.delay_max_ms)
        )
        if (args.attack_shared_bw_min_mbps is None) != (args.attack_shared_bw_max_mbps is None):
            raise ValueError(
                "--attack-shared-bw-min-mbps and --attack-shared-bw-max-mbps must be set together"
            )
        if args.attack_shared_bw_min_mbps is not None and args.attack_shared_bw_max_mbps is not None:
            attack_uplink_bw_min = float(args.attack_shared_bw_min_mbps)
            attack_uplink_bw_max = float(args.attack_shared_bw_max_mbps)
            attack_downlink_bw_min = float(args.attack_shared_bw_min_mbps)
            attack_downlink_bw_max = float(args.attack_shared_bw_max_mbps)
        if (args.attack_shared_loss_min is None) != (args.attack_shared_loss_max is None):
            raise ValueError("--attack-shared-loss-min and --attack-shared-loss-max must be set together")
        if args.attack_shared_loss_min is not None and args.attack_shared_loss_max is not None:
            attack_uplink_loss_min = float(args.attack_shared_loss_min)
            attack_uplink_loss_max = float(args.attack_shared_loss_max)
            attack_downlink_loss_min = float(args.attack_shared_loss_min)
            attack_downlink_loss_max = float(args.attack_shared_loss_max)
        if (args.attack_shared_delay_min_ms is None) != (args.attack_shared_delay_max_ms is None):
            raise ValueError(
                "--attack-shared-delay-min-ms and --attack-shared-delay-max-ms must be set together"
            )
        if args.attack_shared_delay_min_ms is not None and args.attack_shared_delay_max_ms is not None:
            attack_uplink_delay_min = float(args.attack_shared_delay_min_ms)
            attack_uplink_delay_max = float(args.attack_shared_delay_max_ms)
            attack_downlink_delay_min = float(args.attack_shared_delay_min_ms)
            attack_downlink_delay_max = float(args.attack_shared_delay_max_ms)
        if attack_uplink_bw_min > attack_uplink_bw_max:
            raise ValueError("--attack-uplink-bw-min-mbps must be <= --attack-uplink-bw-max-mbps")
        if attack_downlink_bw_min > attack_downlink_bw_max:
            raise ValueError("--attack-downlink-bw-min-mbps must be <= --attack-downlink-bw-max-mbps")
        if attack_uplink_loss_min > attack_uplink_loss_max:
            raise ValueError("--attack-uplink-loss-min must be <= --attack-uplink-loss-max")
        if attack_downlink_loss_min > attack_downlink_loss_max:
            raise ValueError("--attack-downlink-loss-min must be <= --attack-downlink-loss-max")
        if attack_uplink_delay_min > attack_uplink_delay_max:
            raise ValueError("--attack-uplink-delay-min-ms must be <= --attack-uplink-delay-max-ms")
        if attack_downlink_delay_min > attack_downlink_delay_max:
            raise ValueError("--attack-downlink-delay-min-ms must be <= --attack-downlink-delay-max-ms")
        if use_gap_objective:
            if float(args.init_uplink_loss) != 0.0 or float(args.init_downlink_loss) != 0.0:
                raise ValueError("--init-uplink-loss and --init-downlink-loss must be 0 with --attack-mode independent_gap")
            if any(
                value is not None and float(value) != 0.0
                for value in (
                    args.attack_uplink_loss_min,
                    args.attack_uplink_loss_max,
                    args.attack_downlink_loss_min,
                    args.attack_downlink_loss_max,
                    args.attack_shared_loss_min,
                    args.attack_shared_loss_max,
                )
            ):
                raise ValueError("loss is fixed to 0 with --attack-mode independent_gap")
            if any(
                value is not None
                for value in (
                    args.attack_uplink_delay_min_ms,
                    args.attack_uplink_delay_max_ms,
                    args.attack_downlink_delay_min_ms,
                    args.attack_downlink_delay_max_ms,
                    args.attack_shared_delay_min_ms,
                    args.attack_shared_delay_max_ms,
                )
            ):
                raise ValueError(
                    "delay is fixed in --attack-mode independent_gap; use --init-uplink-delay-ms/--init-downlink-delay-ms instead"
                )
            shared_bw_min = max(attack_uplink_bw_min, attack_downlink_bw_min)
            shared_bw_max = min(attack_uplink_bw_max, attack_downlink_bw_max)
            if shared_bw_min > shared_bw_max:
                raise ValueError("gap-mode uplink/downlink bandwidth bounds must overlap")
            fixed_uplink_delay = (
                float(args.init_uplink_delay_ms) if args.init_uplink_delay_ms is not None else float(args.latency_ms)
            )
            fixed_downlink_delay = (
                float(args.init_downlink_delay_ms) if args.init_downlink_delay_ms is not None else float(args.latency_ms)
            )
            online_attack_bounds = AttackBounds(
                uplink_bw_mbps=(shared_bw_min, shared_bw_max),
                downlink_bw_mbps=(shared_bw_min, shared_bw_max),
                uplink_loss=(0.0, 0.0),
                downlink_loss=(0.0, 0.0),
                uplink_delay_ms=(fixed_uplink_delay, fixed_uplink_delay),
                downlink_delay_ms=(fixed_downlink_delay, fixed_downlink_delay),
            )
        else:
            online_attack_bounds = AttackBounds(
                uplink_bw_mbps=(attack_uplink_bw_min, attack_uplink_bw_max),
                downlink_bw_mbps=(attack_downlink_bw_min, attack_downlink_bw_max),
                uplink_loss=(attack_uplink_loss_min, attack_uplink_loss_max),
                downlink_loss=(attack_downlink_loss_min, attack_downlink_loss_max),
                uplink_delay_ms=(attack_uplink_delay_min, attack_uplink_delay_max),
                downlink_delay_ms=(attack_downlink_delay_min, attack_downlink_delay_max),
            )
        shared_bandwidth_action = (
            args.attack_shared_bw_min_mbps is not None and args.attack_shared_bw_max_mbps is not None
        )
        shared_loss_action = args.attack_shared_loss_min is not None and args.attack_shared_loss_max is not None
        shared_delay_action = (
            args.attack_shared_delay_min_ms is not None and args.attack_shared_delay_max_ms is not None
        )
        use_vecnormalize = bool(use_gap_objective or shared_bandwidth_action)

        if use_gap_objective:
            env = ParallelGapAttackEnv(
                repo_root=repo_root,
                launch_config=launch_config,
                bounds=online_attack_bounds,
                obs_history_len=int(args.obs_history_len),
                attack_interval_ms=float(args.attack_interval_ms),
                max_episode_steps=int(args.episode_steps),
                launch_timeout_s=float(args.launch_timeout_s),
                step_timeout_s=float(args.step_timeout_s),
                runtime_dir=resolved_runtime_dir,
                baseline_gap_alpha=float(args.baseline_gap_alpha),
                baseline_hard_max=bool(args.baseline_hard_max),
                baseline_methods=baseline_methods,
                smooth_penalty_scale=float(resolved_smooth_penalty_scale),
                sync_guard_ms=float(args.sync_guard_ms),
                launch_retries=int(args.gap_launch_retries),
            )
        else:
            env = IndependentAttackEnv(
                repo_root=repo_root,
                launch_config=launch_config,
                bounds=online_attack_bounds,
                obs_history_len=int(args.obs_history_len),
                attack_interval_ms=float(args.attack_interval_ms),
                max_episode_steps=int(args.episode_steps),
                launch_timeout_s=float(args.launch_timeout_s),
                step_timeout_s=float(args.step_timeout_s),
                runtime_dir=resolved_runtime_dir,
                shared_bandwidth_action=shared_bandwidth_action,
                shared_loss_action=shared_loss_action,
                shared_delay_action=shared_delay_action,
                smooth_penalty_scale=float(resolved_smooth_penalty_scale),
            )
        base_venv = DummyVecEnv([lambda: Monitor(env)])
        if use_vecnormalize:
            venv = VecNormalize(base_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            venv = base_venv
        if int(args.ppo_n_steps) < 1:
            raise ValueError("--ppo-n-steps must be >= 1")
        if int(args.ppo_batch_size) < 1:
            raise ValueError("--ppo-batch-size must be >= 1")
        if int(args.ppo_n_epochs) < 1:
            raise ValueError("--ppo-n-epochs must be >= 1")
        if int(args.ppo_batch_size) > int(args.ppo_n_steps):
            raise ValueError("--ppo-batch-size must be <= --ppo-n-steps when using one environment")

        def _checkpoint_path(step: int, *, suffix: str | None = None) -> str:
            stem = f"{model_stem}-step{int(step):09d}"
            if suffix:
                stem = f"{stem}-{suffix}"
            return os.path.join(checkpoint_dir, f"{stem}.zip")

        def _checkpoint_vecnormalize_path(step: int, *, suffix: str | None = None) -> str:
            stem = f"{model_stem}-step{int(step):09d}"
            if suffix:
                stem = f"{stem}-{suffix}"
            return os.path.join(checkpoint_dir, f"{stem}.vecnormalize.pkl")

        class WandbInfoCallback(BaseCallback):
            def __init__(self, *, log_every: int = 100, checkpoint_every: int = 10_000) -> None:
                super().__init__()
                self._log_every = max(int(log_every), 1)
                self._checkpoint_every = max(int(checkpoint_every), 0)
                self._last_checkpoint_step = 0
                self._window_records: list[dict[str, float]] = []
                self._episode_records: list[dict[str, float]] = []

            def _save_checkpoint(self, *, suffix: str | None = None) -> None:
                if self.model is None or not model_stem:
                    return
                step = int(self.num_timesteps)
                if step <= 0:
                    return
                self.model.save(_checkpoint_path(step, suffix=suffix))
                if use_vecnormalize and isinstance(venv, VecNormalize):
                    venv.save(_checkpoint_vecnormalize_path(step, suffix=suffix))
                self._last_checkpoint_step = step

            def _on_step(self) -> bool:
                if wandb is None:
                    return True
                infos = self.locals.get("infos", [])
                if infos:
                    info = infos[0] if isinstance(infos, list) else infos
                    if isinstance(info, dict):
                        numeric_info = {
                            str(key): float(value)
                            for key, value in info.items()
                            if isinstance(value, (int, float, np.floating, np.integer))
                        }
                        step_payload = {
                            f"train/step/{target_key}": float(numeric_info[source_key])
                            for source_key, target_key in _WANDB_STEP_INFO_KEYS.items()
                            if source_key in numeric_info
                        }
                        if step_payload:
                            step_payload["train/num_timesteps"] = float(self.num_timesteps)
                            wandb.log(step_payload, step=int(self.num_timesteps))
                        self._window_records.append(numeric_info)
                        self._episode_records.append(numeric_info)

                        if "episode" in info and isinstance(info["episode"], dict):
                            episode = info["episode"]
                            episode_payload = {
                                "train/episode_return": float(episode.get("r", 0.0)),
                                "train/episode_length": float(episode.get("l", 0.0)),
                                "train/num_timesteps": float(self.num_timesteps),
                            }
                            episode_payload.update(
                                _aggregate_selected_metrics(
                                    self._episode_records,
                                    _WANDB_AGGREGATE_INFO_KEYS,
                                    prefix="train/episode_stats",
                                )
                            )
                            wandb.log(episode_payload, step=int(self.num_timesteps))
                            self._episode_records = []

                        if int(self.num_timesteps) % self._log_every != 0:
                            return True

                        payload = {}
                        for key, value in info.items():
                            if isinstance(value, (int, float, np.floating, np.integer)):
                                payload[str(key)] = float(value)
                        payload.update(
                            _aggregate_selected_metrics(
                                self._window_records,
                                _WANDB_AGGREGATE_INFO_KEYS,
                                prefix="train/window",
                            )
                        )
                        payload["train/num_timesteps"] = float(self.num_timesteps)
                        wandb.log(payload, step=int(self.num_timesteps))
                        self._window_records = []
                return True

            def _on_rollout_end(self) -> None:
                if self._checkpoint_every > 0 and int(self.num_timesteps) - int(self._last_checkpoint_step) >= self._checkpoint_every:
                    self._save_checkpoint()
                if wandb is None:
                    return
                logger_values = getattr(self.model.logger, "name_to_value", {})
                payload: dict[str, float] = {"train/num_timesteps": float(self.num_timesteps)}
                for key, value in logger_values.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        payload[f"sb3/{key}"] = float(value)
                wandb.log(payload, step=int(self.num_timesteps))

        hidden_layers = [int(args.policy_width) for _ in range(max(int(args.policy_depth), 1))]
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=float(args.lr),
            seed=int(args.seed),
            n_steps=int(args.ppo_n_steps),
            batch_size=int(args.ppo_batch_size),
            n_epochs=int(args.ppo_n_epochs),
            ent_coef=float(args.ppo_ent_coef),
            policy_kwargs={"net_arch": {"pi": hidden_layers, "vf": hidden_layers}},
            verbose=1,
        )
        config_payload = {
            **vars(args),
            **resolved_launch_config,
            "repo_root": repo_root,
            "baseline_methods": list(baseline_methods),
            "smooth_penalty_scale": float(resolved_smooth_penalty_scale),
            "run_namespace": run_namespace.metadata(),
            "policy_action_transform": (
                "log_unit_interval_shared_bandwidth" if (use_gap_objective or shared_bandwidth_action) else "identity"
            ),
            "vecnormalize_enabled": bool(use_vecnormalize),
            "checkpoint_dir": os.path.relpath(checkpoint_dir, repo_root)
            if os.path.commonpath([repo_root, checkpoint_dir]) == repo_root
            else checkpoint_dir,
            "checkpoint_every": int(args.checkpoint_every),
            "action_space_low": [float(x) for x in env.action_space.low.tolist()],
            "action_space_high": [float(x) for x in env.action_space.high.tolist()],
            "effective_action_low": [float(x) for x in online_attack_bounds.low.tolist()],
            "effective_action_high": [float(x) for x in online_attack_bounds.high.tolist()],
        }
        if checkpoint_config_path:
            save_json(checkpoint_config_path, config_payload)
        callback = WandbInfoCallback(
            log_every=int(args.wandb_log_every),
            checkpoint_every=int(args.checkpoint_every),
        )
        try:
            model.learn(total_timesteps=int(args.total_steps), callback=callback)
        except BaseException:
            if model_stem:
                callback._save_checkpoint(suffix="interrupted")
            raise

        model_path = os.path.join(out_dir, f"{model_stem}.zip")
        model.save(model_path)
        if use_vecnormalize and isinstance(venv, VecNormalize):
            vecnormalize_path = os.path.join(out_dir, f"{model_stem}.vecnormalize.pkl")
            venv.save(vecnormalize_path)
        config_payload["vecnormalize_path"] = (
            os.path.relpath(vecnormalize_path, repo_root)
            if vecnormalize_path is not None and os.path.commonpath([repo_root, vecnormalize_path]) == repo_root
            else vecnormalize_path
        )
        config_path = os.path.join(out_dir, f"{model_stem}.config.json")
        save_json(config_path, config_payload)

        if wandb_run is not None:
            payload: dict[str, float | str] = {
                "artifact/model_path": model_path,
                "artifact/config_path": config_path,
            }
            if vecnormalize_path is not None:
                payload["artifact/vecnormalize_path"] = vecnormalize_path
            wandb_run.log(payload)
        print(model_path)
    finally:
        if venv is not None:
            venv.close()
        elif env is not None:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()
        run_namespace.release()


if __name__ == "__main__":  # pragma: no cover
    main()
