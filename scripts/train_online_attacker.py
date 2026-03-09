"""
Run this after `scripts/prepare_trace_splits.py`.

Example usage:
python scripts/train_online_attacker.py
   --train-manifest attacks/train/manifest.json\
   --total-steps 600000 --attack-interval-ms 100 --out-dir attacks/output/models\
   --wandb --wandb-project sage-online-train --wandb-name v1
   
python scripts/train_online_attacker.py \
  --train-manifest attacks/train/manifest.json \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --ppo-n-steps 256 \
  --ppo-batch-size 64 \
  --ppo-n-epochs 4 \
  --out-dir attacks/output/models \
  --wandb --wandb-project sage-online-train --wandb-name v1-fast

python scripts/train_online_attacker.py \
  --train-manifest attacks/train/manifest.json \
  --smooth-penalty-scale 0.05 \
  --attack-uplink-bw-min-mbps 5 --attack-uplink-bw-max-mbps 150 \
  --attack-downlink-bw-min-mbps 5 --attack-downlink-bw-max-mbps 150 \
  --attack-uplink-loss-min 0.0 --attack-uplink-loss-max 0.02 \
  --attack-downlink-loss-min 0.0 --attack-downlink-loss-max 0.02 \
  --attack-uplink-delay-min-ms 5 --attack-uplink-delay-max-ms 80 \
  --attack-downlink-delay-min-ms 5 --attack-downlink-delay-max-ms 80 \
  --total-steps 300000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --wandb --wandb-tags scratch --wandb-project sage-online-train --wandb-name hotnets19

python scripts/train_online_attacker.py \
  --train-manifest attacks/train/manifest.json \
  --smooth-penalty-scale 0.00 \
  --attack-uplink-bw-min-mbps 0 --attack-uplink-bw-max-mbps 2000 \
  --attack-downlink-bw-min-mbps 0 --attack-downlink-bw-max-mbps 2000 \
  --attack-uplink-loss-min 0.0 --attack-uplink-loss-max 0.8 \
  --attack-downlink-loss-min 0.0 --attack-downlink-loss-max 0.8 \
  --attack-uplink-delay-min-ms 0 --attack-uplink-delay-max-ms 1000 \
  --attack-downlink-delay-min-ms 0 --attack-downlink-delay-max-ms 1000 \
  --effective-bw-cap-mbps 2000 \
  --total-steps 30000 \
  --attack-interval-ms 100 \
  --out-dir attacks/output/models \
  --wandb --wandb-tags scratch --wandb-project sage-online-train --wandb-name rl-unconstrained

"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
import time
from typing import Any

import numpy as np

from attacks.envs.online_sage_env import AttackBounds
from attacks.online import SageLaunchConfig, acquire_run_namespace


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        IndependentAttackEnv,
        TraceConditionedAttackEnv,
        load_trace_entries,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        try_import_wandb,
    )
else:
    from ._trace_attack_common import (
        IndependentAttackEnv,
        TraceConditionedAttackEnv,
        load_trace_entries,
        materialize_trace_splits,
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
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

        return PPO, BaseCallback, Monitor, DummyVecEnv
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for online attacker training") from exc


def _ensure_manifest(repo_root: str, manifest_path: str) -> str:
    resolved = resolve_repo_path(repo_root, manifest_path)
    if os.path.exists(resolved):
        return resolved
    train_dir = resolve_repo_path(repo_root, "attacks/train")
    test_dir = resolve_repo_path(repo_root, "attacks/test")
    materialize_trace_splits(repo_root=repo_root, train_root=train_dir, test_root=test_dir)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing train manifest: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument(
        "--attack-mode",
        type=str,
        default="independent",
        choices=["independent", "trace_conditioned"],
    )
    parser.add_argument("--train-manifest", type=str, default="attacks/train/manifest.json")
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-mode", type=str, default="random", choices=["random", "round_robin"])
    parser.add_argument("--out-dir", type=str, default="attacks/output/models")
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--obs-history-len", type=int, default=4)
    parser.add_argument("--attack-interval-ms", type=float, default=100.0)
    parser.add_argument("--episode-steps", type=int, default=6000)
    parser.add_argument("--launch-timeout-s", type=float, default=90.0)
    parser.add_argument("--step-timeout-s", type=float, default=10.0)
    parser.add_argument("--smooth-penalty-scale", type=float, default=0.0)
    parser.add_argument("--reward-rate-weight", type=float, default=1.0)
    parser.add_argument("--reward-rtt-weight", type=float, default=0.05)
    parser.add_argument("--reward-loss-weight", type=float, default=2.0)
    parser.add_argument("--bw-scale-min", type=float, default=0.1)
    parser.add_argument("--bw-scale-max", type=float, default=2.0)
    parser.add_argument("--loss-max", type=float, default=0.15)
    parser.add_argument("--delay-max-ms", type=float, default=150.0)
    parser.add_argument("--effective-bw-cap-mbps", type=float, default=2000.0)
    parser.add_argument("--attack-uplink-bw-min-mbps", type=float, default=None)
    parser.add_argument("--attack-uplink-bw-max-mbps", type=float, default=None)
    parser.add_argument("--attack-downlink-bw-min-mbps", type=float, default=None)
    parser.add_argument("--attack-downlink-bw-max-mbps", type=float, default=None)
    parser.add_argument("--attack-uplink-loss-min", type=float, default=None)
    parser.add_argument("--attack-uplink-loss-max", type=float, default=None)
    parser.add_argument("--attack-downlink-loss-min", type=float, default=None)
    parser.add_argument("--attack-downlink-loss-max", type=float, default=None)
    parser.add_argument("--attack-uplink-delay-min-ms", type=float, default=None)
    parser.add_argument("--attack-uplink-delay-max-ms", type=float, default=None)
    parser.add_argument("--attack-downlink-delay-min-ms", type=float, default=None)
    parser.add_argument("--attack-downlink-delay-max-ms", type=float, default=None)
    parser.add_argument("--policy-width", type=int, default=128)
    parser.add_argument("--policy-depth", type=int, default=2)
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-n-epochs", type=int, default=10)

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
    PPO, BaseCallback, Monitor, DummyVecEnv = _require_sb3()
    manifest_path: str | None = None
    train_entries: list[Any] = []
    if str(args.attack_mode) == "trace_conditioned":
        manifest_path = _ensure_manifest(repo_root, str(args.train_manifest))
        train_entries = load_trace_entries(manifest_path)
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(args.actor_id),
        port=int(args.port),
        label=str(args.wandb_name or args.log_prefix or "train-online-attacker"),
    )
    resolved_runtime_dir = run_namespace.runtime_dir
    resolved_launch_config = {
        "run_id": run_namespace.run_id,
        "runtime_dir_resolved": resolved_runtime_dir,
        "launch_port_base_resolved": int(run_namespace.port_base),
        "launch_actor_id_base_resolved": int(run_namespace.actor_id_base),
        "runtime_slot": int(run_namespace.slot),
    }

    wandb = None
    wandb_run = None
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
                "train_manifest_resolved": manifest_path,
                "num_train_traces": len(train_entries),
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
    attack_uplink_loss_min = float(args.attack_uplink_loss_min) if args.attack_uplink_loss_min is not None else 0.0
    attack_uplink_loss_max = (
        float(args.attack_uplink_loss_max) if args.attack_uplink_loss_max is not None else float(args.loss_max)
    )
    attack_downlink_loss_min = (
        float(args.attack_downlink_loss_min) if args.attack_downlink_loss_min is not None else 0.0
    )
    attack_downlink_loss_max = (
        float(args.attack_downlink_loss_max)
        if args.attack_downlink_loss_max is not None
        else float(args.loss_max)
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
    online_attack_bounds = AttackBounds(
        uplink_bw_mbps=(attack_uplink_bw_min, attack_uplink_bw_max),
        downlink_bw_mbps=(attack_downlink_bw_min, attack_downlink_bw_max),
        uplink_loss=(attack_uplink_loss_min, attack_uplink_loss_max),
        downlink_loss=(attack_downlink_loss_min, attack_downlink_loss_max),
        uplink_delay_ms=(attack_uplink_delay_min, attack_uplink_delay_max),
        downlink_delay_ms=(attack_downlink_delay_min, attack_downlink_delay_max),
    )

    if str(args.attack_mode) == "trace_conditioned":
        env = TraceConditionedAttackEnv(
            repo_root=repo_root,
            trace_entries=train_entries,
            launch_config=launch_config,
            online_attack_bounds=online_attack_bounds,
            obs_history_len=int(args.obs_history_len),
            attack_interval_ms=float(args.attack_interval_ms),
            max_episode_steps=int(args.episode_steps),
            launch_timeout_s=float(args.launch_timeout_s),
            step_timeout_s=float(args.step_timeout_s),
            runtime_dir=resolved_runtime_dir,
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
            bw_scale_min=float(args.bw_scale_min),
            bw_scale_max=float(args.bw_scale_max),
            effective_bw_cap_mbps=float(args.effective_bw_cap_mbps),
            loss_min=0.0,
            loss_max=float(args.loss_max),
            delay_min_ms=0.0,
            delay_max_ms=float(args.delay_max_ms),
            reward_rate_weight=float(args.reward_rate_weight),
            reward_rtt_weight=float(args.reward_rtt_weight),
            reward_loss_weight=float(args.reward_loss_weight),
            smooth_penalty_scale=float(args.smooth_penalty_scale),
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
            reward_rate_weight=float(args.reward_rate_weight),
            reward_rtt_weight=float(args.reward_rtt_weight),
            reward_loss_weight=float(args.reward_loss_weight),
            smooth_penalty_scale=float(args.smooth_penalty_scale),
        )
    venv = DummyVecEnv([lambda: Monitor(env)])
    if int(args.ppo_n_steps) < 1:
        raise ValueError("--ppo-n-steps must be >= 1")
    if int(args.ppo_batch_size) < 1:
        raise ValueError("--ppo-batch-size must be >= 1")
    if int(args.ppo_n_epochs) < 1:
        raise ValueError("--ppo-n-epochs must be >= 1")
    if int(args.ppo_batch_size) > int(args.ppo_n_steps):
        raise ValueError("--ppo-batch-size must be <= --ppo-n-steps when using one environment")

    class WandbInfoCallback(BaseCallback):
        def __init__(self, *, log_every: int = 100) -> None:
            super().__init__()
            self._log_every = max(int(log_every), 1)

        def _on_step(self) -> bool:
            if wandb is None:
                return True
            if int(self.num_timesteps) % self._log_every != 0:
                return True
            infos = self.locals.get("infos", [])
            if infos:
                info = infos[0] if isinstance(infos, list) else infos
                if isinstance(info, dict):
                    payload = {}
                    for key, value in info.items():
                        if isinstance(value, (int, float, np.floating, np.integer)):
                            payload[str(key)] = float(value)
                    if "episode" in info and isinstance(info["episode"], dict):
                        episode = info["episode"]
                        payload["train/episode_return"] = float(episode.get("r", 0.0))
                        payload["train/episode_length"] = float(episode.get("l", 0.0))
                    payload["train/num_timesteps"] = float(self.num_timesteps)
                    wandb.log(payload, step=int(self.num_timesteps))
            return True

        def _on_rollout_end(self) -> None:
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
        policy_kwargs={"net_arch": {"pi": hidden_layers, "vf": hidden_layers}},
        verbose=1,
    )
    callback = WandbInfoCallback(log_every=int(args.wandb_log_every))
    model.learn(total_timesteps=int(args.total_steps), callback=callback)

    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_prefix = "trace_online_adv_ppo" if str(args.attack_mode) == "trace_conditioned" else "online_adv_ppo"
    model_stem = f"{model_prefix}_{stamp}_{run_namespace.run_id}"
    model_path = os.path.join(out_dir, f"{model_stem}.zip")
    model.save(model_path)

    config_payload = {
        **vars(args),
        **resolved_launch_config,
        "repo_root": repo_root,
        "train_manifest_resolved": manifest_path,
        "num_train_traces": len(train_entries),
        "run_namespace": run_namespace.metadata(),
        "action_space_low": [float(x) for x in env.action_space.low.tolist()],
        "action_space_high": [float(x) for x in env.action_space.high.tolist()],
    }
    config_path = os.path.join(out_dir, f"{model_stem}.config.json")
    save_json(config_path, config_payload)

    usage_path = None
    if str(args.attack_mode) == "trace_conditioned":
        usage_path = os.path.join(out_dir, f"{model_stem}.trace_usage.json")
        save_json(usage_path, {"trace_usage_counts": env.trace_usage_counts()})

    if wandb_run is not None:
        payload: dict[str, float | str] = {
            "artifact/model_path": model_path,
            "artifact/config_path": config_path,
            "artifact/num_train_traces": float(len(train_entries)),
        }
        if usage_path is not None:
            payload["artifact/trace_usage_path"] = usage_path
        wandb_run.log(payload)
        wandb_run.finish()

    run_namespace.release()
    print(model_path)


if __name__ == "__main__":  # pragma: no cover
    main()
