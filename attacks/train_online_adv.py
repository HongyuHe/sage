from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import time
from typing import Any

from attacks.envs import OnlineSageAttackEnv
from attacks.online import SageLaunchConfig, acquire_run_namespace


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
        from stable_baselines3.common.monitor import Monitor  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

        return PPO, BaseCallback, Monitor, DummyVecEnv
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for online Sage attack training") from exc


def _try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore

        if not hasattr(wandb, "init"):
            raise ImportError("wandb import did not resolve to the package")
        return wandb
    except Exception:
        return None


def _resolve_path(repo_root: str, path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(repo_root, expanded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="attacks/output/models")
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--obs-history-len", type=int, default=4)
    parser.add_argument("--attack-interval-ms", type=float, default=100.0)
    parser.add_argument("--episode-steps", type=int, default=120)
    parser.add_argument("--launch-timeout-s", type=float, default=90.0)
    parser.add_argument("--step-timeout-s", type=float, default=10.0)
    parser.add_argument("--smooth-penalty-scale", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)

    parser.add_argument("--latency-ms", type=int, default=25)
    parser.add_argument("--port", type=int, default=5101)
    parser.add_argument("--downlink-trace", type=str, default="wired48")
    parser.add_argument("--uplink-trace", type=str, default="wired48")
    parser.add_argument("--iteration-id", type=int, default=0)
    parser.add_argument("--qsize-packets", type=int, default=128)
    parser.add_argument("--env-bw-mbps", type=int, default=48)
    parser.add_argument("--bw2-mbps", type=int, default=48)
    parser.add_argument("--trace-period-s", type=int, default=7)
    parser.add_argument("--sage-mode", type=int, default=0)
    parser.add_argument("--log-prefix", type=str, default="adv")
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--actor-id", type=int, default=900)
    parser.add_argument("--duration-steps", type=int, default=6000)
    parser.add_argument("--num-flows", type=int, default=1)
    parser.add_argument("--save-logs", type=int, default=0)
    parser.add_argument("--analyze-logs", type=int, default=0)
    parser.add_argument("--mm-adv-bin", type=str, default=None)

    parser.add_argument("--init-uplink-bw-mbps", type=float, default=None)
    parser.add_argument("--init-downlink-bw-mbps", type=float, default=None)
    parser.add_argument("--init-uplink-loss", type=float, default=0.0)
    parser.add_argument("--init-downlink-loss", type=float, default=0.0)
    parser.add_argument("--init-uplink-delay-ms", type=float, default=None)
    parser.add_argument("--init-downlink-delay-ms", type=float, default=None)
    parser.add_argument("--init-uplink-queue-packets", type=int, default=None)
    parser.add_argument("--init-downlink-queue-packets", type=int, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sage-online-adv")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-log-every", type=int, default=100)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    PPO, BaseCallback, Monitor, DummyVecEnv = _require_sb3()
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(args.actor_id),
        port=int(args.port),
        label=str(args.wandb_name or args.log_prefix or "train-online-adv"),
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
        wandb = _try_import_wandb()
        if wandb is None:
            raise RuntimeError("--wandb was set but the wandb package is unavailable")
        wandb_run = wandb.init(
            project=str(args.wandb_project),
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=str(args.wandb_mode),
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            config={**vars(args), **resolved_launch_config},
        )

    launch_config = replace(
        SageLaunchConfig(
            sage_script="sage_rl/sage.sh",
            latency_ms=int(args.latency_ms),
            port=int(args.port),
            downlink_trace=str(args.downlink_trace),
            uplink_trace=str(args.uplink_trace),
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
            initial_uplink_bw_mbps=args.init_uplink_bw_mbps,
            initial_downlink_bw_mbps=args.init_downlink_bw_mbps,
            initial_uplink_loss=args.init_uplink_loss,
            initial_downlink_loss=args.init_downlink_loss,
            initial_uplink_delay_ms=args.init_uplink_delay_ms,
            initial_downlink_delay_ms=args.init_downlink_delay_ms,
            initial_uplink_queue_packets=args.init_uplink_queue_packets,
            initial_downlink_queue_packets=args.init_downlink_queue_packets,
        ),
        port=int(run_namespace.port_base),
        actor_id=int(run_namespace.actor_id_base),
    )

    env = OnlineSageAttackEnv(
        repo_root=repo_root,
        launch_config=launch_config,
        obs_history_len=int(args.obs_history_len),
        attack_interval_ms=float(args.attack_interval_ms),
        max_episode_steps=int(args.episode_steps),
        launch_timeout_s=float(args.launch_timeout_s),
        step_timeout_s=float(args.step_timeout_s),
        smooth_penalty_scale=float(args.smooth_penalty_scale),
        reward_scale=float(args.reward_scale),
        runtime_dir=resolved_runtime_dir,
    )
    venv = DummyVecEnv([lambda: Monitor(env)])

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
            if not infos:
                return True
            info = infos[0] if isinstance(infos, list) else infos
            if not isinstance(info, dict):
                return True
            payload: dict[str, float | int] = {}
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    payload[str(key)] = value
            payload["train/num_timesteps"] = int(self.num_timesteps)
            wandb.log(payload, step=int(self.num_timesteps))
            return True

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=float(args.lr),
        seed=int(args.seed),
        policy_kwargs={"net_arch": {"pi": [128, 64], "vf": [128, 64]}},
        verbose=1,
    )
    callback = WandbInfoCallback(log_every=int(args.wandb_log_every))
    model.learn(total_timesteps=int(args.total_steps), callback=callback)

    out_dir = _resolve_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(out_dir, f"sage_adv_ppo_{stamp}_{run_namespace.run_id}.zip")
    model.save(model_path)

    config_path = os.path.join(out_dir, f"sage_adv_ppo_{stamp}_{run_namespace.run_id}.config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                **resolved_launch_config,
                "attack_mode": "independent",
                "run_namespace": run_namespace.metadata(),
                "action_space_low": [float(x) for x in env.action_space.low.tolist()],
                "action_space_high": [float(x) for x in env.action_space.high.tolist()],
            },
            f,
            indent=2,
            sort_keys=True,
        )

    if wandb_run is not None:
        wandb_run.log({"artifact/model_path": model_path})
        wandb_run.finish()

    run_namespace.release()
    print(model_path)


if __name__ == "__main__":  # pragma: no cover
    main()
