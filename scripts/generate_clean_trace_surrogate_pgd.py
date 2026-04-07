"""
Generate clean-trace surrogate-PGD adversarial schedules in the standard manifest format.

Example usage:
time python scripts/generate_clean_trace_surrogate_pgd.py \
  --surrogate-path attacks/output/models/gap-constrained-all-loss_50ms_300k_clean_trace_surrogate.pt \
  --config-path attacks/models/gap_adv_20260321_gap-constrained-all-loss_50ms_300k.config.json \
  --clean-manifest attacks/test/manifest.json \
  --out-dir attacks/adv_traces/gap-constrained-all-loss_50ms_300k_clean_trace_pgd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from attacks.envs import ParallelGapAttackEnv, baseline_methods_from_config
from attacks.online import SageLaunchConfig, acquire_run_namespace
from attacks.surrogate import (
    bandwidth_projection_bounds,
    load_clean_trace_sequences,
    load_clean_trace_surrogate_from_checkpoint,
    replay_bounds_for_action_schedules,
    shared_components_to_action_schedule,
)


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        attack_bounds_from_config,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        utc_now_iso,
        write_bandwidth_trace,
    )
else:
    from ._trace_attack_common import (
        attack_bounds_from_config,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        utc_now_iso,
        write_bandwidth_trace,
    )


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _load_training_config(repo_root: str, config_path: str) -> tuple[str, dict[str, Any]]:
    resolved = resolve_repo_path(repo_root, config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing training config: {resolved}")
    return resolved, _load_json(resolved)


def _ensure_clean_manifest(repo_root: str, manifest_path: str) -> str:
    resolved = resolve_repo_path(repo_root, manifest_path)
    if os.path.exists(resolved):
        return resolved
    materialize_trace_splits(
        repo_root=repo_root,
        train_root=resolve_repo_path(repo_root, "attacks/train"),
        test_root=resolve_repo_path(repo_root, "attacks/test"),
    )
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing clean manifest: {resolved}")
    return resolved


def _resolved_launch_config(
    *,
    config_payload: dict[str, Any],
    run_namespace,
) -> SageLaunchConfig:
    return replace(
        SageLaunchConfig(
            sage_script="sage_rl/sage.sh",
            latency_ms=int(config_payload.get("latency_ms", 25)),
            port=int(config_payload.get("port", 5101)),
            downlink_trace="wired48",
            uplink_trace="wired48",
            iteration_id=int(config_payload.get("iteration_id", 0)),
            qsize_packets=int(config_payload.get("qsize_packets", 128)),
            env_bw_mbps=int(config_payload.get("env_bw_mbps", 48)),
            bw2_mbps=int(config_payload.get("bw2_mbps", 48)),
            trace_period_s=int(config_payload.get("trace_period_s", 7)),
            first_time_mode=int(config_payload.get("sage_mode", 0)),
            log_prefix=str(config_payload.get("log_prefix", "clean-trace-surrogate-pgd")),
            duration_seconds=int(config_payload.get("duration_seconds", 60)),
            actor_id=int(config_payload.get("actor_id", 900)),
            duration_steps=int(config_payload.get("duration_steps", 6000)),
            num_flows=int(config_payload.get("num_flows", 1)),
            save_logs=int(config_payload.get("save_logs", 0)),
            analyze_logs=int(config_payload.get("analyze_logs", 0)),
            mm_adv_bin=config_payload.get("mm_adv_bin"),
            initial_uplink_loss=float(config_payload.get("init_uplink_loss", 0.0)),
            initial_downlink_loss=float(config_payload.get("init_downlink_loss", 0.0)),
            initial_uplink_delay_ms=config_payload.get("init_uplink_delay_ms"),
            initial_downlink_delay_ms=config_payload.get("init_downlink_delay_ms"),
            initial_uplink_queue_packets=config_payload.get("init_uplink_queue_packets"),
            initial_downlink_queue_packets=config_payload.get("init_downlink_queue_packets"),
        ),
        port=int(run_namespace.port_base),
        actor_id=int(run_namespace.actor_id_base),
    )


def _trace_set_name(out_dir: str) -> str:
    return os.path.basename(os.path.abspath(out_dir.rstrip(os.sep))) or "generated"


def _build_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _project_bandwidth(
    adv_bw: torch.Tensor,
    base_bw: torch.Tensor,
    *,
    mask: torch.Tensor,
    eps_abs: float,
    eps_rel: float,
    global_low: float,
    global_high: float,
) -> torch.Tensor:
    projected = adv_bw
    if float(eps_abs) > 0.0:
        projected = torch.maximum(projected, base_bw - float(eps_abs))
        projected = torch.minimum(projected, base_bw + float(eps_abs))
    if float(eps_rel) > 0.0:
        rel_low = base_bw * float(max(0.0, 1.0 - float(eps_rel)))
        rel_high = base_bw * float(1.0 + float(eps_rel))
        projected = torch.maximum(projected, rel_low)
        projected = torch.minimum(projected, rel_high)
    projected = torch.clamp(projected, min=float(global_low), max=float(global_high))
    return torch.where(mask, projected, base_bw)


def _project_loss(
    adv_loss: torch.Tensor,
    base_loss: torch.Tensor,
    *,
    mask: torch.Tensor,
    eps_abs: float,
    global_low: float,
    global_high: float,
) -> torch.Tensor:
    projected = adv_loss
    if float(eps_abs) > 0.0:
        projected = torch.maximum(projected, base_loss - float(eps_abs))
        projected = torch.minimum(projected, base_loss + float(eps_abs))
    projected = torch.clamp(projected, min=float(global_low), max=float(global_high))
    return torch.where(mask, projected, base_loss)


def _apply_gradient_step(
    values: torch.Tensor,
    grad: torch.Tensor,
    *,
    alpha: float,
    grad_mode: str,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_grad = torch.where(mask, grad, torch.zeros_like(grad))
    if str(grad_mode) == "raw":
        grad_norm = masked_grad.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        step = float(alpha) * masked_grad / grad_norm
    else:
        step = float(alpha) * masked_grad.sign()
    return values.detach() + step


def _pgd_attack_batch(
    model,
    *,
    shared_bw: torch.Tensor,
    shared_loss: torch.Tensor,
    lengths: torch.Tensor,
    steps: int,
    alpha_bw: float,
    alpha_loss: float,
    eps_abs_bw: float,
    eps_rel_bw: float,
    eps_abs_loss: float,
    global_bw_low: float,
    global_bw_high: float,
    global_loss_low: float,
    global_loss_high: float,
    optimize_loss: bool,
    random_start: bool,
    grad_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    base_bw = shared_bw.detach().clone()
    base_loss = shared_loss.detach().clone()
    mask = _build_valid_mask(lengths, width=shared_bw.shape[1])

    adv_bw = base_bw.clone()
    adv_loss = base_loss.clone()
    if bool(random_start):
        if float(eps_abs_bw) > 0.0 or float(eps_rel_bw) > 0.0:
            noise_bw = torch.zeros_like(base_bw)
            if float(eps_abs_bw) > 0.0:
                noise_bw = noise_bw + torch.empty_like(base_bw).uniform_(-float(eps_abs_bw), float(eps_abs_bw))
            if float(eps_rel_bw) > 0.0:
                rel_noise = torch.empty_like(base_bw).uniform_(-float(eps_rel_bw), float(eps_rel_bw))
                noise_bw = noise_bw + base_bw * rel_noise
            adv_bw = _project_bandwidth(
                base_bw + noise_bw,
                base_bw,
                mask=mask,
                eps_abs=eps_abs_bw,
                eps_rel=eps_rel_bw,
                global_low=global_bw_low,
                global_high=global_bw_high,
            )
        if bool(optimize_loss) and float(eps_abs_loss) > 0.0:
            noise_loss = torch.empty_like(base_loss).uniform_(-float(eps_abs_loss), float(eps_abs_loss))
            adv_loss = _project_loss(
                base_loss + noise_loss,
                base_loss,
                mask=mask,
                eps_abs=eps_abs_loss,
                global_low=global_loss_low,
                global_high=global_loss_high,
            )

    with torch.no_grad():
        clean_pred = model(base_bw, base_loss, lengths)

    for _ in range(int(steps)):
        adv_bw_var = adv_bw.detach().clone().requires_grad_(True)
        adv_loss_var = adv_loss.detach().clone().requires_grad_(bool(optimize_loss))
        pred = model(adv_bw_var, adv_loss_var, lengths)
        objective = pred.sum()
        if bool(optimize_loss):
            grad_bw, grad_loss = torch.autograd.grad(
                objective,
                [adv_bw_var, adv_loss_var],
                retain_graph=False,
                create_graph=False,
            )
        else:
            grad_bw = torch.autograd.grad(
                objective,
                adv_bw_var,
                retain_graph=False,
                create_graph=False,
            )[0]
            grad_loss = None

        adv_bw = _apply_gradient_step(
            adv_bw_var,
            grad_bw,
            alpha=float(alpha_bw),
            grad_mode=str(grad_mode),
            mask=mask,
        )
        adv_bw = _project_bandwidth(
            adv_bw,
            base_bw,
            mask=mask,
            eps_abs=eps_abs_bw,
            eps_rel=eps_rel_bw,
            global_low=global_bw_low,
            global_high=global_bw_high,
        )

        if bool(optimize_loss) and grad_loss is not None:
            adv_loss = _apply_gradient_step(
                adv_loss_var,
                grad_loss,
                alpha=float(alpha_loss),
                grad_mode=str(grad_mode),
                mask=mask,
            )
            adv_loss = _project_loss(
                adv_loss,
                base_loss,
                mask=mask,
                eps_abs=eps_abs_loss,
                global_low=global_loss_low,
                global_high=global_loss_high,
            )

    with torch.no_grad():
        adv_pred = model(adv_bw, adv_loss, lengths)
    return adv_bw.detach(), adv_loss.detach(), clean_pred.detach(), adv_pred.detach()


def _effective_actions_from_step_records(
    *,
    step_records: list[dict[str, Any]],
    fallback_schedule: list[np.ndarray],
) -> list[np.ndarray]:
    effective_actions: list[np.ndarray] = []
    for index, record in enumerate(step_records):
        values = record.get("effective_action")
        if isinstance(values, list) and len(values) == 6:
            effective_actions.append(np.asarray(values, dtype=np.float32))
            continue
        if index < len(fallback_schedule):
            effective_actions.append(np.asarray(fallback_schedule[index], dtype=np.float32))
    if not effective_actions:
        effective_actions = [np.asarray(action, dtype=np.float32) for action in fallback_schedule]
    return effective_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean-trace surrogate-PGD schedules for Sage.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--surrogate-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--clean-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num-traces", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--alpha-bw", type=float, default=1.0)
    parser.add_argument("--eps-abs-bw", type=float, default=10.0)
    parser.add_argument("--eps-rel-bw", type=float, default=0.0)
    parser.add_argument("--optimize-loss", action="store_true")
    parser.add_argument("--alpha-loss", type=float, default=0.005)
    parser.add_argument("--eps-abs-loss", type=float, default=0.02)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--grad-mode", choices=["sign", "raw"], default="sign")
    args = parser.parse_args()

    repo_root = os.path.abspath(str(args.repo_root))
    surrogate_path = resolve_repo_path(repo_root, str(args.surrogate_path))
    config_path, config_payload = _load_training_config(repo_root, str(args.config_path))
    clean_manifest_path = _ensure_clean_manifest(repo_root, str(args.clean_manifest))
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    attack_mode = str(config_payload.get("attack_mode", "independent_gap"))
    if attack_mode != "independent_gap":
        raise ValueError(
            f"clean-trace surrogate PGD currently supports only attack_mode=independent_gap, got {attack_mode!r}"
        )

    model, checkpoint_payload = load_clean_trace_surrogate_from_checkpoint(
        surrogate_path,
        device=str(args.device),
        eval_mode=True,
    )
    sequences = load_clean_trace_sequences(
        manifest_path=clean_manifest_path,
        config_payload=config_payload,
        limit=int(args.num_traces),
    )
    if not sequences:
        raise RuntimeError("no clean trace sequences were loaded from the manifest")

    base_bounds = attack_bounds_from_config(config_payload)
    clean_replay_bounds = replay_bounds_for_action_schedules(
        base_bounds,
        [sequence.to_action_schedule() for sequence in sequences],
    )
    global_loss_low = max(float(clean_replay_bounds.uplink_loss[0]), float(clean_replay_bounds.downlink_loss[0]))
    global_loss_high = min(float(clean_replay_bounds.uplink_loss[1]), float(clean_replay_bounds.downlink_loss[1]))
    if float(global_loss_high) <= float(global_loss_low) + 1e-12:
        optimize_loss = False
    else:
        optimize_loss = bool(args.optimize_loss)
    if bool(args.optimize_loss) and not optimize_loss:
        raise ValueError("loss optimization was requested but the config does not expose a positive shared loss range")

    max_len = max(sequence.num_steps for sequence in sequences)
    x_shared_bw = np.zeros((len(sequences), max_len), dtype=np.float32)
    x_shared_loss = np.zeros((len(sequences), max_len), dtype=np.float32)
    x_len = np.zeros((len(sequences),), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        length = int(sequence.num_steps)
        x_shared_bw[index, :length] = sequence.shared_bandwidth_mbps
        x_shared_loss[index, :length] = sequence.shared_loss_rate
        x_len[index] = length
    global_bw_low, global_bw_high = bandwidth_projection_bounds(
        x_shared_bw,
        x_len,
        eps_abs=float(args.eps_abs_bw),
        eps_rel=float(args.eps_rel_bw),
    )

    x_shared_bw_t = torch.from_numpy(x_shared_bw).float().to(str(args.device))
    x_shared_loss_t = torch.from_numpy(x_shared_loss).float().to(str(args.device))
    x_len_t = torch.from_numpy(x_len).long().to(str(args.device))

    adv_bw_chunks: list[np.ndarray] = []
    adv_loss_chunks: list[np.ndarray] = []
    clean_pred_chunks: list[np.ndarray] = []
    adv_pred_chunks: list[np.ndarray] = []
    delta_bw_chunks: list[np.ndarray] = []
    delta_loss_chunks: list[np.ndarray] = []
    for start in range(0, len(sequences), int(args.batch)):
        end = min(start + int(args.batch), len(sequences))
        adv_bw_b, adv_loss_b, clean_pred_b, adv_pred_b = _pgd_attack_batch(
            model,
            shared_bw=x_shared_bw_t[start:end],
            shared_loss=x_shared_loss_t[start:end],
            lengths=x_len_t[start:end],
            steps=int(args.steps),
            alpha_bw=float(args.alpha_bw),
            alpha_loss=float(args.alpha_loss),
            eps_abs_bw=float(args.eps_abs_bw),
            eps_rel_bw=float(args.eps_rel_bw),
            eps_abs_loss=float(args.eps_abs_loss),
            global_bw_low=float(global_bw_low),
            global_bw_high=float(global_bw_high),
            global_loss_low=float(global_loss_low),
            global_loss_high=float(global_loss_high),
            optimize_loss=bool(optimize_loss),
            random_start=bool(args.random_start),
            grad_mode=str(args.grad_mode),
        )
        adv_bw_np = adv_bw_b.cpu().numpy().astype(np.float32, copy=False)
        adv_loss_np = adv_loss_b.cpu().numpy().astype(np.float32, copy=False)
        clean_pred_np = clean_pred_b.cpu().numpy().astype(np.float32, copy=False)
        adv_pred_np = adv_pred_b.cpu().numpy().astype(np.float32, copy=False)

        adv_bw_chunks.append(adv_bw_np)
        adv_loss_chunks.append(adv_loss_np)
        clean_pred_chunks.append(clean_pred_np)
        adv_pred_chunks.append(adv_pred_np)
        delta_bw_chunks.append(np.max(np.abs(adv_bw_np - x_shared_bw[start:end]), axis=1))
        delta_loss_chunks.append(np.max(np.abs(adv_loss_np - x_shared_loss[start:end]), axis=1))

    adv_bw = np.concatenate(adv_bw_chunks, axis=0)
    adv_loss = np.concatenate(adv_loss_chunks, axis=0)
    clean_pred = np.concatenate(clean_pred_chunks, axis=0)
    adv_pred = np.concatenate(adv_pred_chunks, axis=0)
    pred_gain = adv_pred - clean_pred
    delta_bw = np.concatenate(delta_bw_chunks, axis=0).astype(np.float32, copy=False)
    delta_loss = np.concatenate(delta_loss_chunks, axis=0).astype(np.float32, copy=False)

    adv_schedules: list[list[np.ndarray]] = []
    for index, sequence in enumerate(sequences):
        length = int(sequence.num_steps)
        adv_schedules.append(
            shared_components_to_action_schedule(
                shared_bandwidth_mbps=adv_bw[index, :length],
                shared_loss_rate=adv_loss[index, :length],
                uplink_delay_ms=float(sequence.uplink_delay_ms),
                downlink_delay_ms=float(sequence.downlink_delay_ms),
            )
        )

    baseline_methods = baseline_methods_from_config(config_payload)
    trace_set_name = _trace_set_name(out_dir)
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=f"clean-trace-surrogate-pgd-{trace_set_name}",
        ports_per_run=len(baseline_methods) + 1,
    )
    resolved_runtime_dir = run_namespace.runtime_dir
    adv_replay_bounds = replay_bounds_for_action_schedules(base_bounds, adv_schedules)

    generated_entries: list[dict[str, Any]] = []
    per_trace_summary: list[dict[str, Any]] = []
    env: ParallelGapAttackEnv | None = None
    try:
        env = ParallelGapAttackEnv(
            repo_root=repo_root,
            launch_config=_resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace),
            bounds=adv_replay_bounds,
            obs_history_len=int(config_payload.get("obs_history_len", 4)),
            attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
            max_episode_steps=int(config_payload.get("episode_steps", 6000)),
            launch_timeout_s=float(config_payload.get("launch_timeout_s", 90.0)),
            step_timeout_s=float(config_payload.get("step_timeout_s", 10.0)),
            runtime_dir=resolved_runtime_dir,
            baseline_gap_alpha=float(config_payload.get("baseline_gap_alpha", 2.0)),
            baseline_hard_max=bool(config_payload.get("baseline_hard_max", False)),
            baseline_methods=baseline_methods,
            smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
            sync_guard_ms=float(config_payload.get("sync_guard_ms", 25.0)),
            launch_retries=int(config_payload.get("gap_launch_retries", 6)),
            shared_bin_loss_enabled=bool(config_payload.get("shared_bin_loss_enabled", False)),
            shared_bin_loss_bin_ms=float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
        )
        for trace_index, (sequence, action_schedule) in enumerate(zip(sequences, adv_schedules)):
            result = run_online_policy_episode(
                env,
                action_fn=lambda observation, info, step, schedule=action_schedule: schedule[min(step, len(schedule) - 1)],
                max_steps=len(action_schedule),
                episode_id=sequence.trace_name,
            )
            effective_actions = _effective_actions_from_step_records(
                step_records=result.step_records,
                fallback_schedule=action_schedule,
            )
            bundle_dir = os.path.join(out_dir, f"{trace_index:03d}-{sequence.trace_id}")
            os.makedirs(bundle_dir, exist_ok=True)

            uplink_trace_path = os.path.join(bundle_dir, "uplink.trace")
            downlink_trace_path = os.path.join(bundle_dir, "downlink.trace")
            write_bandwidth_trace(
                bandwidth_mbps=[float(action[0]) for action in effective_actions],
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                out_path=uplink_trace_path,
            )
            write_bandwidth_trace(
                bandwidth_mbps=[float(action[1]) for action in effective_actions],
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                out_path=downlink_trace_path,
            )

            summary_row = {
                "trace_id": str(sequence.trace_id),
                "trace_name": str(sequence.trace_name),
                "pred_clean": float(clean_pred[trace_index]),
                "pred_adv": float(adv_pred[trace_index]),
                "pred_gain": float(pred_gain[trace_index]),
                "delta_linf_bw_mbps": float(delta_bw[trace_index]),
                "delta_linf_loss_rate": float(delta_loss[trace_index]),
                "actual_episode_total_reward": float(result.total_reward),
                "actual_gap_value_mean": float(result.metrics.get("gap_value_mean", float("nan"))),
                "actual_gap_best_baseline_gap_mean": float(
                    result.metrics.get("gap_best_baseline_gap_mean", float("nan"))
                ),
            }
            per_trace_summary.append(summary_row)

            schedule_payload = {
                "created_at_utc": utc_now_iso(),
                "attack_mode": attack_mode,
                "generation_method": "clean_trace_surrogate_pgd",
                "trace_id": str(sequence.trace_id),
                "trace_name": str(sequence.trace_name),
                "source_trace": dict(sequence.source_trace),
                "model_path": surrogate_path,
                "surrogate_path": surrogate_path,
                "generation_model_type": "clean_trace_surrogate",
                "training_config_path": config_path,
                "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
                "num_steps": int(result.num_steps),
                "pgd": {
                    "steps": int(args.steps),
                    "alpha_bw": float(args.alpha_bw),
                    "eps_abs_bw": float(args.eps_abs_bw),
                    "eps_rel_bw": float(args.eps_rel_bw),
                    "optimize_loss": bool(optimize_loss),
                    "alpha_loss": float(args.alpha_loss),
                    "eps_abs_loss": float(args.eps_abs_loss),
                    "random_start": bool(args.random_start),
                    "grad_mode": str(args.grad_mode),
                },
                "surrogate_target_key": str(checkpoint_payload.get("target_key", "")),
                "pred_clean": float(clean_pred[trace_index]),
                "pred_adv": float(adv_pred[trace_index]),
                "pred_gain": float(pred_gain[trace_index]),
                "delta_linf_bw_mbps": float(delta_bw[trace_index]),
                "delta_linf_loss_rate": float(delta_loss[trace_index]),
                "metrics": result.metrics,
                "steps": result.step_records,
            }
            schedule_path = os.path.join(bundle_dir, "schedule.json")
            save_json(schedule_path, schedule_payload)

            generated_entries.append(
                {
                    "trace_id": str(sequence.trace_id),
                    "trace_name": str(sequence.trace_name),
                    "source_trace": dict(sequence.source_trace),
                    "bundle_dir": os.path.relpath(bundle_dir, repo_root),
                    "schedule_path": os.path.relpath(schedule_path, repo_root),
                    "uplink_trace_path": os.path.relpath(uplink_trace_path, repo_root),
                    "downlink_trace_path": os.path.relpath(downlink_trace_path, repo_root),
                    "metrics": result.metrics,
                    "pred_gain": float(pred_gain[trace_index]),
                    "delta_linf_bw_mbps": float(delta_bw[trace_index]),
                    "delta_linf_loss_rate": float(delta_loss[trace_index]),
                }
            )
    finally:
        if env is not None:
            env.close()
        run_namespace.release()

    summary_path = os.path.join(out_dir, "attack_summary.json")
    save_json(
        summary_path,
        {
            "created_at_utc": utc_now_iso(),
            "source_manifest": clean_manifest_path,
            "surrogate_path": surrogate_path,
            "training_config_path": config_path,
            "num_traces": len(sequences),
            "pred_gain_mean": float(np.mean(pred_gain)),
            "pred_gain_max": float(np.max(pred_gain)),
            "delta_linf_bw_mbps_mean": float(np.mean(delta_bw)),
            "delta_linf_bw_mbps_max": float(np.max(delta_bw)),
            "delta_linf_loss_rate_mean": float(np.mean(delta_loss)),
            "delta_linf_loss_rate_max": float(np.max(delta_loss)),
            "per_trace": per_trace_summary,
        },
    )

    generated_manifest_path = os.path.join(out_dir, "generated_manifest.json")
    save_json(
        generated_manifest_path,
        {
            "created_at_utc": utc_now_iso(),
            "repo_root": repo_root,
            "trace_set_name": trace_set_name,
            "attack_mode": attack_mode,
            "generation_method": "clean_trace_surrogate_pgd",
            "baseline_methods": list(baseline_methods),
            "model_path": surrogate_path,
            "surrogate_path": surrogate_path,
            "training_config_path": config_path,
            "clean_manifest_resolved": clean_manifest_path,
            "test_manifest_resolved": clean_manifest_path,
            "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
            "num_reference_test_traces": len(sequences),
            "num_generated_traces": len(generated_entries),
            "pgd": {
                "steps": int(args.steps),
                "alpha_bw": float(args.alpha_bw),
                "eps_abs_bw": float(args.eps_abs_bw),
                "eps_rel_bw": float(args.eps_rel_bw),
                "optimize_loss": bool(optimize_loss),
                "alpha_loss": float(args.alpha_loss),
                "eps_abs_loss": float(args.eps_abs_loss),
                "random_start": bool(args.random_start),
                "grad_mode": str(args.grad_mode),
            },
            "generated_entries": generated_entries,
        },
    )
    print(generated_manifest_path)


if __name__ == "__main__":
    main()
