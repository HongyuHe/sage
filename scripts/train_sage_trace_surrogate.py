"""
Train a clean-trace Sage surrogate regressor.

Example usage:
time python scripts/train_sage_trace_surrogate.py \
  --data attacks/output/surrogate-datasets/gap-constrained-all-loss_50ms_300k/clean_trace_surrogate_dataset.npz \
  --out attacks/output/models/gap-constrained-all-loss_50ms_300k_clean_trace_surrogate.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from attacks.surrogate import CleanTraceSurrogateRegressor


class CleanTraceSurrogateDataset(Dataset):
    def __init__(self, data_path: str, *, target_key: str, max_env_error_rate: float) -> None:
        data_path_obj = Path(str(data_path))
        if data_path_obj.is_dir():
            npz_paths = sorted(data_path_obj.glob("*.npz"))
        elif data_path_obj.is_file():
            npz_paths = [data_path_obj]
        else:
            raise ValueError(f"--data must be a .npz file or directory: {data_path}")
        if not npz_paths:
            raise ValueError(f"No .npz files found under: {data_path}")

        loaded: list[dict[str, np.ndarray]] = []
        max_len = 0
        total_rows = 0
        for npz_path in npz_paths:
            with np.load(npz_path, allow_pickle=False) as data:
                x_bw = np.asarray(data["X_shared_bw"], dtype=np.float32)
                x_loss = np.asarray(data["X_shared_loss"], dtype=np.float32)
                x_len = np.asarray(data["X_len"], dtype=np.int64)
                if target_key not in data.files:
                    raise ValueError(f"{npz_path}: missing target key {target_key!r}")
                y_target = np.asarray(data[target_key], dtype=np.float32)
                trace_name = (
                    np.asarray(data["trace_name"]).astype(str)
                    if "trace_name" in data.files
                    else np.asarray([f"trace_{index:04d}" for index in range(x_bw.shape[0])]).astype(str)
                )
                env_error_steps = (
                    np.asarray(data["Y_env_error_steps"], dtype=np.float32)
                    if "Y_env_error_steps" in data.files
                    else np.zeros((x_bw.shape[0],), dtype=np.float32)
                )
                replay_num_steps = (
                    np.asarray(data["Y_num_steps"], dtype=np.float32)
                    if "Y_num_steps" in data.files
                    else np.asarray(x_len, dtype=np.float32)
                )

                if x_bw.shape != x_loss.shape:
                    raise ValueError(f"{npz_path}: X_shared_bw and X_shared_loss must have matching shape")
                if x_bw.ndim != 2:
                    raise ValueError(f"{npz_path}: X_shared_bw must have shape [N, L]")
                if x_len.shape != (x_bw.shape[0],):
                    raise ValueError(f"{npz_path}: X_len must have shape [N]")
                if y_target.shape != (x_bw.shape[0],):
                    raise ValueError(f"{npz_path}: {target_key} must have shape [N]")

                keep = np.isfinite(y_target)
                if float(max_env_error_rate) >= 0.0:
                    denom = np.maximum(replay_num_steps, 1.0)
                    keep &= (env_error_steps / denom) <= float(max_env_error_rate) + 1e-9

                if not np.all(keep):
                    x_bw = x_bw[keep]
                    x_loss = x_loss[keep]
                    x_len = x_len[keep]
                    y_target = y_target[keep]
                    trace_name = trace_name[keep]

                if x_bw.shape[0] == 0:
                    continue

                loaded.append(
                    {
                        "X_shared_bw": x_bw,
                        "X_shared_loss": x_loss,
                        "X_len": x_len,
                        "Y_target": y_target,
                        "trace_name": trace_name.astype(str),
                    }
                )
                max_len = max(max_len, int(x_bw.shape[1]))
                total_rows += int(x_bw.shape[0])

        if not loaded or total_rows <= 0:
            raise ValueError("No finite training rows found in the provided dataset(s)")

        self.X_shared_bw = np.zeros((total_rows, max_len), dtype=np.float32)
        self.X_shared_loss = np.zeros((total_rows, max_len), dtype=np.float32)
        self.X_len = np.zeros((total_rows,), dtype=np.int64)
        self.Y_target = np.zeros((total_rows,), dtype=np.float32)
        self.trace_name = np.empty((total_rows,), dtype="<U256")

        cursor = 0
        for chunk in loaded:
            count = int(chunk["X_shared_bw"].shape[0])
            width = int(chunk["X_shared_bw"].shape[1])
            self.X_shared_bw[cursor:cursor + count, :width] = chunk["X_shared_bw"]
            self.X_shared_loss[cursor:cursor + count, :width] = chunk["X_shared_loss"]
            self.X_len[cursor:cursor + count] = chunk["X_len"]
            self.Y_target[cursor:cursor + count] = chunk["Y_target"]
            self.trace_name[cursor:cursor + count] = chunk["trace_name"]
            cursor += count

        self.N = total_rows
        self.max_len = max_len
        self.target_key = str(target_key)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "shared_bw": torch.from_numpy(self.X_shared_bw[idx]).float(),
            "shared_loss": torch.from_numpy(self.X_shared_loss[idx]).float(),
            "length": torch.tensor(self.X_len[idx]).long(),
            "target": torch.tensor(self.Y_target[idx]).float(),
            "trace_name": str(self.trace_name[idx]),
        }


def make_splits(n: int, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least two samples to create train/validation splits")
    rng = np.random.RandomState(int(seed))
    index = np.arange(n, dtype=np.int64)
    rng.shuffle(index)
    val_count = int(round(float(n) * float(val_split)))
    val_count = min(max(val_count, 1), n - 1)
    return index[val_count:], index[:val_count]


def batch_to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shared_bw = batch["shared_bw"].to(device)  # type: ignore[union-attr]
    shared_loss = batch["shared_loss"].to(device)  # type: ignore[union-attr]
    length = batch["length"].to(device)  # type: ignore[union-attr]
    target = batch["target"].to(device)  # type: ignore[union-attr]
    return shared_bw, shared_loss, length, target


def evaluate_model(
    model: CleanTraceSurrogateRegressor,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    count = 0

    for batch in loader:
        shared_bw, shared_loss, length, target = batch_to_device(batch, device)
        pred = model(shared_bw, shared_loss, length)
        loss = loss_fn(pred, target)

        if train:
            if optimizer is None:
                raise ValueError("optimizer is required when train=True")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_abs += float(torch.abs(pred.detach() - target).sum().cpu().item())
        total_sq += float(torch.square(pred.detach() - target).sum().cpu().item())
        count += batch_size

    denom = max(count, 1)
    return {
        "loss": total_loss / denom,
        "mae": total_abs / denom,
        "rmse": math.sqrt(total_sq / denom),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a clean-trace surrogate for Sage trace PGD.")
    parser.add_argument("--data", required=True, help="Path to a collected .npz dataset or a directory of .npz files")
    parser.add_argument("--out", required=True, help="Checkpoint output path")
    parser.add_argument("--target-key", type=str, default="Y_episode_total_reward")
    parser.add_argument("--max-env-error-rate", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset = CleanTraceSurrogateDataset(
        str(args.data),
        target_key=str(args.target_key),
        max_env_error_rate=float(args.max_env_error_rate),
    )
    train_idx, val_idx = make_splits(len(dataset), val_split=float(args.val_split), seed=int(args.seed))
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch), shuffle=False, num_workers=0)

    model = CleanTraceSurrogateRegressor(
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        head_dim=int(args.head_dim),
    ).to(str(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = evaluate_model(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=str(args.device),
            train=True,
            optimizer=optimizer,
        )
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=str(args.device),
            train=False,
        )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae']:.4f} train_rmse={train_metrics['rmse']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.4f} val_rmse={val_metrics['rmse']:.4f}"
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = dict(val_metrics)

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint")

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_kwargs": model.export_config(),
            "best_val_metrics": best_metrics,
            "train_size": int(len(train_ds)),
            "val_size": int(len(val_ds)),
            "max_seq_len": int(dataset.max_len),
            "target_key": str(args.target_key),
            "target_mean": float(np.mean(dataset.Y_target)),
            "target_std": float(np.std(dataset.Y_target)),
            "train_args": vars(args),
        },
        out_path,
    )
    print(f"[SAVE] checkpoint -> {out_path}  (best_val_mae={best_val_mae:.4f})")


if __name__ == "__main__":
    main()
