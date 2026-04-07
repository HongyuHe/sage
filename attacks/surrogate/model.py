from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CleanTraceSurrogateRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 6,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.model_kwargs = {
            "input_dim": int(input_dim),
            "embed_dim": int(embed_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "head_dim": int(head_dim),
        }
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
            nn.ReLU(),
        )
        self.encoder = nn.GRU(
            input_size=int(embed_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        fused_dim = int(hidden_dim) * 6
        self.head = nn.Sequential(
            nn.Linear(fused_dim, int(head_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_dim), 1),
        )

    def export_config(self) -> dict[str, int | float]:
        return dict(self.model_kwargs)

    def _build_features(
        self,
        shared_bw: torch.Tensor,
        shared_loss: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if shared_bw.shape != shared_loss.shape:
            raise ValueError("shared_bw and shared_loss must have matching shape")

        shared_bw = torch.clamp(shared_bw, min=0.0)
        shared_loss = torch.clamp(shared_loss, min=0.0, max=1.0)
        bw_delta = torch.zeros_like(shared_bw)
        loss_delta = torch.zeros_like(shared_loss)
        if shared_bw.shape[1] > 1:
            bw_delta[:, 1:] = torch.abs(shared_bw[:, 1:] - shared_bw[:, :-1])
            loss_delta[:, 1:] = torch.abs(shared_loss[:, 1:] - shared_loss[:, :-1])
        effective_payload = shared_bw * (1.0 - shared_loss)
        position = torch.arange(shared_bw.shape[1], device=shared_bw.device, dtype=shared_bw.dtype).unsqueeze(0)
        lengths_f = lengths.to(device=shared_bw.device, dtype=shared_bw.dtype).clamp(min=1.0)
        relative_index = position / torch.clamp(lengths_f.unsqueeze(1) - 1.0, min=1.0)
        return torch.stack(
            [
                torch.log1p(shared_bw),
                shared_loss,
                torch.log1p(torch.clamp(effective_payload, min=0.0)),
                torch.log1p(torch.clamp(bw_delta, min=0.0)),
                loss_delta,
                relative_index,
            ],
            dim=-1,
        )

    def forward(
        self,
        shared_bw: torch.Tensor,
        shared_loss: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if shared_bw.ndim != 2 or shared_loss.ndim != 2:
            raise ValueError("shared_bw and shared_loss must have shape [batch, seq_len]")
        if shared_bw.shape != shared_loss.shape:
            raise ValueError("shared_bw and shared_loss must have matching shape")
        if lengths.ndim != 1 or lengths.shape[0] != shared_bw.shape[0]:
            raise ValueError("lengths must have shape [batch]")

        lengths = lengths.to(device=shared_bw.device, dtype=torch.long).clamp(min=1, max=shared_bw.shape[1])
        mask = torch.arange(shared_bw.shape[1], device=shared_bw.device).unsqueeze(0) < lengths.unsqueeze(1)
        features = self._build_features(shared_bw, shared_loss, lengths)
        features = self.input_proj(features)

        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.encoder(packed)
        encoded, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=shared_bw.shape[1])

        mask_f = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        mean_pool = (encoded * mask_f).sum(dim=1) / lengths.unsqueeze(1).to(dtype=encoded.dtype)
        max_pool = encoded.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)

        if self.encoder.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]

        fused = torch.cat([final_hidden, mean_pool, max_pool], dim=1)
        return self.head(fused).squeeze(1)


def extract_state_dict(ckpt: dict) -> dict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def get_model_kwargs_from_checkpoint(ckpt: dict) -> dict:
    if isinstance(ckpt, dict) and "model_kwargs" in ckpt:
        return dict(ckpt["model_kwargs"])
    return {}


def load_clean_trace_surrogate_from_checkpoint(
    ckpt_path: str,
    *,
    device: str = "cpu",
    eval_mode: bool = True,
) -> tuple[CleanTraceSurrogateRegressor, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CleanTraceSurrogateRegressor(**get_model_kwargs_from_checkpoint(ckpt)).to(device)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    if eval_mode:
        model.eval()
    return model, ckpt
