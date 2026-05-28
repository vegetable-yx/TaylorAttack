from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .activations import get_activation
from .adapters import MLPAdapter, adapt_mlp


def dim_selection(
    max_min_gap: torch.Tensor, select_dim_num: int, max_gap: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror Taylor-Unswift's dimension selection."""
    select_dim_num = min(int(select_dim_num), int(max_min_gap.numel()))
    max_min_gap_sort, sort_index = torch.sort(max_min_gap)
    mask = max_min_gap_sort < max_gap
    if int(mask.sum()) >= select_dim_num:
        valid_index = sort_index[mask]
        approx_dim = valid_index[-select_dim_num:]
        ffn_dim = torch.cat([valid_index[:-select_dim_num], sort_index[~mask]], dim=0)
    else:
        approx_dim = sort_index[:select_dim_num]
        ffn_dim = sort_index[select_dim_num:]
    return approx_dim, ffn_dim


@dataclass(frozen=True)
class ProtectionConfig:
    select_dim: int = 10000
    select_dim_ratio: float | None = None
    grad_order: int = 6
    grad_order_min: int = 8
    delta_hidden_state_thd: float = 2.5
    max_gap: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict) -> "ProtectionConfig":
        ratio = data.get("select_dim_ratio", cls.select_dim_ratio)
        select_dim_ratio = None if ratio in (None, "") else float(ratio)
        return cls(
            select_dim=int(data.get("select_dim", cls.select_dim)),
            select_dim_ratio=select_dim_ratio,
            grad_order=int(data.get("grad_order", cls.grad_order)),
            grad_order_min=int(data.get("grad_order_min", cls.grad_order_min)),
            delta_hidden_state_thd=float(
                data.get("delta_hidden_state_thd", cls.delta_hidden_state_thd)
            ),
            max_gap=float(data.get("max_gap", cls.max_gap)),
        )

    def resolved_select_dim(self, intermediate_size: int) -> int:
        if self.select_dim_ratio is not None:
            if not 0.0 < self.select_dim_ratio <= 1.0:
                raise ValueError("select_dim_ratio must be in (0, 1]")
            return min(
                intermediate_size,
                max(1, int(round(intermediate_size * self.select_dim_ratio))),
            )
        return min(intermediate_size, max(1, int(self.select_dim)))


def normalize_stats(stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    aliases = {
        "mean": ("mean", "hidden_states_intermediate_mean"),
        "std": ("std", "hidden_states_intermediate_std"),
        "max": ("max", "hidden_states_intermediate_max"),
        "min": ("min", "hidden_states_intermediate_min"),
    }
    out: dict[str, torch.Tensor] = {}
    for key, names in aliases.items():
        for name in names:
            if name in stats:
                out[key] = stats[name]
                break
        if key not in out:
            raise KeyError(f"Missing calibration statistic {key!r}")
    return out


class TaylorProtectedMLP(nn.Module):
    """Taylor-Unswift style protected SwiGLU/GeGLU MLP.

    The public tensors follow the vulnerable release order used by
    Taylor-Unswift: zeroth-order output is left unperturbed while odd-order
    derivative terms and gate/local point are perturbed.
    """

    def __init__(
        self,
        adapter: MLPAdapter,
        stats: dict[str, torch.Tensor],
        config: ProtectionConfig,
        activation: str | None = None,
        layer_index: int = 0,
    ) -> None:
        super().__init__()
        stats = normalize_stats(stats)
        self.layer_index = layer_index
        self.activation_name = activation or adapter.activation
        self.activation = get_activation(self.activation_name)
        self.grad_order = int(config.grad_order)
        self.grad_order_min = int(config.grad_order_min)
        self.delta_hidden_state_thd = float(config.delta_hidden_state_thd)
        self.hidden_size = adapter.hidden_size
        self.intermediate_size = adapter.intermediate_size

        device = adapter.gate_proj.weight.device
        dtype = adapter.gate_proj.weight.dtype
        taylor_dtype = torch.float32
        hs_max = stats["max"].to(device=device, dtype=dtype)
        hs_min = stats["min"].to(device=device, dtype=dtype)
        max_min_gap = hs_max - hs_min
        all_local_point = ((hs_min + hs_max) / 2.0).to(taylor_dtype)
        select_dim = config.resolved_select_dim(adapter.intermediate_size)
        sampled_dim, ffn_dim = dim_selection(max_min_gap, select_dim, max_gap=config.max_gap)
        sampled_dim = sampled_dim.to(device=device)
        ffn_dim = ffn_dim.to(device=device)

        self.register_buffer("sampled_dim", sampled_dim, persistent=True)
        self.register_buffer("ffn_dim", ffn_dim, persistent=True)

        local_point = all_local_point[sampled_dim].to(taylor_dtype)
        gate_sampled = adapter.gate_proj.weight.data[sampled_dim, :].detach().clone()
        up_sampled = adapter.up_proj.weight.data[sampled_dim, :].detach().clone()
        down_sampled = (
            adapter.down_proj.weight.data[:, sampled_dim].detach().clone().to(taylor_dtype)
        )

        self.register_buffer(
            "ffn_gate_proj_weight",
            adapter.gate_proj.weight.data[ffn_dim, :].detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "ffn_up_proj_weight",
            adapter.up_proj.weight.data[ffn_dim, :].detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "ffn_down_proj_weight",
            adapter.down_proj.weight.data[:, ffn_dim].detach().clone(),
            persistent=True,
        )

        local_approx_output = self.activation.torch_fn(local_point).unsqueeze(0) * down_sampled
        grad_matrix = self.activation.torch_derivatives(
            local_point, self.grad_order, taylor_dtype
        )
        fuse_weight = down_sampled.unsqueeze(-1) * grad_matrix.unsqueeze(0)

        perturb = (
            torch.randint(0, 2, size=(len(sampled_dim),), device=device, dtype=torch.int8)
            .to(dtype)
            .mul(2)
            .sub(1)
        )
        grad_orders = torch.arange(1, self.grad_order + 1, device=device, dtype=torch.int64)
        perturb_powers = torch.cat(
            [perturb.unsqueeze(-1) ** int(order.item()) for order in grad_orders], dim=-1
        )

        self.register_buffer(
            "gate_proj_weight", gate_sampled * perturb.unsqueeze(-1), persistent=True
        )
        self.register_buffer("up_proj_weight", up_sampled, persistent=True)
        self.register_buffer("local_point", local_point * perturb, persistent=True)
        self.register_buffer("local_approx_output", local_approx_output, persistent=True)
        self.register_buffer("fuse_weight", fuse_weight * perturb_powers, persistent=True)
        self.register_buffer(
            "grad_order_buf",
            grad_orders.to(device=device, dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "factorials",
            torch.tensor(
                [math.factorial(idx) for idx in range(1, self.grad_order + 1)],
                device=device,
                dtype=dtype,
            ),
            persistent=True,
        )

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        stats: dict[str, torch.Tensor],
        config: ProtectionConfig | dict,
        activation: str | None = None,
        layer_index: int = 0,
    ) -> "TaylorProtectedMLP":
        cfg = config if isinstance(config, ProtectionConfig) else ProtectionConfig.from_mapping(config)
        adapter = adapt_mlp(mlp, activation=activation)
        return cls(adapter, stats, cfg, activation=activation, layer_index=layer_index)

    def sampled_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        approx_up = hidden_states @ self.up_proj_weight.T
        gate = hidden_states @ self.gate_proj_weight.T
        delta = gate - self.local_point

        approx_output = approx_up @ self.local_approx_output.T
        over_threshold = delta.abs() > self.delta_hidden_state_thd
        for idx in range(self.grad_order):
            order = idx + 1
            power = (delta ** order) / self.factorials[idx]
            if order > self.grad_order_min:
                power = power * (~over_threshold).to(power.dtype)
            approx_output = approx_output + (power * approx_up) @ self.fuse_weight[:, :, idx].T
        return approx_output

    def ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_output = hidden_states @ self.ffn_up_proj_weight.T
        gate_output = hidden_states @ self.ffn_gate_proj_weight.T
        return (
            self.activation.torch_fn(gate_output) * up_output
        ) @ self.ffn_down_proj_weight.T

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.sampled_forward(hidden_states) + self.ffn_forward(hidden_states)
