from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .activations import get_activation
from .adapters import adapt_mlp
from .config import ensure_dir
from .protection import TaylorProtectedMLP


class RatioInverter:
    def __init__(
        self,
        activation: str,
        z_min: float = -10.0,
        z_max: float = 10.0,
        grid_size: int = 200_000,
    ) -> None:
        self.spec = get_activation(activation)
        self.z_grid = np.linspace(z_min, z_max, grid_size, dtype=np.float64)
        s0 = np.asarray(self.spec.np_fn(self.z_grid), dtype=np.float64)
        s2 = np.asarray(self.spec.np_d2(self.z_grid), dtype=np.float64)
        safe_s0 = np.where(np.abs(s0) > 1e-8, s0, np.nan)
        self.ratio_grid = s2 / safe_s0
        self.valid_grid = np.isfinite(self.ratio_grid)

    def invert(self, ratio: float) -> float:
        if not np.isfinite(ratio):
            return 0.0
        residual = np.abs(self.ratio_grid[self.valid_grid] - ratio)
        best_valid_index = int(np.argmin(residual))
        valid_z = self.z_grid[self.valid_grid]
        z = float(valid_z[best_valid_index])
        return self.refine(z, ratio)

    def best_from_candidates(
        self, ratio: float, candidates: list[float], refine: bool = False
    ) -> float:
        finite_candidates = [float(val) for val in candidates if np.isfinite(val)]
        if not finite_candidates:
            return self.invert(ratio)
        scored = [(abs(self._equation(val, ratio)), val) for val in finite_candidates]
        scored.sort(key=lambda item: item[0])
        return self.refine(scored[0][1], ratio) if refine else float(scored[0][1])

    def _equation(self, z: float, ratio: float) -> float:
        return float(self.spec.np_d2(z) - ratio * self.spec.np_fn(z))

    def refine(self, z: float, ratio: float) -> float:
        # Optional scipy refinement when available; the dense grid is the fallback.
        try:
            from scipy.optimize import fsolve  # type: ignore

            result, _, ier, _ = fsolve(
                lambda val: self._equation(float(np.asarray(val).flat[0]), ratio),
                z,
                full_output=True,
                xtol=1e-10,
            )
            if ier == 1 and np.isfinite(result[0]):
                return float(result[0])
        except Exception:
            pass

        current = z
        for _ in range(12):
            f_val = self._equation(current, ratio)
            h = 1e-4 * max(1.0, abs(current))
            deriv = (self._equation(current + h, ratio) - self._equation(current - h, ratio)) / (2 * h)
            if not np.isfinite(deriv) or abs(deriv) < 1e-12:
                break
            nxt = current - f_val / deriv
            if not np.isfinite(nxt) or abs(nxt) > 20:
                break
            if abs(nxt - current) < 1e-9:
                return float(nxt)
            current = float(nxt)
        return float(current)


@dataclass
class RecoveryResult:
    W_gate_sampled: np.ndarray
    W_up_sampled: np.ndarray
    W_down_sampled: np.ndarray
    z0: np.ndarray
    perturb: np.ndarray
    W_gate_ffn: np.ndarray
    W_up_ffn: np.ndarray
    W_down_ffn: np.ndarray
    sampled_dim: np.ndarray
    ffn_dim: np.ndarray
    elapsed_sec: float
    failed_dims: int
    ls_residual: np.ndarray


def _activation_coefficients(
    activation: str,
    z_values: np.ndarray,
    perturb: float,
    grad_order: int,
) -> np.ndarray:
    spec = get_activation(activation)
    # Match TaylorProtectedMLP's public Taylor buffers, which are generated in fp32.
    z_tensor = torch.as_tensor(z_values, dtype=torch.float32)
    zero_order = spec.torch_fn(z_tensor).detach().cpu().numpy()
    derivatives = (
        spec.torch_derivatives(z_tensor, grad_order, torch.float32).detach().cpu().numpy()
    )
    powers = np.asarray(
        [perturb ** order for order in range(1, grad_order + 1)], dtype=np.float64
    )
    return np.concatenate([zero_order[:, np.newaxis], derivatives * powers], axis=1)


def _fit_down_column(
    observations: np.ndarray,
    coeff: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, float]:
    coeff_norm_sq = float(np.dot(coeff, coeff))
    obs_norm_sq = float(np.sum(observations * observations))
    if coeff_norm_sq <= eps or not np.isfinite(coeff_norm_sq):
        return np.zeros(observations.shape[0], dtype=np.float64), float("inf")
    projection = observations @ coeff
    weights = projection / coeff_norm_sq
    explained = float(np.dot(projection, projection) / coeff_norm_sq)
    residual = max(obs_norm_sq - explained, 0.0) / max(obs_norm_sq, eps)
    return weights, residual


def attack_protected_mlp(
    protected_mlp: TaylorProtectedMLP,
    inverter: RatioInverter | None = None,
    eps: float = 1e-7,
) -> RecoveryResult:
    spec = get_activation(protected_mlp.activation_name)
    inverter = inverter or RatioInverter(protected_mlp.activation_name)
    start = time.time()

    local_approx = protected_mlp.local_approx_output.float().cpu().numpy()
    fuse_w = protected_mlp.fuse_weight.float().cpu().numpy()
    gate_perturbed = protected_mlp.gate_proj_weight.float().cpu().numpy()
    local_point_public = protected_mlp.local_point.float().cpu().numpy()
    up_raw = protected_mlp.up_proj_weight.float().cpu().numpy()

    hidden_size, sampled_size = local_approx.shape
    grad_order = int(fuse_w.shape[2])
    if grad_order < 1:
        raise ValueError("grad_order must be >= 1 for the algebraic attack")

    coeff_by_perturb = {
        -1.0: _activation_coefficients(
            protected_mlp.activation_name,
            -local_point_public.astype(np.float64),
            -1.0,
            grad_order,
        ),
        1.0: _activation_coefficients(
            protected_mlp.activation_name,
            local_point_public.astype(np.float64),
            1.0,
            grad_order,
        ),
    }

    z0_rec = np.zeros(sampled_size, dtype=np.float64)
    perturb_rec = np.ones(sampled_size, dtype=np.float64)
    W_down_rec = np.zeros((hidden_size, sampled_size), dtype=np.float64)
    ls_residual = np.full(sampled_size, np.inf, dtype=np.float64)
    failed_dims = 0

    for dim in range(sampled_size):
        observations = np.concatenate(
            [local_approx[:, dim, np.newaxis], fuse_w[:, dim, :]], axis=1
        ).astype(np.float64, copy=False)
        finite_rows = np.isfinite(observations).all(axis=1)
        if int(finite_rows.sum()) < 5:
            failed_dims += 1
            continue
        observations = observations[finite_rows]

        best: tuple[float, float, np.ndarray, float] | None = None
        for perturb in (-1.0, 1.0):
            coeff = coeff_by_perturb[perturb][dim, :]
            weights, residual = _fit_down_column(observations, coeff, eps=eps)
            coeff_norm = float(np.linalg.norm(coeff))
            score = residual + (1e-12 / max(coeff_norm, eps))
            if best is None or score < best[0]:
                best = (score, perturb, weights, residual)

        if best is None or not np.isfinite(best[0]):
            if abs(float(local_point_public[dim])) > 1e-8 and grad_order >= 2:
                denom = local_approx[:, dim]
                valid = np.abs(denom) > eps
                ratio_vals = fuse_w[valid, dim, 1] / denom[valid]
                finite_ratios = ratio_vals[np.isfinite(ratio_vals)]
                ratio = float(np.median(finite_ratios)) if finite_ratios.size else 0.0
                z0 = inverter.best_from_candidates(
                    ratio,
                    [abs(float(local_point_public[dim])), -abs(float(local_point_public[dim]))],
                )
                perturb = 1.0 if z0 * float(local_point_public[dim]) >= 0.0 else -1.0
                coeff = _activation_coefficients(
                    protected_mlp.activation_name,
                    np.asarray([z0], dtype=np.float64),
                    perturb,
                    grad_order,
                )[0]
                weights, residual = _fit_down_column(observations, coeff, eps=eps)
                best = (residual, perturb, weights, residual)
            else:
                failed_dims += 1
                continue

        _, perturb, weights, residual = best
        z0 = perturb * float(local_point_public[dim])
        z0_rec[dim] = z0
        perturb_rec[dim] = perturb
        W_down_rec[finite_rows, dim] = weights
        ls_residual[dim] = residual
        if residual > 1e-4:
            failed_dims += 1

    W_gate_rec = gate_perturbed * perturb_rec[:, np.newaxis]
    elapsed = time.time() - start
    return RecoveryResult(
        W_gate_sampled=W_gate_rec,
        W_up_sampled=up_raw,
        W_down_sampled=W_down_rec,
        z0=z0_rec,
        perturb=perturb_rec,
        W_gate_ffn=protected_mlp.ffn_gate_proj_weight.float().cpu().numpy(),
        W_up_ffn=protected_mlp.ffn_up_proj_weight.float().cpu().numpy(),
        W_down_ffn=protected_mlp.ffn_down_proj_weight.float().cpu().numpy(),
        sampled_dim=protected_mlp.sampled_dim.cpu().numpy(),
        ffn_dim=protected_mlp.ffn_dim.cpu().numpy(),
        elapsed_sec=elapsed,
        failed_dims=failed_dims,
        ls_residual=ls_residual,
    )


def rel_err(recovered: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(recovered - truth) / (np.linalg.norm(truth) + 1e-12))


def element_success_count(
    recovered: np.ndarray, truth: np.ndarray, threshold: float = 0.01
) -> tuple[int, int, float, float]:
    if recovered.size == 0 or truth.size == 0:
        return 0, int(recovered.size), 0.0, 0.0
    denom = np.maximum(np.abs(truth), 1e-12)
    errors = np.abs(recovered - truth) / denom
    finite = np.isfinite(errors)
    total = int(errors.size)
    success = int(((errors < threshold) & finite).sum())
    return success, total, float(np.mean(errors[finite])), float(np.median(errors[finite]))


def column_error_summary(recovered: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if recovered.ndim != 2 or truth.ndim != 2 or recovered.shape != truth.shape:
        return {}
    denom = np.linalg.norm(truth, axis=0) + 1e-12
    errors = np.linalg.norm(recovered - truth, axis=0) / denom
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {
            "column_relative_error_median": 0.0,
            "column_relative_error_p99": 0.0,
            "column_relative_error_max": 0.0,
            "columns_gt_1pct": 0,
            "columns_gt_10pct": 0,
            "columns_gt_100pct": 0,
        }
    return {
        "column_relative_error_median": float(np.median(finite)),
        "column_relative_error_p90": float(np.quantile(finite, 0.90)),
        "column_relative_error_p99": float(np.quantile(finite, 0.99)),
        "column_relative_error_max": float(np.max(finite)),
        "columns_gt_1pct": int((finite > 0.01).sum()),
        "columns_gt_10pct": int((finite > 0.10).sum()),
        "columns_gt_100pct": int((finite > 1.0).sum()),
    }


def compare_recovery(
    mlp: torch.nn.Module, result: RecoveryResult, threshold: float = 0.01
) -> dict[str, Any]:
    adapter = adapt_mlp(mlp)
    W_gate = adapter.gate_proj.weight.detach().float().cpu().numpy()
    W_up = adapter.up_proj.weight.detach().float().cpu().numpy()
    W_down = adapter.down_proj.weight.detach().float().cpu().numpy()
    sd = result.sampled_dim
    fd = result.ffn_dim

    categories = {
        "gate_sampled": (result.W_gate_sampled, W_gate[sd, :]),
        "up_sampled": (result.W_up_sampled, W_up[sd, :]),
        "down_sampled": (result.W_down_sampled, W_down[:, sd]),
        "gate_ffn": (result.W_gate_ffn, W_gate[fd, :]),
        "up_ffn": (result.W_up_ffn, W_up[fd, :]),
        "down_ffn": (result.W_down_ffn, W_down[:, fd]),
    }
    metrics: dict[str, Any] = {
        "elapsed_sec": result.elapsed_sec,
        "failed_dims": result.failed_dims,
        "categories": {},
        "protected_dims": int(len(sd)),
        "ffn_dims": int(len(fd)),
    }
    total_success = 0
    total_params = 0
    rel_errors = []
    for name, (rec, truth) in categories.items():
        success, total, mean_e, median_e = element_success_count(rec, truth, threshold)
        total_success += success
        total_params += total
        rel_e = rel_err(rec, truth)
        rel_errors.append(rel_e)
        metrics["categories"][name] = {
            "success": success,
            "total": total,
            "success_ratio": success / total if total else 1.0,
            "relative_fro_error": rel_e,
            "mean_element_relative_error": mean_e,
            "median_element_relative_error": median_e,
            **column_error_summary(rec, truth),
        }

    metrics["taylormlp_params"] = total_params
    metrics["recovered_params"] = total_success
    metrics["recovered_ratio"] = total_success / total_params if total_params else 1.0
    metrics["mean_relative_fro_error"] = float(np.mean(rel_errors))
    finite_residuals = result.ls_residual[np.isfinite(result.ls_residual)]
    if finite_residuals.size:
        metrics["ls_residual_mean"] = float(np.mean(finite_residuals))
        metrics["ls_residual_p99"] = float(np.quantile(finite_residuals, 0.99))
        metrics["ls_residual_max"] = float(np.max(finite_residuals))
    return metrics


def apply_recovery_to_mlp(mlp: torch.nn.Module, result: RecoveryResult) -> None:
    adapter = adapt_mlp(mlp)
    gate = adapter.gate_proj.weight.data
    up = adapter.up_proj.weight.data
    down = adapter.down_proj.weight.data
    device = gate.device
    dtype = gate.dtype
    sd = torch.as_tensor(result.sampled_dim, device=device, dtype=torch.long)
    fd = torch.as_tensor(result.ffn_dim, device=device, dtype=torch.long)

    gate[sd, :] = torch.as_tensor(result.W_gate_sampled, device=device, dtype=dtype)
    up[sd, :] = torch.as_tensor(result.W_up_sampled, device=device, dtype=dtype)
    down[:, sd] = torch.as_tensor(result.W_down_sampled, device=device, dtype=dtype)
    gate[fd, :] = torch.as_tensor(result.W_gate_ffn, device=device, dtype=dtype)
    up[fd, :] = torch.as_tensor(result.W_up_ffn, device=device, dtype=dtype)
    down[:, fd] = torch.as_tensor(result.W_down_ffn, device=device, dtype=dtype)


def write_attack_metrics(
    metrics: list[dict[str, Any]], output_dir: str | Path, model_name: str
) -> None:
    out = ensure_dir(output_dir)
    json_path = out / "attack_metrics.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    total_params = sum(int(item["taylormlp_params"]) for item in metrics)
    recovered = sum(int(item["recovered_params"]) for item in metrics)
    elapsed = sum(float(item["elapsed_sec"]) for item in metrics)
    rows = [
        "| Model | # TaylorMLP Parameters | # Recovered Parameters | Recovered Ratio | Run Time |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| {model_name} | {total_params:,} | {recovered:,} | {recovered / total_params:.2%} | {elapsed:.2f}s |",
    ]
    (out / "table1_attack_performance.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
