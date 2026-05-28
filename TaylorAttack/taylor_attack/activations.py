from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


def stable_sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.where(
        x_arr >= 0.0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr)),
    )
    return float(out) if np.isscalar(x) else out


def silu_np(x: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x, dtype=np.float64) * stable_sigmoid_np(x)


def silu_d1_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    s = stable_sigmoid_np(x_arr)
    return s + x_arr * s * (1.0 - s)


def silu_d2_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    s = stable_sigmoid_np(x_arr)
    ds = s * (1.0 - s)
    return 2.0 * ds + x_arr * ds * (1.0 - 2.0 * s)


def gelu_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    phi = 0.5 * (1.0 + np.vectorize(math.erf)(x_arr / math.sqrt(2.0)))
    out = x_arr * phi
    return float(out) if np.isscalar(x) else out


def gelu_d1_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    phi = 0.5 * (1.0 + np.vectorize(math.erf)(x_arr / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * x_arr * x_arr) / math.sqrt(2.0 * math.pi)
    out = phi + x_arr * pdf
    return float(out) if np.isscalar(x) else out


def gelu_d2_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    pdf = np.exp(-0.5 * x_arr * x_arr) / math.sqrt(2.0 * math.pi)
    out = (2.0 - x_arr * x_arr) * pdf
    return float(out) if np.isscalar(x) else out


def gelu_tanh_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    inner = math.sqrt(2.0 / math.pi) * (x_arr + 0.044715 * x_arr**3)
    out = 0.5 * x_arr * (1.0 + np.tanh(inner))
    return float(out) if np.isscalar(x) else out


def gelu_tanh_d1_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    coeff = math.sqrt(2.0 / math.pi)
    inner = coeff * (x_arr + 0.044715 * x_arr**3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    inner_d1 = coeff * (1.0 + 3.0 * 0.044715 * x_arr * x_arr)
    out = 0.5 * (1.0 + tanh_inner) + 0.5 * x_arr * sech2 * inner_d1
    return float(out) if np.isscalar(x) else out


def gelu_tanh_d2_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    coeff = math.sqrt(2.0 / math.pi)
    inner = coeff * (x_arr + 0.044715 * x_arr**3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    inner_d1 = coeff * (1.0 + 3.0 * 0.044715 * x_arr * x_arr)
    inner_d2 = coeff * (6.0 * 0.044715 * x_arr)
    out = sech2 * inner_d1 + 0.5 * x_arr * sech2 * (
        inner_d2 - 2.0 * tanh_inner * inner_d1 * inner_d1
    )
    return float(out) if np.isscalar(x) else out


def silu_derivatives_torch(
    x: torch.Tensor, grad_order: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return SiLU derivatives of orders 1..grad_order.

    This mirrors Taylor-Unswift's recurrence while avoiding scipy at runtime.
    """
    if grad_order < 1:
        raise ValueError("grad_order must be >= 1")
    calc_dtype = dtype or x.dtype
    x_calc = x.to(calc_dtype)
    sigmoid_x = torch.sigmoid(x_calc)
    sigmoid_derivs = torch.zeros(
        x_calc.numel(), grad_order, device=x.device, dtype=calc_dtype
    )
    silu_derivs = torch.zeros_like(sigmoid_derivs)

    flat_x = x_calc.reshape(-1)
    flat_sigmoid = sigmoid_x.reshape(-1)
    sigmoid_derivs[:, 0] = flat_sigmoid * (1.0 - flat_sigmoid)
    silu_derivs[:, 0] = flat_x * sigmoid_derivs[:, 0] + flat_sigmoid

    for grad_idx in range(1, grad_order):
        if grad_idx == 1:
            sigmoid_derivs[:, grad_idx] = (
                sigmoid_derivs[:, 0] - 2.0 * flat_sigmoid * sigmoid_derivs[:, 0]
            )
        else:
            acc = torch.zeros_like(sigmoid_derivs[:, grad_idx])
            for idx in range(grad_idx + 1):
                coeff = math.comb(grad_idx, idx)
                if idx == 0:
                    term = -coeff * flat_sigmoid * sigmoid_derivs[:, grad_idx - 1]
                elif idx == grad_idx:
                    term = coeff * (
                        sigmoid_derivs[:, grad_idx - 1]
                        - sigmoid_derivs[:, grad_idx - 1] * flat_sigmoid
                    )
                else:
                    term = (
                        -coeff
                        * sigmoid_derivs[:, idx - 1]
                        * sigmoid_derivs[:, grad_idx - idx - 1]
                    )
                acc = acc + term
            sigmoid_derivs[:, grad_idx] = acc

        silu_derivs[:, grad_idx] = (
            flat_x * sigmoid_derivs[:, grad_idx]
            + (grad_idx + 1) * sigmoid_derivs[:, grad_idx - 1]
        )

    return silu_derivs.reshape(*x.shape, grad_order).detach()


def gelu_derivatives_torch(
    x: torch.Tensor, grad_order: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return exact-GELU derivatives of orders 1..grad_order."""
    if grad_order < 1:
        raise ValueError("grad_order must be >= 1")
    calc_dtype = dtype or x.dtype
    x_calc = x.to(calc_dtype).reshape(-1)
    pdf0 = torch.exp(-0.5 * x_calc * x_calc) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(x_calc / math.sqrt(2.0)))

    gaussian_derivs = torch.zeros(
        x_calc.numel(), grad_order, device=x.device, dtype=calc_dtype
    )
    gelu_derivs = torch.zeros_like(gaussian_derivs)

    if grad_order >= 1:
        gaussian_derivs[:, 0] = -x_calc * pdf0
        gelu_derivs[:, 0] = cdf + x_calc * pdf0

    for grad_idx in range(1, grad_order):
        if grad_idx == 1:
            gaussian_derivs[:, grad_idx] = -x_calc * gaussian_derivs[:, 0] - pdf0
            gelu_derivs[:, grad_idx] = x_calc * gaussian_derivs[:, 0] + 2 * pdf0
        else:
            gaussian_derivs[:, grad_idx] = (
                -x_calc * gaussian_derivs[:, grad_idx - 1]
                - grad_idx * gaussian_derivs[:, grad_idx - 2]
            )
            gelu_derivs[:, grad_idx] = (
                x_calc * gaussian_derivs[:, grad_idx - 1]
                + (grad_idx + 1) * gaussian_derivs[:, grad_idx - 2]
            )

    return gelu_derivs.reshape(*x.shape, grad_order).detach()


def _autograd_elementwise_derivatives(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    grad_order: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if grad_order < 1:
        raise ValueError("grad_order must be >= 1")
    calc_dtype = dtype or x.dtype
    x_calc = x.detach().to(calc_dtype).requires_grad_(True)
    current = fn(x_calc)
    derivs = []
    for idx in range(grad_order):
        grad = torch.autograd.grad(
            current.sum(),
            x_calc,
            create_graph=idx < grad_order - 1,
            retain_graph=idx < grad_order - 1,
        )[0]
        derivs.append(grad)
        current = grad
    return torch.stack(derivs, dim=-1).detach()


def gelu_tanh_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def gelu_tanh_derivatives_torch(
    x: torch.Tensor, grad_order: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    return _autograd_elementwise_derivatives(gelu_tanh_torch, x, grad_order, dtype)


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    torch_fn: Callable[[torch.Tensor], torch.Tensor]
    torch_derivatives: Callable[[torch.Tensor, int, torch.dtype | None], torch.Tensor]
    np_fn: Callable[[np.ndarray | float], np.ndarray | float]
    np_d1: Callable[[np.ndarray | float], np.ndarray | float]
    np_d2: Callable[[np.ndarray | float], np.ndarray | float]


def get_activation(name: str) -> ActivationSpec:
    normalized = name.lower()
    if normalized in {"silu", "swish"}:
        return ActivationSpec(
            name="silu",
            torch_fn=torch.nn.functional.silu,
            torch_derivatives=silu_derivatives_torch,
            np_fn=silu_np,
            np_d1=silu_d1_np,
            np_d2=silu_d2_np,
        )
    if normalized in {"gelu", "geglu"}:
        return ActivationSpec(
            name="gelu",
            torch_fn=torch.nn.functional.gelu,
            torch_derivatives=gelu_derivatives_torch,
            np_fn=gelu_np,
            np_d1=gelu_d1_np,
            np_d2=gelu_d2_np,
        )
    if normalized in {
        "gelu_pytorch_tanh",
        "gelu_tanh",
        "approximate_gelu",
        "tanh_gelu",
    }:
        return ActivationSpec(
            name="gelu_pytorch_tanh",
            torch_fn=gelu_tanh_torch,
            torch_derivatives=gelu_tanh_derivatives_torch,
            np_fn=gelu_tanh_np,
            np_d1=gelu_tanh_d1_np,
            np_d2=gelu_tanh_d2_np,
        )
    raise ValueError(
        f"Unsupported activation {name!r}. Expected silu, gelu, or gelu_pytorch_tanh."
    )


def infer_activation_name(module: torch.nn.Module) -> str:
    act = getattr(module, "act_fn", None)
    if act is None:
        return "silu"
    text = f"{act.__class__.__name__} {act}".lower()
    if "gelu" in text and "tanh" in text:
        return "gelu_pytorch_tanh"
    if "gelu" in text:
        return "gelu"
    if "silu" in text or "swish" in text:
        return "silu"
    raise ValueError(f"Could not infer activation from {act!r}")
