from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .activations import infer_activation_name


@dataclass(frozen=True)
class MLPAdapter:
    module: torch.nn.Module
    gate_proj: torch.nn.Linear
    up_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    activation: str

    @property
    def hidden_size(self) -> int:
        return int(self.gate_proj.weight.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate_proj.weight.shape[0])


def adapt_mlp(module: torch.nn.Module, activation: str | None = None) -> MLPAdapter:
    missing = [
        name
        for name in ("gate_proj", "up_proj", "down_proj")
        if not hasattr(module, name)
    ]
    if missing:
        raise ValueError(
            f"Unsupported MLP module {module.__class__.__name__}; missing {missing}"
        )

    gate = getattr(module, "gate_proj")
    up = getattr(module, "up_proj")
    down = getattr(module, "down_proj")
    for name, layer in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError(f"{name} must be torch.nn.Linear, got {type(layer)!r}")
        if layer.bias is not None:
            raise ValueError(
                "This reproduction path follows Taylor-Unswift Llama/Qwen/Gemma "
                f"MLPs and does not support biased {name} layers."
            )

    inferred = activation or infer_activation_name(module)
    return MLPAdapter(module=module, gate_proj=gate, up_proj=up, down_proj=down, activation=inferred)


def get_transformer_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "decoder", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("text_model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            return node
    raise ValueError("Could not locate transformer layers on this model")
