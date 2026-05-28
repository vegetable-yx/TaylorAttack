from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .adapters import adapt_mlp, get_transformer_layers
from .config import ensure_dir


@torch.no_grad()
def collect_mlp_stats(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_tokens: int = 20_000,
    sequence_length: int = 4096,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    layers = list(get_transformer_layers(model))
    adapters = [adapt_mlp(layer.mlp) for layer in layers]
    stats: dict[int, dict[str, torch.Tensor | int]] = {}
    for idx, adapter in enumerate(adapters):
        size = adapter.intermediate_size
        weight = adapter.gate_proj.weight
        stats[idx] = {
            "count": 0,
            "sum": torch.zeros(size, device=weight.device, dtype=torch.float64),
            "sum_sq": torch.zeros(size, device=weight.device, dtype=torch.float64),
            "max": torch.full((size,), -torch.inf, device=weight.device, dtype=torch.float64),
            "min": torch.full((size,), torch.inf, device=weight.device, dtype=torch.float64),
        }

    hooks = []

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            hidden = inputs[0]
            adapter = adapters[layer_idx]
            gate = hidden @ adapter.gate_proj.weight.T
            flat = gate.reshape(-1, gate.shape[-1]).detach().to(torch.float64)
            item = stats[layer_idx]
            item["count"] = int(item["count"]) + flat.shape[0]
            item["sum"] = item["sum"] + flat.sum(dim=0)
            item["sum_sq"] = item["sum_sq"] + (flat * flat).sum(dim=0)
            item["max"] = torch.maximum(item["max"], flat.max(dim=0).values)
            item["min"] = torch.minimum(item["min"], flat.min(dim=0).values)

        return hook

    for idx, layer in enumerate(layers):
        hooks.append(layer.mlp.register_forward_hook(make_hook(idx)))

    seen_tokens = 0
    try:
        for text in texts:
            if not text or not text.strip():
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length,
            )
            input_ids = encoded["input_ids"].to(device)
            if input_ids.numel() == 0:
                continue
            seen_tokens += int(input_ids.numel())
            model(input_ids=input_ids)
            if seen_tokens >= max_tokens:
                break
    finally:
        for hook in hooks:
            hook.remove()

    output: dict[str, torch.Tensor] = {}
    for idx, item in stats.items():
        count = max(int(item["count"]), 1)
        mean = item["sum"] / count
        var = item["sum_sq"] / count - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        prefix = f"model.layer[{idx}].mlp.hidden_states_intermediate"
        output[f"{prefix}_mean"] = mean.cpu().to(torch.bfloat16)
        output[f"{prefix}_std"] = std.cpu().to(torch.bfloat16)
        output[f"{prefix}_max"] = item["max"].cpu().to(torch.bfloat16)
        output[f"{prefix}_min"] = item["min"].cpu().to(torch.bfloat16)
    return output


def load_texts_from_dataset(calibration_config: dict[str, Any]) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to run calibration") from exc

    dataset_id = calibration_config["dataset_id"]
    dataset_config = calibration_config.get("dataset_config")
    split = calibration_config.get("split", "test")
    text_field = calibration_config.get("text_field", "text")
    ds = load_dataset(dataset_id, dataset_config, split=split)
    return [row[text_field] for row in ds if row.get(text_field)]


def save_stats_with_metadata(
    stats: dict[str, torch.Tensor], output_path: str | Path, metadata: dict[str, Any]
) -> None:
    output = Path(output_path)
    ensure_dir(output.parent)
    payload: dict[str, Any] = dict(stats)
    payload["_metadata"] = json.dumps(metadata, sort_keys=True)
    torch.save(payload, output)


def layer_stats_from_buffer(buffer: dict[str, Any], layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layer[{layer_idx}].mlp.hidden_states_intermediate"
    return {
        "mean": buffer[f"{prefix}_mean"],
        "std": buffer[f"{prefix}_std"],
        "max": buffer[f"{prefix}_max"],
        "min": buffer[f"{prefix}_min"],
    }
