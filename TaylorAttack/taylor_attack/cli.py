from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .calibration import (
    collect_mlp_stats,
    layer_stats_from_buffer,
    load_texts_from_dataset,
    save_stats_with_metadata,
)
from .config import ensure_dir, find_benchmark_config, find_model_config, load_config
from .download import download_datasets, download_models
from .evaluate import kl_pair, write_kl_table
from .protection import ProtectionConfig, TaylorProtectedMLP
from .recovery import (
    RatioInverter,
    apply_recovery_to_mlp,
    attack_protected_mlp,
    compare_recovery,
    write_attack_metrics,
)


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), torch.bfloat16)


def _load_model_and_tokenizer(model_id: str, dtype: str, device_map: str = "auto"):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to load Hugging Face models") from exc
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    common_kwargs = {
        "torch_dtype": _torch_dtype(dtype),
        "device_map": device_map,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
    except (ValueError, OSError) as causal_exc:
        if AutoModelForImageTextToText is None:
            raise causal_exc
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **common_kwargs)
        except Exception:
            raise causal_exc
    model.eval()
    return model, tokenizer


def _layers(model: torch.nn.Module):
    from .adapters import get_transformer_layers

    return list(get_transformer_layers(model))


def cmd_download(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not args.models and not args.datasets:
        args.models = True
        args.datasets = True
    if args.models:
        download_models(config, cache_dir=args.cache_dir)
    if args.datasets:
        download_datasets(config, args.output)


def cmd_calibrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_cfg = find_model_config(config, args.model)
    run_cfg = config.get("run", {})
    cal_cfg = config.get("calibration", {})
    dtype = args.dtype or run_cfg.get("dtype", "bfloat16")
    model, tokenizer = _load_model_and_tokenizer(model_cfg["id"], dtype=dtype)
    texts = load_texts_from_dataset(cal_cfg)
    stats = collect_mlp_stats(
        model,
        tokenizer,
        texts,
        max_tokens=int(args.max_tokens or cal_cfg.get("max_tokens", 20000)),
        sequence_length=int(cal_cfg.get("sequence_length", 4096)),
        device=args.device,
    )
    output = args.output or (
        Path(run_cfg.get("output_root", "TaylorAttack/results"))
        / "calibration"
        / f"{model_cfg['short_name']}_hidden_states.pt"
    )
    save_stats_with_metadata(
        stats,
        output,
        {
            "model": model_cfg["id"],
            "dtype": dtype,
            "max_tokens": args.max_tokens or cal_cfg.get("max_tokens", 20000),
        },
    )
    print(f"wrote {output}")


def cmd_attack(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_cfg = find_model_config(config, args.model)
    run_cfg = config.get("run", {})
    protection_cfg = ProtectionConfig.from_mapping(config.get("protection", {}))
    dtype = args.dtype or run_cfg.get("dtype", "bfloat16")
    output = ensure_dir(
        args.output
        or (
            Path(run_cfg.get("output_root", "TaylorAttack/results"))
            / f"{model_cfg['short_name']}_attack_{int(time.time())}"
        )
    )

    if not args.stats_file:
        raise ValueError("attack requires --stats-file from the calibrate command")
    stats_buf = torch.load(args.stats_file, map_location="cpu")
    model, _ = _load_model_and_tokenizer(model_cfg["id"], dtype=dtype)
    layers = _layers(model)
    layer_indices = list(range(len(layers)))
    if args.layers is not None:
        layer_indices = layer_indices[: args.layers]
    if args.layer is not None:
        layer_indices = [args.layer]

    inverter = RatioInverter(model_cfg.get("activation", "silu"))
    metrics = []
    for layer_idx in layer_indices:
        print(f"[attack] layer {layer_idx}/{len(layers) - 1}")
        mlp = layers[layer_idx].mlp
        layer_stats = layer_stats_from_buffer(stats_buf, layer_idx)
        protected = TaylorProtectedMLP.from_mlp(
            mlp,
            layer_stats,
            protection_cfg,
            activation=model_cfg.get("activation"),
            layer_index=layer_idx,
        )
        result = attack_protected_mlp(protected, inverter=inverter)
        layer_metrics = compare_recovery(mlp, result)
        layer_metrics["layer_index"] = layer_idx
        metrics.append(layer_metrics)
        if args.save_recovered_model:
            apply_recovery_to_mlp(mlp, result)
        del protected, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_attack_metrics(metrics, output, model_cfg["id"])
    if args.save_recovered_model:
        recovered_dir = output / "recovered_model"
        model.save_pretrained(recovered_dir)
        print(f"wrote recovered model to {recovered_dir}")
    print(f"wrote attack metrics to {output}")


def cmd_kl(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    bench_names = args.benchmark or ["math500", "gpqa_diamond"]
    out = ensure_dir(args.output)
    rows: list[dict[str, Any]] = []
    for name in bench_names:
        bench = find_benchmark_config(config, name)
        values = kl_pair(
            args.original_model,
            args.recovered_model,
            bench,
            out,
            max_samples=args.max_samples,
            dtype=args.dtype,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
            top_logprobs_num=args.top_logprobs,
            max_token_ids=args.max_token_ids,
            sglang_tp_size=args.sglang_tp_size,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_max_running_requests=args.sglang_max_running_requests,
            sglang_context_length=args.sglang_context_length,
            disable_thinking=not args.enable_thinking,
        )
        rows.append({"model": args.model_label, "benchmark": name, **values})
    write_kl_table(rows, out / "kl_divergence.md")
    print(f"wrote KL table to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TaylorMLP attack reproduction CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download configured models and datasets")
    p.add_argument("--config", default="TaylorAttack/configs/default.yaml")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default="TaylorAttack/data")
    p.add_argument("--models", action="store_true")
    p.add_argument("--datasets", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("calibrate", help="collect TaylorMLP activation stats")
    p.add_argument("--config", default="TaylorAttack/configs/default.yaml")
    p.add_argument("--model", required=True, help="model id or short_name from config")
    p.add_argument("--output", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("attack", help="protect and recover model MLP layers")
    p.add_argument("--config", default="TaylorAttack/configs/default.yaml")
    p.add_argument("--model", required=True, help="model id or short_name from config")
    p.add_argument("--stats-file", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--layers", type=int, default=None, help="attack first N layers")
    p.add_argument("--layer", type=int, default=None, help="attack one layer only")
    p.add_argument("--dtype", default=None)
    p.add_argument("--save-recovered-model", action="store_true")
    p.set_defaults(func=cmd_attack)

    p = sub.add_parser("kl", help="compute KL(original || recovered)")
    p.add_argument("--config", default="TaylorAttack/configs/default.yaml")
    p.add_argument("--original-model", required=True)
    p.add_argument("--recovered-model", required=True)
    p.add_argument("--model-label", default="model")
    p.add_argument("--benchmark", action="append")
    p.add_argument("--output", default="TaylorAttack/results/kl")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--progress-interval", type=int, default=128)
    p.add_argument("--top-logprobs", type=int, default=16)
    p.add_argument("--max-token-ids", type=int, default=2048)
    p.add_argument("--sglang-tp-size", type=int, default=1)
    p.add_argument("--sglang-mem-fraction-static", type=float, default=0.88)
    p.add_argument("--sglang-max-running-requests", type=int, default=512)
    p.add_argument("--sglang-context-length", type=int, default=None)
    p.add_argument(
        "--enable-thinking",
        action="store_true"
    )
    p.set_defaults(func=cmd_kl)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
