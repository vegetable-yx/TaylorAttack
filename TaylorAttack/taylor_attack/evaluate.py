from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from .dataset import BenchmarkExample, load_benchmark_examples
from .config import ensure_dir


def _progress(prefix: str, done: int, total: int, start_time: float) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    rate = done / elapsed
    print(
        f"[benchmark] {prefix}: {done}/{total} prompts, {rate:.2f} prompts/s",
        flush=True,
    )



def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_prompts_for_model(
    prompts: list[str],
    tokenizer_path: str,
    *,
    disable_thinking: bool = True,
) -> list[str]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return prompts

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        print(
            f"[benchmark] tokenizer load failed for {tokenizer_path}; using raw prompts: {exc}",
            flush=True,
        )
        return prompts

    if not getattr(tokenizer, "chat_template", None):
        print(
            f"[benchmark] tokenizer for {tokenizer_path} has no chat template; using raw prompts",
            flush=True,
        )
        return prompts

    model_hint = tokenizer_path.lower()
    formatted: list[str] = []
    used_enable_thinking = False
    for prompt in prompts:
        content = prompt
        if disable_thinking and "qwen" in model_hint and "/no_think" not in content:
            content = f"{content}\n\n/no_think"
        messages = [{"role": "user", "content": content}]
        if disable_thinking:
            try:
                formatted.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                used_enable_thinking = True
                continue
            except TypeError:
                pass
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    mode = "no-think" if disable_thinking else "default"
    extra = " with enable_thinking=False" if used_enable_thinking else ""
    print(
        f"[benchmark] formatted {len(prompts)} prompts with chat template ({mode}{extra})",
        flush=True,
    )
    return formatted


def _resolve_config_path(model_or_repo: str) -> Path | None:
    local = Path(model_or_repo)
    if local.exists():
        path = local / "config.json" if local.is_dir() else local
        return path if path.exists() else None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(
            hf_hub_download(
                model_or_repo,
                "config.json",
                token=os.environ.get("HF_TOKEN") or None,
            )
        )
    except Exception:
        return None


def _prepare_sglang_model_path(model_path: str, tokenizer_path: str | None) -> str:
    model_dir = Path(model_path)
    if tokenizer_path is None or not model_dir.is_dir():
        return model_path

    recovered_config_path = model_dir / "config.json"
    original_config_path = _resolve_config_path(tokenizer_path)
    recovered_config = _load_json_if_exists(recovered_config_path)
    original_config = _load_json_if_exists(original_config_path) if original_config_path else None
    if not recovered_config or not original_config:
        return model_path

    same_model_type = recovered_config.get("model_type") == original_config.get("model_type")
    same_arch = recovered_config.get("architectures") == original_config.get("architectures")
    if same_model_type and same_arch:
        return model_path

    sidecar_dir = model_dir.parent / f"{model_dir.name}_sglang"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for child in model_dir.iterdir():
        target = sidecar_dir / child.name
        if child.name == "config.json":
            continue
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(os.path.relpath(child.resolve(), sidecar_dir.resolve()))
        except OSError:
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
    shutil.copy2(original_config_path, sidecar_dir / "config.json")
    generation_config = original_config_path.parent / "generation_config.json"
    if generation_config.exists() and not (sidecar_dir / "generation_config.json").exists():
        shutil.copy2(generation_config, sidecar_dir / "generation_config.json")
    print(
        f"[benchmark] prepared SGLang sidecar config for {model_path}: "
        f"{recovered_config.get('model_type')} -> {original_config.get('model_type')}",
        flush=True,
    )
    return str(sidecar_dir)


def _make_sglang_engine(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    tokenizer_path: str | None = None,
    tp_size: int = 1,
    mem_fraction_static: float | None = 0.88,
    max_running_requests: int | None = 512,
    context_length: int | None = None,
):
    try:
        import sglang as sgl
    except ImportError as exc:
        raise RuntimeError("Install sglang to run benchmark evaluation") from exc

    engine_model_path = _prepare_sglang_model_path(model_path, tokenizer_path)
    engine_kwargs: dict[str, Any] = {
        "model_path": engine_model_path,
        "dtype": dtype,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "log_level": "error",
        "disable_piecewise_cuda_graph": True,
    }
    if tokenizer_path:
        engine_kwargs["tokenizer_path"] = tokenizer_path
    if mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = mem_fraction_static
    if max_running_requests is not None:
        engine_kwargs["max_running_requests"] = max_running_requests
    if context_length is not None:
        engine_kwargs["context_length"] = context_length
    return sgl.Engine(**engine_kwargs), engine_model_path


def _extract_top_logprobs(output: Any) -> list[list[tuple[float, int]]]:
    meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
    positions = meta.get("input_top_logprobs") or []
    out: list[list[tuple[float, int]]] = []
    for entries in positions:
        converted = []
        for item in entries or []:
            if not item or item[0] is None or item[1] is None:
                continue
            converted.append((float(item[0]), int(item[1])))
        out.append(converted)
    return out


def _extract_token_id_logprob_maps(output: Any) -> list[dict[int, float]]:
    meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
    positions = meta.get("input_token_ids_logprobs") or []
    out: list[dict[int, float]] = []
    for entries in positions:
        converted: dict[int, float] = {}
        for item in entries or []:
            if not item or item[0] is None or item[1] is None:
                continue
            converted[int(item[1])] = float(item[0])
        out.append(converted)
    return out


def _union_top_token_ids(
    top_logprobs: list[list[tuple[float, int]]],
    max_token_ids: int,
) -> list[int]:
    seen = set()
    out = []
    for position in top_logprobs:
        for _, token_id in position:
            if token_id in seen:
                continue
            seen.add(token_id)
            out.append(token_id)
            if len(out) >= max_token_ids:
                return out
    return out


def _topk_kl_values(
    original_top_logprobs: list[list[tuple[float, int]]],
    recovered_logprob_maps: list[dict[int, float]],
) -> list[float]:
    values = []
    for original_position, recovered_position in zip(
        original_top_logprobs, recovered_logprob_maps
    ):
        original_lps = []
        recovered_lps = []
        for original_lp, token_id in original_position:
            recovered_lp = recovered_position.get(token_id)
            if recovered_lp is None:
                continue
            original_lps.append(original_lp)
            recovered_lps.append(recovered_lp)
        if len(original_lps) <= 1:
            continue
        original_log_z = math.log(sum(math.exp(x) for x in original_lps))
        recovered_log_z = math.log(sum(math.exp(x) for x in recovered_lps))
        kl = 0.0
        for original_lp, recovered_lp in zip(original_lps, recovered_lps):
            log_p = original_lp - original_log_z
            log_q = recovered_lp - recovered_log_z
            kl += math.exp(log_p) * (log_p - log_q)
        values.append(max(kl, 0.0))
    return values


def score_top_logprobs_sglang(
    model_path: str,
    prompts: list[str],
    *,
    dtype: str = "bfloat16",
    tokenizer_path: str | None = None,
    batch_size: int = 128,
    progress_label: str,
    progress_interval: int = 128,
    top_logprobs_num: int = 16,
    tp_size: int = 1,
    mem_fraction_static: float | None = 0.88,
    max_running_requests: int | None = 512,
    context_length: int | None = None,
) -> list[list[list[tuple[float, int]]]]:
    batch_size = max(batch_size, 1)
    engine, engine_model_path = _make_sglang_engine(
        model_path,
        dtype=dtype,
        tokenizer_path=tokenizer_path,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        context_length=context_length,
    )
    print(
        f"[kl] starting SGLang top-{top_logprobs_num} scorer: model={engine_model_path}",
        flush=True,
    )
    sampling_params = {"temperature": 0.0, "max_new_tokens": 0}
    results: list[list[list[tuple[float, int]]]] = []
    start_time = time.perf_counter()
    total = len(prompts)
    try:
        for start in range(0, total, batch_size):
            chunk = prompts[start : start + batch_size]
            raw_outputs = engine.generate(
                chunk,
                sampling_params,
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=top_logprobs_num,
            )
            if isinstance(raw_outputs, dict):
                raw_outputs = [raw_outputs]
            results.extend(_extract_top_logprobs(item) for item in raw_outputs)
            done = min(start + len(chunk), total)
            if done == total or done % max(progress_interval, batch_size) == 0:
                _progress(progress_label, done, total, start_time)
    finally:
        engine.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def score_token_ids_logprobs_sglang(
    model_path: str,
    prompts: list[str],
    token_ids_by_prompt: list[list[int]],
    *,
    dtype: str = "bfloat16",
    tokenizer_path: str | None = None,
    batch_size: int = 128,
    progress_label: str,
    progress_interval: int = 128,
    tp_size: int = 1,
    mem_fraction_static: float | None = 0.88,
    max_running_requests: int | None = 512,
    context_length: int | None = None,
) -> list[list[dict[int, float]]]:
    batch_size = max(batch_size, 1)
    engine, engine_model_path = _make_sglang_engine(
        model_path,
        dtype=dtype,
        tokenizer_path=tokenizer_path,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        context_length=context_length,
    )
    print(f"[kl] starting SGLang token-id scorer: model={engine_model_path}", flush=True)
    sampling_params = {"temperature": 0.0, "max_new_tokens": 0}
    results: list[list[dict[int, float]]] = []
    start_time = time.perf_counter()
    total = len(prompts)
    try:
        for start in range(0, total, batch_size):
            chunk = prompts[start : start + batch_size]
            token_chunk = token_ids_by_prompt[start : start + batch_size]
            raw_outputs = engine.generate(
                chunk,
                sampling_params,
                return_logprob=True,
                logprob_start_len=0,
                token_ids_logprob=token_chunk,
            )
            if isinstance(raw_outputs, dict):
                raw_outputs = [raw_outputs]
            results.extend(_extract_token_id_logprob_maps(item) for item in raw_outputs)
            done = min(start + len(chunk), total)
            if done == total or done % max(progress_interval, batch_size) == 0:
                _progress(progress_label, done, total, start_time)
    finally:
        engine.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def kl_pair(
    model_a_path: str,
    model_b_path: str,
    benchmark: dict[str, Any],
    output_dir: str | Path,
    max_samples: int | None = None,
    dtype: str = "bfloat16",
    batch_size: int = 128,
    progress_interval: int = 128,
    top_logprobs_num: int = 16,
    max_token_ids: int = 2048,
    sglang_tp_size: int = 1,
    sglang_mem_fraction_static: float | None = 0.88,
    sglang_max_running_requests: int | None = 512,
    sglang_context_length: int | None = None,
    disable_thinking: bool = True,
) -> dict[str, float]:
    start_time = time.perf_counter()
    examples = load_benchmark_examples(benchmark, max_samples=max_samples)
    prompts = [ex.prompt for ex in examples]
    formatted_prompts = _format_prompts_for_model(
        prompts,
        model_a_path,
        disable_thinking=disable_thinking,
    )
    print(
        f"[kl] loaded {len(prompts)} examples for {benchmark['name']} "
        f"(top_logprobs={top_logprobs_num})",
        flush=True,
    )
    original_top_logprobs = score_top_logprobs_sglang(
        model_a_path,
        formatted_prompts,
        dtype=dtype,
        batch_size=batch_size,
        progress_label=f"{benchmark['name']}/original-topk",
        progress_interval=progress_interval,
        top_logprobs_num=top_logprobs_num,
        tp_size=sglang_tp_size,
        mem_fraction_static=sglang_mem_fraction_static,
        max_running_requests=sglang_max_running_requests,
        context_length=sglang_context_length,
    )
    token_ids_by_prompt = [
        _union_top_token_ids(item, max_token_ids=max_token_ids)
        for item in original_top_logprobs
    ]
    recovered_logprobs = score_token_ids_logprobs_sglang(
        model_b_path,
        formatted_prompts,
        token_ids_by_prompt,
        dtype=dtype,
        tokenizer_path=model_a_path,
        batch_size=batch_size,
        progress_label=f"{benchmark['name']}/recovered-tokenids",
        progress_interval=progress_interval,
        tp_size=sglang_tp_size,
        mem_fraction_static=sglang_mem_fraction_static,
        max_running_requests=sglang_max_running_requests,
        context_length=sglang_context_length,
    )

    values = []
    for original_item, recovered_item in zip(original_top_logprobs, recovered_logprobs):
        values.extend(_topk_kl_values(original_item, recovered_item))

    tensor = torch.tensor(values, dtype=torch.float64)
    result = {
        "mean": float(tensor.mean()) if tensor.numel() else 0.0,
        "std": float(tensor.std(unbiased=False)) if tensor.numel() else 0.0,
        "median": float(tensor.median()) if tensor.numel() else 0.0,
        "top_logprobs_num": top_logprobs_num,
        "tokens": int(tensor.numel()),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    out = ensure_dir(output_dir)
    (out / f"{benchmark['name']}_kl.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def write_kl_table(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Model | Benchmark | Top-K | Tokens | KL mean | KL std | KL median | Run Time |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = "".join(
        f"| {row['model']} | {row['benchmark']} | {row.get('top_logprobs_num', '')} | "
        f"{row.get('tokens', '')} | {row['mean']:.6g} | "
        f"{row['std']:.6g} | {row['median']:.6g} | "
        f"{float(row.get('elapsed_sec', 0.0)):.2f}s |\n"
        for row in rows
    )
    path.write_text(header + body, encoding="utf-8")
