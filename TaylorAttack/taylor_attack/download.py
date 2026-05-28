from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ensure_dir


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_models(config: dict[str, Any], cache_dir: str | Path | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to download models") from exc

    token = get_hf_token()
    for model in config.get("models", []):
        model_id = model["id"]
        print(f"[model] downloading {model_id}")
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
            local_files_only=False,
        )


def download_datasets(config: dict[str, Any], output_root: str | Path) -> None:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install datasets and huggingface_hub to download datasets") from exc

    token = get_hf_token()
    root = ensure_dir(output_root)
    for bench in config.get("datasets", []):
        name = bench["name"]
        dataset_id = bench["dataset_id"]
        out = ensure_dir(root / name)
        print(f"[dataset] downloading {name}: {dataset_id}")
        try:
            if bench.get("source") == "hf_file":
                path = hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    filename=bench["filename"],
                    token=token,
                )
                marker = out / "source_path.txt"
                marker.write_text(path + "\n", encoding="utf-8")
            else:
                dataset_config = bench.get("dataset_config")
                load_kwargs = bench.get("load_kwargs") or {}
                dataset = load_dataset(
                    dataset_id,
                    dataset_config,
                    split=bench.get("split"),
                    token=token,
                    **load_kwargs,
                )
                dataset.save_to_disk(str(out / "dataset"))
        except Exception as exc:
            if dataset_id.lower() == "idavidrein/gpqa":
                raise RuntimeError(
                    "GPQA-Diamond uses the official gated Idavidrein/gpqa source. "
                    "Accept the dataset terms on Hugging Face and rerun with an authorized HF_TOKEN."
                ) from exc
            raise
