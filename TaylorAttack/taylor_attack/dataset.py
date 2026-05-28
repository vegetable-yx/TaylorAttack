from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .download import get_hf_token


@dataclass(frozen=True)
class BenchmarkExample:
    prompt: str
    answer: str
    metric: str


def _stable_shuffle(
    items: list[tuple[str, str]], seed_text: str
) -> list[tuple[str, str]]:
    keyed = []
    for label, text in items:
        digest = hashlib.sha256(
            f"{seed_text}|{label}|{text}".encode("utf-8")
        ).hexdigest()
        keyed.append((digest, label, text))
    keyed.sort()
    return [(label, text) for _, label, text in keyed]


def load_benchmark_examples(
    bench: dict[str, Any], max_samples: int | None = None
) -> list[BenchmarkExample]:
    if bench.get("source") == "hf_file":
        return _load_hf_csv_examples(bench, max_samples=max_samples)
    return _load_dataset_examples(bench, max_samples=max_samples)


def _load_dataset_examples(
    bench: dict[str, Any], max_samples: int | None = None
) -> list[BenchmarkExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load benchmark examples") from exc

    ds = load_dataset(
        bench["dataset_id"],
        bench.get("dataset_config"),
        split=bench.get("split", "test"),
        token=get_hf_token(),
        **(bench.get("load_kwargs") or {}),
    )
    out = []
    q_field = bench.get("question_field", "question")
    a_field = bench.get("answer_field", "answer")
    for row in ds:
        question = str(row[q_field])
        answer = str(row[a_field])
        out.append(
            BenchmarkExample(
                prompt=_math_prompt(question),
                answer=answer,
                metric=bench.get("metric", "exact_match"),
            )
        )
        if max_samples and len(out) >= max_samples:
            break
    return out


def _load_hf_csv_examples(
    bench: dict[str, Any], max_samples: int | None = None
) -> list[BenchmarkExample]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to load HF CSV benchmarks") from exc

    try:
        csv_path = hf_hub_download(
            repo_id=bench["dataset_id"],
            repo_type="dataset",
            filename=bench["filename"],
            token=get_hf_token(),
        )
    except Exception as exc:
        if bench["dataset_id"].lower() == "idavidrein/gpqa":
            raise RuntimeError(
                "GPQA-Diamond requires accepting Idavidrein/gpqa terms on Hugging Face "
                "and an authorized HF_TOKEN."
            ) from exc
        raise

    out = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "Question" in row and "Correct Answer" in row:
                question = row["Question"]
                options = [
                    ("correct", row["Correct Answer"]),
                    ("wrong1", row.get("Incorrect Answer 1", "")),
                    ("wrong2", row.get("Incorrect Answer 2", "")),
                    ("wrong3", row.get("Incorrect Answer 3", "")),
                ]
                shuffled = _stable_shuffle(options, question)
                labels = ["A", "B", "C", "D"]
                correct_label = labels[[name for name, _ in shuffled].index("correct")]
                option_text = "\n".join(
                    f"{label}. {text}" for label, (_, text) in zip(labels, shuffled)
                )
                prompt = (
                    "Answer the multiple-choice question."
                    "Select exactly one option. Put the final "
                    "answer on the last line exactly as <answer>A</answer>, replacing A with "
                    "A, B, C, or D.\n\n"
                    f"Question: {question}\n\nOptions:\n{option_text}\n\nAnswer:"
                )
                out.append(
                    BenchmarkExample(
                        prompt=prompt, answer=correct_label, metric="multiple_choice"
                    )
                )
            else:
                values = list(row.values())
                out.append(
                    BenchmarkExample(
                        prompt=_math_prompt(values[0]),
                        answer=values[-1],
                        metric=bench.get("metric", "exact_match"),
                    )
                )
            if max_samples and len(out) >= max_samples:
                break
    return out


def _math_prompt(question: str) -> str:
    return (
        "Solve each problem. Put the final answer on the last line in the form "
        "\\boxed{...}.\n\n"
        "Problem:\nA box has 2 red balls and 3 blue balls. How many balls are in the box?\n\n"
        "Solution:\nThere are 2 + 3 = 5 balls.\n\\boxed{5}\n\n"
        "Problem:\nCompute $\\frac{1}{2}+\\frac{1}{3}$.\n\n"
        "Solution:\nA common denominator is 6, so $\\frac{1}{2}+\\frac{1}{3}"
        "=\\frac{3}{6}+\\frac{2}{6}=\\frac{5}{6}$.\n\\boxed{\\frac{5}{6}}\n\n"
        f"Problem:\n{question}\n\nSolution:"
    )
