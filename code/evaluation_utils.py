"""Shared checkpoint, metric, and reporting helpers for AnyMo evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

import config


def load_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= int(max_samples):
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_codebook_artifact(eval_jsonl: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    for parent in [Path(eval_jsonl).parent, *Path(eval_jsonl).parents]:
        candidate = parent / "imu_codebook_lookup.pt"
        if candidate.exists():
            return candidate
    default_path = config.INSTRUCTION_DATA_DIR / "imu_codebook_lookup.pt"
    if default_path.exists():
        return default_path
    raise FileNotFoundError(
        "Could not resolve imu_codebook_lookup.pt; pass --codebook-artifact explicitly."
    )


def resolve_codebook_artifact_for_args(args: argparse.Namespace) -> Path:
    if args.codebook_artifact is not None:
        return Path(args.codebook_artifact)
    if args.eval_jsonl is not None:
        return resolve_codebook_artifact(Path(args.eval_jsonl), None)
    candidate = Path(args.eval_root) / "imu_codebook_lookup.pt"
    if candidate.exists():
        return candidate
    return resolve_codebook_artifact(Path(args.eval_root) / "all_test.jsonl", None)


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def _checkpoint_state_path(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint
    bin_path = checkpoint / "pytorch_model.bin"
    if bin_path.exists():
        return bin_path
    safetensors_path = checkpoint / "model.safetensors"
    if safetensors_path.exists():
        return safetensors_path
    raise FileNotFoundError(
        f"Could not find pytorch_model.bin or model.safetensors in {checkpoint}"
    )


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def resolve_runtime_vocab_size(
    state_dict: dict[str, Any], tokenizer_vocab_size: int
) -> int:
    vocab_size = int(tokenizer_vocab_size)
    embedding_keys = (
        "llm.model.embed_tokens.weight",
        "llm.model.language_model.embed_tokens.weight",
        "llm.lm_head.weight",
    )
    for key in embedding_keys:
        tensor = state_dict.get(key)
        if tensor is not None and getattr(tensor, "shape", None):
            vocab_size = max(vocab_size, int(tensor.shape[0]))
    return vocab_size


def resolve_mapping_vocab_size(
    state_dict: dict[str, Any], runtime_vocab_size: int
) -> int:
    tensor = state_dict.get("token_id_to_code_index")
    if tensor is not None and getattr(tensor, "shape", None):
        return int(tensor.shape[0])
    return int(runtime_vocab_size)


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(float(score) - max_score) for score in scores]
    total = sum(exp_scores)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(scores)] * len(scores)
    return [score / total for score in exp_scores]


def _attach_prediction(row: dict[str, Any], scores: list[float]) -> dict[str, Any]:
    choices = row["choices"]
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    ranks = {idx: rank + 1 for (rank, idx) in enumerate(order)}
    probabilities = _softmax(scores)
    top1 = choices[order[0]]
    top2 = [choices[idx] for idx in order[:2]]
    pred = dict(row)
    pred["choice_scores"] = [
        {
            "key": choice["key"],
            "label_id": int(choice["label_id"]),
            "label": choice["label"],
            "score": float(score),
            "probability": float(probability),
            "rank": int(ranks[idx]),
        }
        for (idx, (choice, score, probability)) in enumerate(
            zip(choices, scores, probabilities)
        )
    ]
    pred["predicted_key"] = str(top1["key"])
    pred["predicted_label_id"] = int(top1["label_id"])
    pred["predicted_label_text"] = str(top1["label"])
    pred["top2_keys"] = [str(choice["key"]) for choice in top2]
    pred["top2_label_ids"] = [int(choice["label_id"]) for choice in top2]
    pred["top2_probabilities"] = [float(probabilities[idx]) for idx in order[:2]]
    pred["predicted_probability"] = float(probabilities[order[0]])
    pred["answer_probability"] = float(
        next(
            (
                probabilities[idx]
                for (idx, choice) in enumerate(choices)
                if str(choice["key"]) == str(row["answer_key"])
            ),
            math.nan,
        )
    )
    pred["probability_margin"] = float(
        probabilities[order[0]] - probabilities[order[1]]
        if len(order) > 1
        else probabilities[order[0]]
    )
    pred["correct"] = int(pred["predicted_key"] == str(row["answer_key"]))
    pred["recall_at_2_hit"] = int(str(row["answer_key"]) in pred["top2_keys"])
    return pred


def compute_classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "recall_at_2": math.nan,
        }
    labels = sorted(
        {int(row["label_id"]) for row in records}
        | {int(row["predicted_label_id"]) for row in records}
    )
    correct = sum(
        (int(row["label_id"]) == int(row["predicted_label_id"]) for row in records)
    )
    recall2 = sum(
        (
            int(row["label_id"]) in [int(x) for x in row.get("top2_label_ids", [])]
            for row in records
        )
    )
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    weighted_f1_sum = 0.0
    total_support = 0
    for label in labels:
        tp = sum(
            (
                int(row["label_id"]) == label
                and int(row["predicted_label_id"]) == label
                for row in records
            )
        )
        fp = sum(
            (
                int(row["label_id"]) != label
                and int(row["predicted_label_id"]) == label
                for row in records
            )
        )
        fn = sum(
            (
                int(row["label_id"]) == label
                and int(row["predicted_label_id"]) != label
                for row in records
            )
        )
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[str(label)] = {
            "support": int(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        f1_values.append(float(f1))
        weighted_f1_sum += float(f1) * int(support)
        total_support += int(support)
    return {
        "num_samples": len(records),
        "accuracy": correct / len(records),
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1_sum / max(total_support, 1),
        "recall_at_2": recall2 / len(records),
        "per_class": per_class,
    }


def build_confusion_matrix(
    records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    label_text_by_id: dict[int, str] = {}
    for row in records:
        true_id = int(row["label_id"])
        pred_id = int(row["predicted_label_id"])
        label_text_by_id.setdefault(true_id, str(row.get("label_text", true_id)))
        label_text_by_id.setdefault(
            pred_id, str(row.get("predicted_label_text", pred_id))
        )
    label_ids = sorted(label_text_by_id)
    index_by_label_id = {label_id: idx for (idx, label_id) in enumerate(label_ids)}
    matrix = [[0 for _ in label_ids] for _ in label_ids]
    for row in records:
        true_idx = index_by_label_id[int(row["label_id"])]
        pred_idx = index_by_label_id[int(row["predicted_label_id"])]
        matrix[true_idx][pred_idx] += 1
    labels = [
        {"label_id": label_id, "label": label_text_by_id[label_id]}
        for label_id in label_ids
    ]
    return (labels, matrix)


def write_confusion_matrix_csv(
    path: Path, labels: list[dict[str, Any]], matrix: list[list[float | int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "true_label_id",
        "true_label",
        *[f"{label['label_id']}:{label['label']}" for label in labels],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in zip(labels, matrix):
            out = {"true_label_id": label["label_id"], "true_label": label["label"]}
            for column_label, value in zip(labels, row):
                out[f"{column_label['label_id']}:{column_label['label']}"] = value
            writer.writerow(out)


def normalize_confusion_matrix(matrix: list[list[int]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        row_sum = sum(row)
        if row_sum:
            normalized.append([float(value) / row_sum for value in row])
        else:
            normalized.append([0.0 for _ in row])
    return normalized


def write_confusion_matrix_plot(
    path: Path, labels: list[dict[str, Any]], matrix: list[list[int]]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    short_labels = [str(label["label"]) for label in labels]
    size = max(8.0, min(22.0, 0.45 * len(labels) + 4.0))
    (fig, ax) = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=list(range(len(labels))),
        yticks=list(range(len(labels))),
        xticklabels=short_labels,
        yticklabels=short_labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sample_index",
        "label_id",
        "label_text",
        "answer_key",
        "predicted_key",
        "predicted_label_id",
        "predicted_label_text",
        "top2_keys",
        "top2_label_ids",
        "top2_probabilities",
        "predicted_probability",
        "answer_probability",
        "probability_margin",
        "correct",
        "recall_at_2_hit",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "sample_index": row.get("sample_index"),
                    "label_id": row.get("label_id"),
                    "label_text": row.get("label_text"),
                    "answer_key": row.get("answer_key"),
                    "predicted_key": row.get("predicted_key"),
                    "predicted_label_id": row.get("predicted_label_id"),
                    "predicted_label_text": row.get("predicted_label_text"),
                    "top2_keys": " ".join(row.get("top2_keys", [])),
                    "top2_label_ids": " ".join(
                        (str(x) for x in row.get("top2_label_ids", []))
                    ),
                    "top2_probabilities": " ".join(
                        (f"{float(x):.8g}" for x in row.get("top2_probabilities", []))
                    ),
                    "predicted_probability": row.get("predicted_probability"),
                    "answer_probability": row.get("answer_probability"),
                    "probability_margin": row.get("probability_margin"),
                    "correct": row.get("correct"),
                    "recall_at_2_hit": row.get("recall_at_2_hit"),
                }
            )


def write_option_probabilities_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sample_index",
        "label_id",
        "label_text",
        "answer_key",
        "predicted_key",
        "correct",
        "choice_key",
        "choice_label_id",
        "choice_label",
        "choice_score",
        "choice_probability",
        "choice_rank",
        "is_answer",
        "is_predicted",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for choice in row.get("choice_scores", []):
                writer.writerow(
                    {
                        "dataset": row.get("dataset"),
                        "sample_index": row.get("sample_index"),
                        "label_id": row.get("label_id"),
                        "label_text": row.get("label_text"),
                        "answer_key": row.get("answer_key"),
                        "predicted_key": row.get("predicted_key"),
                        "correct": row.get("correct"),
                        "choice_key": choice.get("key"),
                        "choice_label_id": choice.get("label_id"),
                        "choice_label": choice.get("label"),
                        "choice_score": choice.get("score"),
                        "choice_probability": choice.get("probability"),
                        "choice_rank": choice.get("rank"),
                        "is_answer": int(
                            str(choice.get("key")) == str(row.get("answer_key"))
                        ),
                        "is_predicted": int(
                            str(choice.get("key")) == str(row.get("predicted_key"))
                        ),
                    }
                )


def write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sample_index",
        "label_id",
        "label_text",
        "answer_key",
        "answer_probability",
        "predicted_key",
        "predicted_label_id",
        "predicted_label_text",
        "predicted_probability",
        "probability_margin",
        "top2_keys",
        "top2_label_ids",
        "top2_probabilities",
        "recall_at_2_hit",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "sample_index": row.get("sample_index"),
                    "label_id": row.get("label_id"),
                    "label_text": row.get("label_text"),
                    "answer_key": row.get("answer_key"),
                    "answer_probability": row.get("answer_probability"),
                    "predicted_key": row.get("predicted_key"),
                    "predicted_label_id": row.get("predicted_label_id"),
                    "predicted_label_text": row.get("predicted_label_text"),
                    "predicted_probability": row.get("predicted_probability"),
                    "probability_margin": row.get("probability_margin"),
                    "top2_keys": " ".join(row.get("top2_keys", [])),
                    "top2_label_ids": " ".join(
                        (str(x) for x in row.get("top2_label_ids", []))
                    ),
                    "top2_probabilities": " ".join(
                        (f"{float(x):.8g}" for x in row.get("top2_probabilities", []))
                    ),
                    "recall_at_2_hit": row.get("recall_at_2_hit"),
                }
            )


def write_analysis_outputs(output_dir: Path, predictions: list[dict[str, Any]]) -> None:
    failures = [row for row in predictions if not int(row.get("correct", 0))]
    write_option_probabilities_csv(output_dir / "option_probabilities.csv", predictions)
    write_jsonl(output_dir / "failures.jsonl", failures)
    write_failures_csv(output_dir / "failures.csv", failures)
    (labels, matrix) = build_confusion_matrix(predictions)
    write_confusion_matrix_csv(output_dir / "confusion_matrix.csv", labels, matrix)
    write_confusion_matrix_csv(
        output_dir / "confusion_matrix_normalized.csv",
        labels,
        normalize_confusion_matrix(matrix),
    )
    write_confusion_matrix_plot(output_dir / "confusion_matrix.png", labels, matrix)


def discover_eval_jsonls(eval_root: Path) -> list[Path]:
    paths = sorted(Path(eval_root).glob("*/*/test.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No per-dataset test.jsonl files found under {eval_root}"
        )
    return paths


def dataset_relative_name(eval_root: Path, jsonl_path: Path) -> str:
    return str(Path(jsonl_path).relative_to(eval_root).parent)


def summarize_metric_row(dataset: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "num_samples": metrics.get("num_samples"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "recall_at_2": metrics.get("recall_at_2"),
    }


def write_metrics_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "num_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "recall_at_2",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
