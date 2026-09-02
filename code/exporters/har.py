from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

import body_part_mapping
import config
import data as project_data
from exporters import instruction as instruction_export
from exporters import tokens as token_export
import imu_token_utils
import train_pqvae
import training_utils as utils


DEFAULT_OUTPUT_DIR = config.HAR_EVALUATION_DIR
DEFAULT_ARRAY_DATASETS = project_data.ARRAY_HAR_DATASETS
DEFAULT_EGO_DATASETS = project_data.EGO_HAR_DATASETS
DEFAULT_OPENPACK_DATASETS = project_data.OPENPACK_HAR_DATASETS


@dataclass(frozen=True)
class GraphEvalSample:
    dataset_name: str
    source: str
    sample_index: int
    label_id: int
    graph_x: np.ndarray
    visible_mask: np.ndarray
    visible_segments: list[str]

    @property
    def time_steps(self) -> int:
        return int(self.graph_x.shape[1])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export zero-shot HAR evaluation data with offline AnyMo IMU tokens."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        choices=("all", "array", "ego", "openpack"),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"], help="Dataset names, or 'all'."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "val", "test"),
        help=(
            "Dataset splits to export. test writes evaluation JSONL; train can additionally "
            "write MCQ and label-contrastive JSONL for downstream training."
        ),
    )
    parser.add_argument(
        "--har-data-root",
        dest="har_data_root",
        type=Path,
        default=project_data.DEFAULT_ARRAY_HAR_ROOT,
        help="Root of the preprocessed array-format HAR datasets.",
    )
    parser.add_argument(
        "--ego4d-root", type=Path, default=project_data.DEFAULT_EGO4D_ROOT
    )
    parser.add_argument(
        "--mmea-root", type=Path, default=project_data.DEFAULT_MMEA_ROOT
    )
    parser.add_argument(
        "--egoexo4d-root", type=Path, default=project_data.DEFAULT_EGOEXO4D_ROOT
    )
    parser.add_argument(
        "--openpack-root", type=Path, default=project_data.DEFAULT_OPENPACK_ROOT
    )
    parser.add_argument(
        "--openpack-sample-mode", choices=("segment", "window"), default="segment"
    )
    parser.add_argument("--openpack-window-seconds", type=float, default=4.0)
    parser.add_argument("--openpack-stride-seconds", type=float, default=2.0)
    parser.add_argument("--openpack-min-segment-seconds", type=float, default=1.0)
    parser.add_argument("--openpack-max-segment-seconds", type=float, default=30.0)
    parser.add_argument("--openpack-allow-missing-sessions", action="store_true")
    parser.add_argument("--target-sample-rate-hz", type=int, default=60)
    parser.add_argument(
        "--encoder-ckpt",
        type=Path,
        default=config.DEFAULT_ENCODER_CHECKPOINT,
    )
    parser.add_argument(
        "--tokenizer-export", type=Path, default=token_export.DEFAULT_TOKENIZER_EXPORT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-samples-per-dataset", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def normalize_sources(sources: Iterable[str]) -> list[str]:
    source_list = [str(source).lower() for source in sources]
    if "all" in source_list:
        return ["array", "ego", "openpack"]
    return list(dict.fromkeys(source_list))


def normalize_splits(splits: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(split).lower() for split in splits))


def resolve_dataset_names(source: str, requested: Iterable[str]) -> list[str]:
    requested_names = [str(name) for name in requested]
    if source == "array":
        available = list(DEFAULT_ARRAY_DATASETS)
    elif source == "ego":
        available = list(DEFAULT_EGO_DATASETS)
    elif source == "openpack":
        available = list(DEFAULT_OPENPACK_DATASETS)
    else:
        raise ValueError(f"Unknown source: {source}")
    if "all" in requested_names:
        return available
    selected = [name for name in requested_names if name in available]
    return selected


def ordered_label_items(label_dictionary: dict[str, str]) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for key, value in label_dictionary.items():
        items.append((int(key), str(value)))
    return sorted(items, key=lambda item: item[0])


def build_dataset_choices(
    label_dictionary: dict[str, str]
) -> list[dict[str, str | int]]:
    labels = ordered_label_items(label_dictionary)
    keys = instruction_export.build_option_keys(len(labels))
    return [
        {"key": key, "label_id": int(label_id), "label": str(label)}
        for key, (label_id, label) in zip(keys, labels)
    ]


def answer_key_for_label(label_id: int, choices: list[dict[str, str | int]]) -> str:
    for choice in choices:
        if int(choice["label_id"]) == int(label_id):
            return str(choice["key"])
    raise KeyError(f"Label id {label_id} was not found in dataset choices")


def label_text_for_id(label_id: int, label_dictionary: dict[str, str]) -> str:
    return str(label_dictionary[str(int(label_id))])


def format_choices(choices: list[dict[str, str | int]]) -> str:
    return "\n".join(f"{choice['key']}: {choice['label']}" for choice in choices)


def build_eval_mcq_record(
    dataset_name: str,
    sample_index: int,
    imu_token_text: str,
    label_id: int,
    label_dictionary: dict[str, str],
    visible_segments: list[str],
    source: str | None = None,
) -> dict[str, Any]:
    choices = build_dataset_choices(label_dictionary)
    answer_key = answer_key_for_label(label_id, choices)
    label_text = label_text_for_id(label_id, label_dictionary)
    prompt = instruction_export.HAR_MCQ_PROMPT_TEMPLATE.format(
        imu_token=imu_token_text,
        sensor_context=instruction_export.format_sensor_context(visible_segments),
        choices=format_choices(choices),
    )
    answer_text = instruction_export.build_har_answer_text(answer_key, label_text)
    record: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        "task": "har_mcq_eval",
        "dataset": str(dataset_name),
        "sample_index": int(sample_index),
        "label_id": int(label_id),
        "label_text": label_text,
        "answer_key": answer_key,
        "answer_text": answer_text,
        "choices": choices,
        "visible_segments": list(visible_segments),
    }
    if source is not None:
        record["source"] = str(source)
    return record


def build_train_mcq_record(
    dataset_name: str,
    sample_index: int,
    imu_token_text: str,
    label_id: int,
    label_dictionary: dict[str, str],
    visible_segments: list[str],
    source: str | None = None,
) -> dict[str, Any]:
    record = build_eval_mcq_record(
        dataset_name=dataset_name,
        sample_index=sample_index,
        imu_token_text=imu_token_text,
        label_id=label_id,
        label_dictionary=label_dictionary,
        visible_segments=visible_segments,
        source=source,
    )
    record["task"] = "har_mcq"
    record["activity_label"] = record["label_text"]
    record["answer_label"] = record["label_text"]
    return record


def build_train_contrastive_record(
    dataset_name: str,
    sample_index: int,
    imu_token_text: str,
    label_id: int,
    label_dictionary: dict[str, str],
    visible_segments: list[str],
    source: str | None = None,
) -> dict[str, Any]:
    label_text = label_text_for_id(label_id, label_dictionary)
    record: dict[str, Any] = {
        "task": "contrastive",
        "imu_token_text": str(imu_token_text),
        "sensor_context": instruction_export.format_sensor_context(visible_segments),
        "activity_label": label_text,
        "positive_texts": [{"type": "label", "text": label_text}],
        "dataset": str(dataset_name),
        "sample_index": int(sample_index),
        "label_id": int(label_id),
        "label_text": label_text,
        "visible_segments": list(visible_segments),
    }
    if source is not None:
        record["source"] = str(source)
    return record


def _group_widths(
    selected_channel_slices: tuple[tuple[tuple[int, int], ...], ...]
) -> list[int]:
    return [
        sum(int(end) - int(start) for start, end in group)
        for group in selected_channel_slices
    ]


def _normalize_sample_array(sample: np.ndarray, channel_axis: int | None) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)
    if channel_axis is None:
        if arr.ndim != 2:
            raise ValueError(f"Expected [T, C] sample array, got shape {arr.shape}")
        return arr
    if int(channel_axis) == 0:
        if arr.ndim != 2:
            raise ValueError(f"Expected [C, T] sample array, got shape {arr.shape}")
        return arr.T.astype(np.float32, copy=False)
    if int(channel_axis) == 1:
        if arr.ndim != 2:
            raise ValueError(f"Expected [T, C] sample array, got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported channel_axis={channel_axis}; expected 0, 1, or None")


def _split_groups(sample_tc: np.ndarray, group_widths: list[int]) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    offset = 0
    for width in group_widths:
        width = int(width)
        groups.append(sample_tc[:, offset : offset + width])
        offset += width
    if offset != int(sample_tc.shape[1]):
        raise ValueError(
            f"Selected group widths sum to {offset}, but sample has {sample_tc.shape[1]} channels"
        )
    return groups


def _segment_name_from_entry(entry: dict[str, object]) -> str:
    segment = str(entry["project_segment_name"])
    if segment not in project_data.SITE_TO_INDEX:
        raise KeyError(
            f"Mapped segment '{segment}' is not in project_data.SITE_TO_INDEX"
        )
    return segment


def _assignment_segments(
    dataset_name: str, source: str, label_id: int, num_groups: int
) -> list[tuple[int, str]]:
    if source == "array":
        entries = list(body_part_mapping.ARRAY_HAR_DATASET_TO_BODY_PARTS[dataset_name])
        if dataset_name == "UTD-MHAD":
            if num_groups != 1 or len(entries) != 2:
                raise ValueError(
                    "UTD-MHAD is expected to have one IMU group and two conditional body slots"
                )
            entry = entries[0] if int(label_id) < 21 else entries[1]
            return [(0, _segment_name_from_entry(entry))]
        if len(entries) != int(num_groups):
            raise ValueError(
                f"{dataset_name} has {num_groups} channel groups but {len(entries)} body mapping entries"
            )
        return [
            (idx, _segment_name_from_entry(entry)) for idx, entry in enumerate(entries)
        ]

    if source == "ego":
        entries = list(body_part_mapping.EGO_DATASET_TO_BODY_PARTS[dataset_name])
        if len(entries) != int(num_groups):
            raise ValueError(
                f"{dataset_name} has {num_groups} channel groups but {len(entries)} body mapping entries"
            )
        return [
            (idx, _segment_name_from_entry(entry)) for idx, entry in enumerate(entries)
        ]

    if source == "openpack":
        entries = list(body_part_mapping.OPENPACK_DATASET_TO_BODY_PARTS[dataset_name])
        if len(entries) != int(num_groups):
            raise ValueError(
                f"{dataset_name} has {num_groups} channel groups but {len(entries)} body mapping entries"
            )
        return [
            (idx, _segment_name_from_entry(entry)) for idx, entry in enumerate(entries)
        ]

    raise ValueError(f"Unknown source: {source}")


def build_graph_view_for_sample(
    dataset_name: str,
    sample: np.ndarray,
    label_id: int,
    selected_channel_slices: tuple[tuple[tuple[int, int], ...], ...],
    *,
    source: str,
    channel_axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sample_tc = _normalize_sample_array(sample, channel_axis=channel_axis)
    groups = _split_groups(sample_tc, _group_widths(selected_channel_slices))
    graph_x = np.zeros(
        (6, sample_tc.shape[0], len(project_data.SITE_NAMES), 1), dtype=np.float32
    )
    visible_mask = np.zeros((len(project_data.SITE_NAMES),), dtype=bool)
    visible_segments: list[str] = []

    for group_index, segment_name in _assignment_segments(
        dataset_name, source, label_id, len(groups)
    ):
        group = np.asarray(groups[group_index], dtype=np.float32)
        if group.shape[1] not in {3, 6}:
            raise ValueError(
                f"Expected group width 3 or 6 for {dataset_name}, got {group.shape[1]}"
            )
        node_index = project_data.SITE_TO_INDEX[segment_name]
        graph_x[: group.shape[1], :, node_index, 0] = group.T
        visible_mask[node_index] = True
        visible_segments.append(segment_name)

    return graph_x, visible_mask, visible_segments


def build_graph_views_for_dataset(
    dataset: Any, source: str
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    if source == "array":
        x_test = list(np.asarray(dataset.x_test, dtype=np.float32))
        channel_axis = None
    elif source in {"ego", "openpack"}:
        x_test = list(dataset.x_test)
        channel_axis = (
            None if dataset.channel_axis is None else int(dataset.channel_axis)
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    graph_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    visible_segments_rows: list[list[str]] = []
    for sample_index, (sample, label_id) in enumerate(zip(x_test, dataset.y_test)):
        graph_x, visible_mask, visible_segments = build_graph_view_for_sample(
            dataset_name=dataset.name,
            sample=sample,
            label_id=int(label_id),
            selected_channel_slices=dataset.selected_channel_slices,
            source=source,
            channel_axis=channel_axis,
        )
        graph_rows.append(graph_x)
        mask_rows.append(visible_mask)
        visible_segments_rows.append(visible_segments)

    lengths = {row.shape[1] for row in graph_rows}
    if len(lengths) != 1:
        raise ValueError(
            f"{dataset.name} has variable test sequence lengths; use build_eval_samples instead"
        )
    return (
        np.stack(graph_rows, axis=0),
        np.stack(mask_rows, axis=0),
        visible_segments_rows,
    )


def build_eval_samples(
    dataset: Any, source: str, limit_samples: int | None = None
) -> list[GraphEvalSample]:
    return build_split_samples(
        dataset, source=source, split="test", limit_samples=limit_samples
    )


def build_split_samples(
    dataset: Any,
    source: str,
    split: str = "test",
    limit_samples: int | None = None,
) -> list[GraphEvalSample]:
    split = str(split).lower()
    x_attr = f"x_{split}"
    y_attr = f"y_{split}"
    if not hasattr(dataset, x_attr) or not hasattr(dataset, y_attr):
        raise ValueError(f"{dataset.name} does not expose split '{split}'")
    x_values = getattr(dataset, x_attr)
    y_values = getattr(dataset, y_attr)
    if x_values is None or y_values is None:
        raise ValueError(f"{dataset.name} split '{split}' is empty")

    if source == "array":
        samples = list(np.asarray(x_values, dtype=np.float32))
        channel_axis = None
    elif source in {"ego", "openpack"}:
        samples = list(x_values)
        channel_axis = (
            None if dataset.channel_axis is None else int(dataset.channel_axis)
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    rows: list[GraphEvalSample] = []
    for sample_index, (sample, label_id) in enumerate(zip(samples, y_values)):
        if limit_samples is not None and len(rows) >= int(limit_samples):
            break
        graph_x, visible_mask, visible_segments = build_graph_view_for_sample(
            dataset_name=dataset.name,
            sample=sample,
            label_id=int(label_id),
            selected_channel_slices=dataset.selected_channel_slices,
            source=source,
            channel_axis=channel_axis,
        )
        rows.append(
            GraphEvalSample(
                dataset_name=dataset.name,
                source=source,
                sample_index=sample_index,
                label_id=int(label_id),
                graph_x=graph_x,
                visible_mask=visible_mask,
                visible_segments=visible_segments,
            )
        )
    return rows


def _batched_by_time(
    samples: list[GraphEvalSample], batch_size: int
) -> Iterable[list[GraphEvalSample]]:
    by_length: dict[int, list[GraphEvalSample]] = {}
    for sample in samples:
        by_length.setdefault(sample.time_steps, []).append(sample)
    for length in sorted(by_length):
        group = by_length[length]
        for start in range(0, len(group), int(batch_size)):
            yield group[start : start + int(batch_size)]


def tokenize_eval_samples(
    samples: list[GraphEvalSample],
    encoder: torch.nn.Module,
    pqvae: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[int, tuple[list[int], str]]:
    outputs: dict[int, tuple[list[int], str]] = {}
    for batch_samples in tqdm(
        list(_batched_by_time(samples, batch_size)), desc="tokenize eval", leave=False
    ):
        xb = torch.tensor(
            np.stack([sample.graph_x for sample in batch_samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
        visible_mask = torch.tensor(
            np.stack([sample.visible_mask for sample in batch_samples], axis=0),
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad():
            latents = encoder(xb, visible_node_mask=visible_mask)["global_seq_latent"]
            codes = pqvae.encode_codes(latents)
            token_ids = imu_token_utils.codes_to_local_imu_ids(
                codes, codebook_size=pqvae.codebook_size
            ).cpu()
        for sample, token_row in zip(batch_samples, token_ids.tolist()):
            ids = [int(token_id) for token_id in token_row]
            outputs[sample.sample_index] = (
                ids,
                imu_token_utils.local_ids_to_token_text(ids, include_bos_eos=True),
            )
    return outputs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_eval_dataset(source: str, name: str, args: argparse.Namespace) -> Any:
    expected_source = (
        "array"
        if name in project_data.ARRAY_HAR_DATASETS
        else "ego"
        if name in project_data.EGO_HAR_DATASETS
        else "openpack"
        if name in project_data.OPENPACK_HAR_DATASETS
        else None
    )
    if source != expected_source:
        raise ValueError(f"Dataset {name!r} does not belong to source {source!r}")
    return project_data.load_har_dataset(
        name,
        target_sample_rate_hz=int(args.target_sample_rate_hz),
        array_root=Path(args.har_data_root),
        ego4d_root=Path(args.ego4d_root),
        mmea_root=Path(args.mmea_root),
        egoexo4d_root=Path(args.egoexo4d_root),
        openpack_root=Path(args.openpack_root),
        openpack_sample_mode=str(args.openpack_sample_mode),
        openpack_window_seconds=float(args.openpack_window_seconds),
        openpack_stride_seconds=float(args.openpack_stride_seconds),
        openpack_min_segment_seconds=float(args.openpack_min_segment_seconds),
        openpack_max_segment_seconds=float(args.openpack_max_segment_seconds),
        openpack_require_all_sessions=not bool(args.openpack_allow_missing_sessions),
    )


def export_one_dataset(
    source: str,
    name: str,
    args: argparse.Namespace,
    encoder: torch.nn.Module,
    pqvae: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    dataset = load_eval_dataset(source, name, args)
    dataset_dir = Path(args.output_dir) / source / name
    split_summaries: dict[str, dict[str, Any]] = {}

    for split in normalize_splits(args.splits):
        samples = build_split_samples(
            dataset,
            source=source,
            split=split,
            limit_samples=args.limit_samples_per_dataset,
        )
        visible_segment_sets = sorted(
            {tuple(sample.visible_segments) for sample in samples}
        )
        tokenized = tokenize_eval_samples(
            samples, encoder, pqvae, device, batch_size=int(args.batch_size)
        )

        if split == "test":
            rows: list[dict[str, Any]] = []
            for sample in samples:
                token_ids, imu_token_text = tokenized[sample.sample_index]
                record = build_eval_mcq_record(
                    dataset_name=dataset.name,
                    sample_index=sample.sample_index,
                    imu_token_text=imu_token_text,
                    label_id=sample.label_id,
                    label_dictionary=dataset.label_dictionary,
                    visible_segments=sample.visible_segments,
                    source=source,
                )
                record["imu_token_ids"] = token_ids
                rows.append(record)
            write_jsonl(dataset_dir / "test.jsonl", rows)
            split_summaries[split] = {
                "num_samples": len(rows),
                "files": ["test.jsonl"],
                "visible_segment_sets": visible_segment_sets,
            }
            continue

        mcq_rows: list[dict[str, Any]] = []
        contrastive_rows: list[dict[str, Any]] = []
        for sample in samples:
            token_ids, imu_token_text = tokenized[sample.sample_index]
            mcq_record = build_train_mcq_record(
                dataset_name=dataset.name,
                sample_index=sample.sample_index,
                imu_token_text=imu_token_text,
                label_id=sample.label_id,
                label_dictionary=dataset.label_dictionary,
                visible_segments=sample.visible_segments,
                source=source,
            )
            mcq_record["imu_token_ids"] = token_ids
            mcq_rows.append(mcq_record)

            contrastive_record = build_train_contrastive_record(
                dataset_name=dataset.name,
                sample_index=sample.sample_index,
                imu_token_text=imu_token_text,
                label_id=sample.label_id,
                label_dictionary=dataset.label_dictionary,
                visible_segments=sample.visible_segments,
                source=source,
            )
            contrastive_record["imu_token_ids"] = token_ids
            contrastive_rows.append(contrastive_record)

        write_jsonl(dataset_dir / f"{split}.jsonl", mcq_rows)
        write_jsonl(dataset_dir / f"{split}_contrastive.jsonl", contrastive_rows)
        split_summaries[split] = {
            "num_samples": len(mcq_rows),
            "files": [f"{split}.jsonl", f"{split}_contrastive.jsonl"],
            "visible_segment_sets": visible_segment_sets,
        }

    visible_segment_sets = sorted(
        {
            tuple(segments)
            for split_summary in split_summaries.values()
            for segments in split_summary.get("visible_segment_sets", [])
        }
    )
    summary = {
        "source": source,
        "dataset": name,
        "num_samples": split_summaries.get("test", {}).get(
            "num_samples",
            sum(int(item["num_samples"]) for item in split_summaries.values()),
        ),
        "num_classes": int(dataset.num_classes),
        "target_sample_rate_hz": int(dataset.target_sample_rate_hz),
        "label_dictionary": dataset.label_dictionary,
        "choices": build_dataset_choices(dataset.label_dictionary),
        "visible_segment_sets": visible_segment_sets,
        "splits": split_summaries,
    }
    if hasattr(dataset, "sample_mode"):
        summary["sample_mode"] = str(dataset.sample_mode)
    if hasattr(dataset, "min_segment_seconds"):
        summary["min_segment_seconds"] = float(dataset.min_segment_seconds)
    if getattr(dataset, "max_segment_seconds", None) is not None:
        summary["max_segment_seconds"] = float(dataset.max_segment_seconds)
    utils.save_json(dataset_dir / "summary.json", summary)
    return summary


def export_all(args: argparse.Namespace) -> dict[str, Any]:
    utils.set_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_export.write_codebook_lookup_artifact(
        Path(args.tokenizer_export), output_dir / "imu_codebook_lookup.pt"
    )

    device = torch.device(args.device)
    encoder = train_pqvae.load_frozen_encoder(
        Path(args.encoder_ckpt), device=device, dropout=float(args.dropout)
    )
    pqvae = token_export.load_frozen_pqvae(Path(args.tokenizer_export), device=device)

    summaries: list[dict[str, Any]] = []
    for source in normalize_sources(args.sources):
        for name in resolve_dataset_names(source, args.datasets):
            summaries.append(
                export_one_dataset(source, name, args, encoder, pqvae, device)
            )

    all_test_rows: list[dict[str, Any]] = []
    all_train_rows: list[dict[str, Any]] = []
    all_train_contrastive_rows: list[dict[str, Any]] = []
    for summary in summaries:
        dataset_dir = output_dir / str(summary["source"]) / str(summary["dataset"])
        test_path = dataset_dir / "test.jsonl"
        if test_path.exists():
            with test_path.open("r", encoding="utf-8") as f:
                all_test_rows.extend(json.loads(line) for line in f if line.strip())
        train_path = dataset_dir / "train.jsonl"
        if train_path.exists():
            with train_path.open("r", encoding="utf-8") as f:
                all_train_rows.extend(json.loads(line) for line in f if line.strip())
        train_contrastive_path = dataset_dir / "train_contrastive.jsonl"
        if train_contrastive_path.exists():
            with train_contrastive_path.open("r", encoding="utf-8") as f:
                all_train_contrastive_rows.extend(
                    json.loads(line) for line in f if line.strip()
                )
    if all_test_rows:
        write_jsonl(output_dir / "all_test.jsonl", all_test_rows)
    if all_train_rows:
        write_jsonl(output_dir / "all_train.jsonl", all_train_rows)
    if all_train_contrastive_rows:
        write_jsonl(
            output_dir / "all_train_contrastive.jsonl", all_train_contrastive_rows
        )
    summary = {
        "output_dir": str(output_dir),
        "splits": normalize_splits(args.splits),
        "total_samples": len(all_test_rows),
        "total_test_samples": len(all_test_rows),
        "total_train_samples": len(all_train_rows),
        "total_train_contrastive_samples": len(all_train_contrastive_rows),
        "datasets": summaries,
        "encoder_ckpt": str(args.encoder_ckpt),
        "tokenizer_export": str(args.tokenizer_export),
    }
    utils.save_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = export_all(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
