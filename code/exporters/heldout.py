from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
import data as project_data
from exporters import har as har_export
from exporters import instruction as instruction_export
from exporters import tokens as token_export
import training_utils as utils


DEFAULT_OUTPUT_DIR = config.NYMERIA_HELDOUT_DIR
DEFAULT_EGOEXO_EVAL_INPUT_DIR = config.EGOEXO4D_PREPARED_DIR
DEFAULT_HELD_OUT_SUBJECTS = (
    "alec_meza",
    "bradley_herman",
    "dominique_frye",
    "justin_ramirez",
    "kyle_parker",
)

RETRIEVAL_IMU_PROMPT_TEMPLATE = (
    "Represent the human motion from the wearable IMU motion tokens.\n\n"
    "The IMU tokens are from IMU sensors attached to the user's {sensor_context}.\n\n"
    "Input IMU token:\n{imu_token}\n\n"
    "Return a compact embedding of the motion."
)
RETRIEVAL_TEXT_PROMPT_TEMPLATE = (
    "Represent the human motion described by the text.\n\n"
    "Motion description:\n{text}\n\n"
    "Return a compact embedding of the motion."
)

SUBJECT_RE = re.compile(r"^\d{8}_s\d+_(?P<subject>.+?)_act\d+_")


def build_argparser() -> argparse.ArgumentParser:
    parser = token_export.build_argparser()
    parser.description = "Export held-out Nymeria retrieval and captioning eval data with AnyMo IMU tokens."
    parser.add_argument(
        "--egoexo-eval-input-dir",
        type=Path,
        default=None,
        help=(
            "If set, tokenize an exported EgoExo4D atomic zero-shot dataset "
            "(retrieval/captioning JSONL plus imu_windows.npz) instead of Nymeria held-out windows."
        ),
    )
    parser.add_argument(
        "--egoexo-target-sample-rate-hz",
        type=int,
        default=project_data.DEFAULT_TARGET_SAMPLE_RATE_HZ,
        help="Target sample rate before mapping EgoExo4D head IMU windows into the ST-GCN graph.",
    )
    parser.add_argument(
        "--held-out-subjects", nargs="+", default=list(DEFAULT_HELD_OUT_SUBJECTS)
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Optional debug limit after held-out text-aligned windows are built.",
    )
    parser.add_argument(
        "--retrieval-samples-per-scenario",
        type=int,
        default=5,
        help=(
            "Also export a deterministic retrieval subset with this many windows "
            "from each held-out scenario. Use 0 to disable."
        ),
    )
    parser.set_defaults(alignment_mode="text_labels", output_dir=DEFAULT_OUTPUT_DIR)
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def subject_from_sample_dir(sample_dir: str) -> str:
    match = SUBJECT_RE.match(str(sample_dir))
    if match:
        return match.group("subject")
    parts = str(sample_dir).split("_")
    if len(parts) >= 6 and parts[0].isdigit() and parts[1].startswith("s"):
        return "_".join(parts[2:-2])
    return ""


def filter_held_out_records(
    records: list[token_export.data.SampleRecord],
    held_out_subjects: list[str],
) -> list[token_export.data.SampleRecord]:
    held_out = {
        str(subject).strip() for subject in held_out_subjects if str(subject).strip()
    }
    return [
        record
        for record in records
        if subject_from_sample_dir(record.sample_dir) in held_out
    ]


def clean_references(
    window: instruction_export.TokenizedInstructionWindow,
    rng: random.Random,
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    primary = instruction_export.clean_text(window.spec.primary_text)
    if primary:
        refs.append(
            {
                "type": "narration",
                "text": instruction_export.clean_narration_text(primary, rng),
            }
        )
    for text in window.spec.augmented_texts:
        cleaned = instruction_export.clean_text(text)
        if cleaned:
            refs.append(
                {
                    "type": "augmented_narration",
                    "text": instruction_export.clean_narration_text(cleaned, rng),
                }
            )
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in refs:
        key = (item["type"], item["text"])
        if item["text"] and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def window_metadata(
    window: instruction_export.TokenizedInstructionWindow, index: int
) -> dict[str, Any]:
    spec = window.spec
    subject = subject_from_sample_dir(spec.sample_dir)
    return {
        "query_id": f"{spec.sample_dir}:{spec.source_row_index}",
        "sample_dir": spec.sample_dir,
        "subject": subject,
        "clip_start": int(spec.clip_start),
        "clip_end": int(spec.clip_end),
        "source_csv_path": spec.source_csv_path,
        "source_row_index": spec.source_row_index,
        "order_index": int(index),
        "label_name": spec.label_name,
        "activity_label": instruction_export.clean_text(spec.activity_label),
        "sensor_context": instruction_export.format_sensor_context(
            window.sensor_context
        ),
        "imu_token_ids": [int(token_id) for token_id in window.imu_token_ids],
        "imu_token_text": window.imu_token_text,
    }


def build_retrieval_record(
    window: instruction_export.TokenizedInstructionWindow,
    index: int,
    references: list[dict[str, str]],
) -> dict[str, Any]:
    metadata = window_metadata(window, index)
    sensor_context = str(metadata["sensor_context"])
    imu_prompt = RETRIEVAL_IMU_PROMPT_TEMPLATE.format(
        imu_token=window.imu_token_text,
        sensor_context=sensor_context,
    )
    positive_texts = [
        {
            "type": item["type"],
            "text": item["text"],
            "text_prompt": RETRIEVAL_TEXT_PROMPT_TEMPLATE.format(text=item["text"]),
        }
        for item in references
    ]
    return {
        "task": "retrieval_eval",
        "query_id": metadata["query_id"],
        "imu_prompt": imu_prompt,
        "imu_token_text": window.imu_token_text,
        "imu_token_ids": metadata["imu_token_ids"],
        "sensor_context": sensor_context,
        "positive_texts": positive_texts,
        "activity_label": metadata["activity_label"],
        "sample_dir": metadata["sample_dir"],
        "subject": metadata["subject"],
        "clip_start": metadata["clip_start"],
        "clip_end": metadata["clip_end"],
        "source_csv_path": metadata["source_csv_path"],
        "source_row_index": metadata["source_row_index"],
        "order_index": metadata["order_index"],
    }


def build_captioning_record(
    window: instruction_export.TokenizedInstructionWindow,
    index: int,
    references: list[dict[str, str]],
) -> dict[str, Any]:
    metadata = window_metadata(window, index)
    answer_text = references[0]["text"]
    prompt = instruction_export.NARRATION_PROMPT_TEMPLATE.format(
        imu_token=window.imu_token_text,
        sensor_context=instruction_export.format_sensor_context(window.sensor_context),
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        "task": "captioning_eval",
        "references": references,
        "activity_label": metadata["activity_label"],
        "sample_dir": metadata["sample_dir"],
        "subject": metadata["subject"],
        "clip_start": metadata["clip_start"],
        "clip_end": metadata["clip_end"],
        "source_csv_path": metadata["source_csv_path"],
        "source_row_index": metadata["source_row_index"],
        "order_index": metadata["order_index"],
        "sensor_context": metadata["sensor_context"],
        "imu_token_ids": metadata["imu_token_ids"],
        "imu_token_text": metadata["imu_token_text"],
    }


def select_retrieval_subset(
    rows: list[dict[str, Any]],
    *,
    samples_per_scenario: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample a deterministic, scenario-balanced retrieval subset."""
    if int(samples_per_scenario) <= 0:
        return [], {}

    rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scenario = str(row.get("sample_dir", "")).strip()
        if not scenario:
            raise ValueError("A retrieval row is missing sample_dir.")
        rows_by_scenario.setdefault(scenario, []).append(row)

    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for scenario in sorted(rows_by_scenario):
        scenario_rows = rows_by_scenario[scenario]
        count = min(int(samples_per_scenario), len(scenario_rows))
        selected.extend(rng.sample(scenario_rows, count))
        counts[scenario] = count
    selected.sort(key=lambda row: int(row.get("order_index", 0)))
    return selected, counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _primary_text_from_egoexo_row(row: dict[str, Any]) -> str:
    for key in ("positive_texts", "references"):
        for item in row.get(key, []) or []:
            if (
                str(item.get("type", "")) == "narration"
                and str(item.get("text", "")).strip()
            ):
                return str(item["text"])
        for item in row.get(key, []) or []:
            if str(item.get("text", "")).strip():
                return str(item["text"])
    if str(row.get("text", "")).strip():
        return str(row["text"])
    raise ValueError(
        f"EgoExo4D row {row.get('query_id', '<unknown>')} has no reference text."
    )


def _egoexo_metadata(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "positive_texts",
        "references",
        "messages",
        "imu_prompt",
        "imu_token_text",
        "imu_token_ids",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def build_egoexo_tokenized_retrieval_record(
    row: dict[str, Any],
    imu_token_ids: list[int],
    imu_token_text: str,
) -> dict[str, Any]:
    metadata = _egoexo_metadata(row)
    sensor_context = str(row.get("sensor_context") or "head")
    references = list(row.get("positive_texts") or [])
    if not references:
        references = [{"type": "narration", "text": _primary_text_from_egoexo_row(row)}]
    positive_texts = [
        {
            "type": str(item.get("type", "narration")),
            "text": str(item["text"]),
            "text_prompt": RETRIEVAL_TEXT_PROMPT_TEMPLATE.format(
                text=str(item["text"])
            ),
        }
        for item in references
        if str(item.get("text", "")).strip()
    ]
    return {
        "task": "retrieval_eval",
        "query_id": str(row["query_id"]),
        "imu_prompt": RETRIEVAL_IMU_PROMPT_TEMPLATE.format(
            imu_token=str(imu_token_text),
            sensor_context=sensor_context,
        ),
        "imu_token_text": str(imu_token_text),
        "imu_token_ids": [int(token_id) for token_id in imu_token_ids],
        "sensor_context": sensor_context,
        "positive_texts": positive_texts,
        "activity_label": str(
            row.get("activity_label") or row.get("parent_task_name") or ""
        ),
        **metadata,
    }


def build_egoexo_tokenized_captioning_record(
    row: dict[str, Any],
    imu_token_ids: list[int],
    imu_token_text: str,
) -> dict[str, Any]:
    metadata = _egoexo_metadata(row)
    sensor_context = str(row.get("sensor_context") or "head")
    references = list(row.get("references") or [])
    if not references:
        references = [{"type": "narration", "text": _primary_text_from_egoexo_row(row)}]
    answer_text = str(references[0]["text"])
    prompt = instruction_export.NARRATION_PROMPT_TEMPLATE.format(
        imu_token=str(imu_token_text),
        sensor_context=sensor_context,
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        "task": "captioning_eval",
        "query_id": str(row["query_id"]),
        "references": [
            {"type": str(item.get("type", "narration")), "text": str(item["text"])}
            for item in references
            if str(item.get("text", "")).strip()
        ],
        "activity_label": str(
            row.get("activity_label") or row.get("parent_task_name") or ""
        ),
        "sensor_context": sensor_context,
        "imu_token_ids": [int(token_id) for token_id in imu_token_ids],
        "imu_token_text": str(imu_token_text),
        **metadata,
    }


def _resolve_egoexo_path(input_dir: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(input_dir) / candidate


def _load_egoexo_window(
    row: dict[str, Any],
    *,
    input_dir: Path,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    npz_path = _resolve_egoexo_path(input_dir, row["imu_window_npz"])
    if npz_path not in cache:
        cache[npz_path] = np.load(npz_path)["windows"]
    windows = cache[npz_path]
    window_index = int(row["window_index"])
    if window_index < 0 or window_index >= int(windows.shape[0]):
        raise IndexError(f"window_index {window_index} is out of range for {npz_path}")
    window = np.asarray(windows[window_index], dtype=np.float32)
    if window.ndim != 2 or window.shape[0] != 6:
        raise ValueError(f"Expected EgoExo4D window shape [6, T], got {window.shape}")
    return window


def _egoexo_graph_sample_from_row(
    row: dict[str, Any],
    *,
    input_dir: Path,
    sample_index: int,
    target_sample_rate_hz: int,
    cache: dict[Path, np.ndarray],
) -> har_export.GraphEvalSample:
    window = _load_egoexo_window(row, input_dir=input_dir, cache=cache)
    resampled = project_data._resample_sequence(
        window,
        original_sample_rate_hz=int(
            row.get("original_sample_rate_hz", project_data.EGOEXO4D_SAMPLE_RATE)
        ),
        target_sample_rate_hz=int(target_sample_rate_hz),
        channel_axis=0,
    )
    graph_x, visible_mask, visible_segments = har_export.build_graph_view_for_sample(
        dataset_name="egoexo4d",
        sample=resampled,
        label_id=0,
        selected_channel_slices=project_data.ALL_IMU_CHANNEL_SLICES,
        source="ego",
        channel_axis=0,
    )
    return har_export.GraphEvalSample(
        dataset_name="egoexo4d",
        source="ego",
        sample_index=int(sample_index),
        label_id=0,
        graph_x=graph_x,
        visible_mask=visible_mask,
        visible_segments=visible_segments,
    )


def _unique_rows_by_query_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id not in unique:
            unique[query_id] = row
    return list(unique.values())


def tokenize_egoexo_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    encoder: torch.nn.Module,
    pqvae: torch.nn.Module,
    device: torch.device,
) -> dict[str, tuple[list[int], str]]:
    input_dir = Path(args.egoexo_eval_input_dir)
    cache: dict[Path, np.ndarray] = {}
    samples = [
        _egoexo_graph_sample_from_row(
            row,
            input_dir=input_dir,
            sample_index=index,
            target_sample_rate_hz=int(args.egoexo_target_sample_rate_hz),
            cache=cache,
        )
        for index, row in enumerate(rows)
    ]
    tokenized = har_export.tokenize_eval_samples(
        samples,
        encoder,
        pqvae,
        device,
        batch_size=int(args.batch_size),
    )
    return {str(row["query_id"]): tokenized[index] for index, row in enumerate(rows)}


def export_egoexo_tokenized_eval_dataset(args: argparse.Namespace) -> dict[str, Any]:
    utils.set_seed(int(args.seed))
    device = torch.device(args.device)
    input_dir = Path(args.egoexo_eval_input_dir or DEFAULT_EGOEXO_EVAL_INPUT_DIR)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows = read_jsonl(input_dir / "retrieval.jsonl")
    captioning_rows = read_jsonl(input_dir / "captioning.jsonl")
    rows_for_tokenization = _unique_rows_by_query_id(
        [*retrieval_rows, *captioning_rows]
    )

    encoder = token_export.train_pqvae.load_frozen_encoder(
        args.encoder_ckpt, device=device, dropout=args.dropout
    )
    pqvae = token_export.load_frozen_pqvae(args.tokenizer_export, device=device)
    token_export.write_codebook_lookup_artifact(
        args.tokenizer_export, output_dir / "imu_codebook_lookup.pt"
    )
    tokens_by_query_id = tokenize_egoexo_rows(
        rows_for_tokenization, args, encoder, pqvae, device
    )

    def token_pair(row: dict[str, Any]) -> tuple[list[int], str]:
        return tokens_by_query_id[str(row["query_id"])]

    tokenized_retrieval = [
        build_egoexo_tokenized_retrieval_record(row, *token_pair(row))
        for row in retrieval_rows
    ]
    tokenized_captioning = [
        build_egoexo_tokenized_captioning_record(row, *token_pair(row))
        for row in captioning_rows
    ]
    write_jsonl(output_dir / "retrieval.jsonl", tokenized_retrieval)
    write_jsonl(output_dir / "captioning.jsonl", tokenized_captioning)

    files = ["retrieval.jsonl", "captioning.jsonl", "imu_codebook_lookup.pt"]
    subset_summaries: dict[str, Any] = {}
    for subset_path in sorted(input_dir.glob("retrieval_*_seed*.jsonl")):
        subset_rows = read_jsonl(subset_path)
        tokenized_subset = [
            build_egoexo_tokenized_retrieval_record(row, *token_pair(row))
            for row in subset_rows
        ]
        write_jsonl(output_dir / subset_path.name, tokenized_subset)
        files.append(subset_path.name)
        counts: dict[str, int] = {}
        for row in subset_rows:
            label = str(row.get("parent_task_name") or row.get("activity_label") or "")
            counts[label] = counts.get(label, 0) + 1
        subset_summaries[subset_path.name] = {
            "num_rows": len(subset_rows),
            "class_counts": counts,
        }

    summary = {
        "mode": "egoexo4d_atomic_tokenized_eval",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "encoder_ckpt": str(args.encoder_ckpt),
        "tokenizer_export": str(args.tokenizer_export),
        "target_sample_rate_hz": int(args.egoexo_target_sample_rate_hz),
        "num_tokenized_windows": len(rows_for_tokenization),
        "exported_retrieval_rows": len(tokenized_retrieval),
        "exported_captioning_rows": len(tokenized_captioning),
        "retrieval_subsets": subset_summaries,
        "files": files,
        "prompt_templates": {
            "captioning": instruction_export.NARRATION_PROMPT_TEMPLATE,
            "retrieval_imu": RETRIEVAL_IMU_PROMPT_TEMPLATE,
            "retrieval_text": RETRIEVAL_TEXT_PROMPT_TEMPLATE,
        },
    }
    utils.save_json(output_dir / "summary.json", summary)
    return summary


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    utils.set_seed(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = token_export.data.load_summary_records(
        args.summary_csv, args.candidate_name, args.base_dir
    )
    held_out_records = filter_held_out_records(records, list(args.held_out_subjects))
    if not held_out_records:
        raise ValueError(
            "No held-out records matched. "
            f"subjects={args.held_out_subjects}, summary_csv={args.summary_csv}, candidate_name={args.candidate_name}"
        )

    encoder = token_export.train_pqvae.load_frozen_encoder(
        args.encoder_ckpt, device=device, dropout=args.dropout
    )
    pqvae = token_export.load_frozen_pqvae(args.tokenizer_export, device=device)
    token_export.write_codebook_lookup_artifact(
        args.tokenizer_export, output_dir / "imu_codebook_lookup.pt"
    )

    specs = instruction_export.build_instruction_window_specs(
        records=held_out_records,
        activity_label_filename=args.activity_label_filename,
        limit_windows=args.limit_windows,
    )
    record_by_sample = {record.sample_dir: record for record in held_out_records}
    windows = instruction_export.tokenize_windows(
        specs, args, record_by_sample, encoder, pqvae, device
    )

    rng = random.Random(int(args.seed) + 104729)
    retrieval_rows: list[dict[str, Any]] = []
    captioning_rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        references = clean_references(window, rng)
        if not references:
            continue
        retrieval_rows.append(build_retrieval_record(window, index, references))
        captioning_rows.append(build_captioning_record(window, index, references))

    write_jsonl(output_dir / "retrieval.jsonl", retrieval_rows)
    write_jsonl(output_dir / "captioning.jsonl", captioning_rows)

    retrieval_subset, retrieval_subset_counts = select_retrieval_subset(
        retrieval_rows,
        samples_per_scenario=int(args.retrieval_samples_per_scenario),
        seed=int(args.seed),
    )
    retrieval_subset_path: Path | None = None
    if retrieval_subset:
        retrieval_subset_path = (
            output_dir / f"retrieval_{len(retrieval_subset)}_seed{int(args.seed)}.jsonl"
        )
        write_jsonl(retrieval_subset_path, retrieval_subset)

    subjects = sorted(
        {subject_from_sample_dir(record.sample_dir) for record in held_out_records}
    )
    summary = {
        "output_dir": str(output_dir),
        "summary_csv": str(args.summary_csv),
        "candidate_name": str(args.candidate_name),
        "held_out_subjects": list(args.held_out_subjects),
        "matched_subjects": subjects,
        "held_out_samples": len(held_out_records),
        "text_windows": len(specs),
        "exported_retrieval_rows": len(retrieval_rows),
        "exported_captioning_rows": len(captioning_rows),
        "retrieval_subset_jsonl": (
            str(retrieval_subset_path) if retrieval_subset_path is not None else None
        ),
        "retrieval_subset_counts": retrieval_subset_counts,
        "tokenizer_export": str(args.tokenizer_export),
        "encoder_ckpt": str(args.encoder_ckpt),
        "activity_label_filename": str(args.activity_label_filename),
        "max_visible_nodes": int(args.max_visible_nodes),
        "files": [
            "retrieval.jsonl",
            "captioning.jsonl",
            "imu_codebook_lookup.pt",
            *([retrieval_subset_path.name] if retrieval_subset_path is not None else []),
        ],
        "prompt_templates": {
            "captioning": instruction_export.NARRATION_PROMPT_TEMPLATE,
            "retrieval_imu": RETRIEVAL_IMU_PROMPT_TEMPLATE,
            "retrieval_text": RETRIEVAL_TEXT_PROMPT_TEMPLATE,
        },
    }
    utils.save_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.egoexo_eval_input_dir is not None:
        summary = export_egoexo_tokenized_eval_dataset(args)
    else:
        summary = export_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
