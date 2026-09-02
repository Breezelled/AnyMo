from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import config
from exporters import tokens as token_export
import imu_token_utils
import training_utils as utils


DEFAULT_OUTPUT_DIR = config.INSTRUCTION_DATA_DIR
DEFAULT_CLASS_POOL_TXT = (
    Path(__file__).resolve().parents[2] / "metadata" / "activity_labels_180.txt"
)

NARRATION_PROMPT_TEMPLATE = (
    "Describe the human motion represented by the wearable IMU motion tokens.\n\n"
    "The IMU tokens are from IMU sensors attached to the user's {sensor_context}.\n\n"
    "Input IMU token:\n{imu_token}"
)
TEXT_TO_IMU_PROMPT_TEMPLATE = (
    "Generate the wearable IMU motion tokens that represent the described human motion.\n\n"
    "The IMU tokens are from IMU sensors attached to the user's {sensor_context}.\n\n"
    "Motion description:\n{motion_text}"
)
HAR_MCQ_PROMPT_TEMPLATE = (
    "Recognize the activity represented by the wearable IMU motion tokens.\n\n"
    "The IMU tokens are from IMU sensors attached to the user's {sensor_context}.\n\n"
    "Input IMU token:\n{imu_token}\n\n"
    "CHOICES:\n{choices}\n\n"
    "Choose the best matching option. Output the option key followed by the selected activity label."
)

UPPER_ACTOR_REPLACEMENTS = ("A person", "A human", "An individual")
LOWER_ACTOR_REPLACEMENTS = ("a person", "a human", "an individual")
ACTOR_MARKER_RE = re.compile(r"(?<![A-Za-z])C(?![A-Za-z])")


@dataclass(frozen=True)
class InstructionWindowSpec:
    sample_dir: str
    clip_start: int
    clip_end: int
    label_name: str
    order_index: int
    primary_text: str
    augmented_texts: tuple[str, ...]
    activity_label: str
    source_csv_path: str | None = None
    source_row_index: int | None = None

    @property
    def window_size(self) -> int:
        return int(self.clip_end) - int(self.clip_start) + 1

    @property
    def script(self) -> str:
        return self.primary_text or self.label_name


@dataclass(frozen=True)
class TokenizedInstructionWindow:
    spec: InstructionWindowSpec
    imu_token_ids: list[int]
    imu_token_text: str
    sensor_context: str


def build_argparser() -> argparse.ArgumentParser:
    parser = token_export.build_argparser()
    parser.description = "Export IMU-token instruction tuning data for narration and HAR MCQ tasks."
    parser.add_argument(
        "--class-pool-txt",
        type=Path,
        default=DEFAULT_CLASS_POOL_TXT,
        help="Final AnyMo-180 activity-label vocabulary.",
    )
    parser.add_argument("--narration-repeats", type=int, default=3)
    parser.add_argument("--mcq-repeats", type=int, default=3)
    parser.add_argument("--mcq-min-choices", type=int, default=35)
    parser.add_argument("--mcq-max-choices", type=int, default=35)
    parser.add_argument(
        "--contrastive-repeats",
        type=int,
        default=6,
        help=(
            "Number of contrastive rows to export per tokenized window. Each row keeps "
            "the primary narration and samples one non-empty augmented narration when available."
        ),
    )
    parser.add_argument(
        "--skip-lm-export",
        action="store_true",
        help="Only write train_contrastive.jsonl and skip train.jsonl generation.",
    )
    parser.add_argument(
        "--only-text-to-imu-export",
        action="store_true",
        help="Only write train_text_to_imu.jsonl and skip train/train_contrastive generation.",
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Optional debug limit after text-aligned windows are built.",
    )
    parser.set_defaults(alignment_mode="text_labels")
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def strip_outer_quotes(text: str) -> str:
    text = clean_text(text)
    quote_pairs = {('"', '"'), ("'", "'"), ("“", "”")}
    while len(text) >= 2 and (text[0], text[-1]) in quote_pairs:
        text = text[1:-1].strip()
    return text


def _is_sentence_initial_actor(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    if not prefix:
        return True
    return prefix[-1] in ".!?\n"


def clean_narration_text(text: str, rng: random.Random) -> str:
    text = strip_outer_quotes(text)

    def replace_match(match: re.Match[str]) -> str:
        if _is_sentence_initial_actor(text, match.start()):
            return rng.choice(UPPER_ACTOR_REPLACEMENTS)
        return rng.choice(LOWER_ACTOR_REPLACEMENTS)

    return ACTOR_MARKER_RE.sub(replace_match, text)


def load_class_pool(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    deduped = list(dict.fromkeys(labels))
    if not deduped:
        raise ValueError(f"No class labels found in {path}")
    return deduped


def build_option_keys(num_options: int) -> list[str]:
    if num_options < 0:
        raise ValueError(f"num_options must be non-negative, got {num_options}")
    letters = "abcdefghijklmnopqrstuvwxyz"
    keys: list[str] = []
    for index in range(num_options):
        letter = letters[index % len(letters)]
        repeat = index // len(letters) + 1
        keys.append(letter * repeat)
    return keys


def build_mcq_choices(
    activity_label: str,
    class_pool: list[str],
    rng: random.Random,
    min_choices: int,
    max_choices: int,
) -> tuple[list[dict[str, str]], str]:
    activity_label = clean_text(activity_label)
    if not activity_label:
        raise ValueError("activity_label must be non-empty")
    unique_pool = list(dict.fromkeys(clean_text(label) for label in class_pool if clean_text(label)))
    negative_pool = [label for label in unique_pool if label != activity_label]
    if not negative_pool:
        raise ValueError("class_pool must contain at least one negative label")

    max_allowed = min(max(1, int(max_choices)), len(negative_pool) + 1)
    min_allowed = min(max(1, int(min_choices)), max_allowed)
    num_choices = rng.randint(min_allowed, max_allowed)
    labels = rng.sample(negative_pool, num_choices - 1) + [activity_label]
    rng.shuffle(labels)

    keys = build_option_keys(len(labels))
    choices = [{"key": key, "label": label} for key, label in zip(keys, labels)]
    answer_key = choices[labels.index(activity_label)]["key"]
    return choices, answer_key


def format_choices(choices: list[dict[str, str]]) -> str:
    return "\n".join(f"{choice['key']}: {choice['label']}" for choice in choices)


def _iter_sensor_context_names(segment_names: object) -> list[str]:
    if isinstance(segment_names, str):
        text = clean_text(segment_names)
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text] if text else []
    return [clean_text(name) for name in segment_names if clean_text(name)]


def format_sensor_context(segment_names: object) -> str:
    names = [format_segment_name_for_prompt(name) for name in _iter_sensor_context_names(segment_names)]
    if not names:
        return "visible body segments"
    return ", ".join(names)


def format_segment_name_for_prompt(segment_name: object) -> str:
    raw_name = clean_text(segment_name)
    if raw_name.startswith("L_"):
        name = "Left " + raw_name[2:]
    elif raw_name.startswith("R_"):
        name = "Right " + raw_name[2:]
    else:
        name = raw_name
    name = name.replace("_", " ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def build_har_answer_text(answer_key: str, activity_label: str) -> str:
    return f"{clean_text(answer_key)}: {clean_text(activity_label)}"


def _base_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def build_narration_record(
    imu_token_text: str,
    narration: str,
    sensor_context: str = "visible body segments",
    metadata: dict[str, Any] | None = None,
    narration_source: str = "primary",
) -> dict[str, Any]:
    record = {
        "messages": [
            {
                "role": "user",
                "content": NARRATION_PROMPT_TEMPLATE.format(
                    imu_token=imu_token_text,
                    sensor_context=format_sensor_context(sensor_context),
                ),
            },
            {"role": "assistant", "content": narration},
        ],
        "task": "narration",
        "narration_source": narration_source,
    }
    record.update(_base_metadata(metadata))
    return record


def build_text_to_imu_record(
    imu_token_text: str,
    motion_text: str,
    sensor_context: str = "visible body segments",
    metadata: dict[str, Any] | None = None,
    narration_source: str = "primary",
) -> dict[str, Any]:
    record = {
        "messages": [
            {
                "role": "user",
                "content": TEXT_TO_IMU_PROMPT_TEMPLATE.format(
                    motion_text=motion_text,
                    sensor_context=format_sensor_context(sensor_context),
                ),
            },
            {"role": "assistant", "content": imu_token_text},
        ],
        "task": "text_to_imu",
        "narration_source": narration_source,
    }
    record.update(_base_metadata(metadata))
    return record


def build_har_mcq_record(
    imu_token_text: str,
    activity_label: str,
    class_pool: list[str],
    rng: random.Random,
    min_choices: int,
    max_choices: int,
    sensor_context: str = "visible body segments",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choices, answer_key = build_mcq_choices(
        activity_label=activity_label,
        class_pool=class_pool,
        rng=rng,
        min_choices=min_choices,
        max_choices=max_choices,
    )
    record = {
        "messages": [
            {
                "role": "user",
                "content": HAR_MCQ_PROMPT_TEMPLATE.format(
                    imu_token=imu_token_text,
                    sensor_context=format_sensor_context(sensor_context),
                    choices=format_choices(choices),
                ),
            },
            {"role": "assistant", "content": build_har_answer_text(answer_key, activity_label)},
        ],
        "task": "har_mcq",
        "activity_label": clean_text(activity_label),
        "choices": choices,
        "answer_key": answer_key,
        "answer_label": clean_text(activity_label),
        "answer_text": build_har_answer_text(answer_key, activity_label),
    }
    record.update(_base_metadata(metadata))
    return record


def strip_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {"messages": record["messages"], "task": record["task"]}


def build_contrastive_records(
    windows: list[TokenizedInstructionWindow],
    seed: int,
    contrastive_repeats: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed) + 7919)
    repeats = max(0, int(contrastive_repeats))
    rows: list[dict[str, Any]] = []
    for window in windows:
        primary_text = clean_narration_text(window.spec.primary_text, rng) if clean_text(window.spec.primary_text) else ""
        augmented_texts = [clean_text(text) for text in window.spec.augmented_texts if clean_text(text)]
        rng.shuffle(augmented_texts)
        reserve_narration_only = bool(primary_text) and repeats > 1
        augmented_repeat_count = repeats - 1 if reserve_narration_only else repeats
        for repeat_index in range(repeats):
            positive_texts: list[dict[str, str]] = []
            if primary_text:
                positive_texts.append({"type": "narration", "text": primary_text})
            if augmented_texts and repeat_index < augmented_repeat_count:
                augmented_source = augmented_texts[repeat_index % len(augmented_texts)]
                augmented_text = clean_narration_text(augmented_source, rng)
                if augmented_text:
                    positive_texts.append({"type": "augmented_narration", "text": augmented_text})
            if not positive_texts:
                continue
            rows.append(
                {
                    "task": "contrastive",
                    "imu_token_text": window.imu_token_text,
                    "sensor_context": format_sensor_context(window.sensor_context),
                    "activity_label": clean_text(window.spec.activity_label),
                    "positive_texts": positive_texts,
                }
            )
    rng.shuffle(rows)
    return rows


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return DEFAULT_OUTPUT_DIR


def build_instruction_window_specs(
    records: list[token_export.data.SampleRecord],
    activity_label_filename: str = token_export.DEFAULT_ACTIVITY_LABEL_FILENAME,
    limit_windows: int | None = None,
) -> list[InstructionWindowSpec]:
    specs: list[InstructionWindowSpec] = []
    order_index = 0
    for record in records:
        csv_path = Path(record.sample_path) / "multimodal_sync_60hz" / activity_label_filename
        raw_time_axis = token_export._load_npz_time_axis(Path(record.real_npz))
        synth_time_axis = token_export._load_synth_time_axis(Path(record.synth_npz))
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader):
                clip_start = int(row["start_idx"])
                clip_end = int(row["end_idx"])
                synth_clip_start, synth_clip_end = token_export.remap_text_indices_to_synth_time_axis(
                    clip_start=clip_start,
                    clip_end=clip_end,
                    raw_time_axis=raw_time_axis,
                    synth_time_axis=synth_time_axis,
                )
                primary_text = clean_text(row.get("Describe my atomic actions"))
                augmented_texts = tuple(
                    clean_text(row.get(f"augmented_atomic_action_{index}"))
                    for index in range(1, 6)
                    if clean_text(row.get(f"augmented_atomic_action_{index}"))
                )
                specs.append(
                    InstructionWindowSpec(
                        sample_dir=record.sample_dir,
                        clip_start=synth_clip_start,
                        clip_end=synth_clip_end,
                        label_name=record.label_name,
                        order_index=order_index,
                        primary_text=primary_text,
                        augmented_texts=augmented_texts,
                        activity_label=clean_text(row.get("activity_label")),
                        source_csv_path=str(csv_path),
                        source_row_index=row_index,
                    )
                )
                order_index += 1
                if limit_windows is not None and len(specs) >= int(limit_windows):
                    return specs
    return specs


def _metadata_for_window(window: TokenizedInstructionWindow) -> dict[str, Any]:
    spec = window.spec
    metadata = {
        "imu_token_ids": [int(token_id) for token_id in window.imu_token_ids],
        "sample_dir": spec.sample_dir,
        "clip_start": int(spec.clip_start),
        "clip_end": int(spec.clip_end),
        "source_csv_path": spec.source_csv_path,
        "source_row_index": spec.source_row_index,
    }
    return metadata


def _visible_segment_names_for_spec(spec: InstructionWindowSpec, args: argparse.Namespace) -> list[str]:
    clip_key = token_export.data.ClipKey(
        sample_dir=spec.sample_dir,
        clip_start=int(spec.clip_start),
        label_name=spec.label_name,
    )
    mask = imu_token_utils.build_deterministic_visible_mask(
        clip_key=clip_key,
        seed=int(args.seed),
        num_nodes=len(token_export.data.SEGMENT_NAMES),
        max_visible_nodes=int(args.max_visible_nodes),
    )
    return [token_export.data.SEGMENT_NAMES[index] for index, is_visible in enumerate(mask.tolist()) if bool(is_visible)]


def _sample_narration_text(spec: InstructionWindowSpec, rng: random.Random, force_primary: bool) -> tuple[str, str]:
    if force_primary or not spec.augmented_texts or rng.random() < 0.5:
        return spec.primary_text, "primary"
    return rng.choice(spec.augmented_texts), "augmented"


def build_instruction_records(
    windows: list[TokenizedInstructionWindow],
    class_pool: list[str],
    seed: int,
    narration_repeats: int,
    mcq_repeats: int,
    mcq_min_choices: int,
    mcq_max_choices: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    narration_rows: list[dict[str, Any]] = []
    for window in windows:
        for repeat_index in range(int(narration_repeats)):
            text, source = _sample_narration_text(window.spec, rng, force_primary=(repeat_index == 0))
            text = clean_narration_text(text, rng)
            if not text:
                continue
            metadata = _metadata_for_window(window)
            metadata["narration_repeat_index"] = repeat_index
            narration_rows.append(
                build_narration_record(
                    imu_token_text=window.imu_token_text,
                    narration=text,
                    sensor_context=window.sensor_context,
                    metadata=metadata,
                    narration_source=source,
                )
            )

    labeled_windows = [window for window in windows if clean_text(window.spec.activity_label)]
    if narration_rows and not labeled_windows:
        raise ValueError("No windows with activity_label were found for HAR MCQ task")

    mcq_rows: list[dict[str, Any]] = []
    for window in labeled_windows:
        for repeat_index in range(int(mcq_repeats)):
            metadata = _metadata_for_window(window)
            metadata["mcq_repeat_index"] = repeat_index
            mcq_rows.append(
                build_har_mcq_record(
                    imu_token_text=window.imu_token_text,
                    activity_label=window.spec.activity_label,
                    class_pool=class_pool,
                    rng=rng,
                    min_choices=mcq_min_choices,
                    max_choices=mcq_max_choices,
                    sensor_context=window.sensor_context,
                    metadata=metadata,
                )
            )

    while len(mcq_rows) < len(narration_rows):
        window = rng.choice(labeled_windows)
        metadata = _metadata_for_window(window)
        metadata["mcq_repeat_index"] = "balanced_resample"
        mcq_rows.append(
            build_har_mcq_record(
                imu_token_text=window.imu_token_text,
                activity_label=window.spec.activity_label,
                class_pool=class_pool,
                rng=rng,
                min_choices=mcq_min_choices,
                max_choices=mcq_max_choices,
                sensor_context=window.sensor_context,
                metadata=metadata,
            )
        )
    if len(mcq_rows) > len(narration_rows):
        rng.shuffle(mcq_rows)
        mcq_rows = mcq_rows[: len(narration_rows)]

    rows = narration_rows + mcq_rows
    rng.shuffle(rows)
    return rows


def build_text_to_imu_records(
    windows: list[TokenizedInstructionWindow],
    seed: int,
    text_to_imu_repeats: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    for window in windows:
        for repeat_index in range(int(text_to_imu_repeats)):
            text, source = _sample_narration_text(window.spec, rng, force_primary=(repeat_index == 0))
            text = clean_narration_text(text, rng)
            if not text:
                continue
            metadata = _metadata_for_window(window)
            metadata["text_to_imu_repeat_index"] = repeat_index
            rows.append(
                build_text_to_imu_record(
                    imu_token_text=window.imu_token_text,
                    motion_text=text,
                    sensor_context=window.sensor_context,
                    metadata=metadata,
                    narration_source=source,
                )
            )
    rng.shuffle(rows)
    return rows


def tokenize_windows(
    specs: list[InstructionWindowSpec],
    args: argparse.Namespace,
    record_by_sample: dict[str, token_export.data.SampleRecord],
    encoder: token_export.model.STGCNEncoder,
    pqvae: token_export.model.MotionPQVAE,
    device: torch.device,
) -> list[TokenizedInstructionWindow]:
    token_rows = token_export.export_window_specs(
        split_name="instruction",
        specs=specs,
        args=args,
        record_by_sample=record_by_sample,
        encoder=encoder,
        pqvae=pqvae,
        device=device,
    )
    ordered_specs = sorted(specs, key=lambda spec: spec.order_index)
    if len(token_rows) != len(ordered_specs):
        raise RuntimeError(f"Token row count {len(token_rows)} != spec count {len(ordered_specs)}")

    windows: list[TokenizedInstructionWindow] = []
    for spec, row in zip(ordered_specs, token_rows):
        if row.get("source_row_index") != spec.source_row_index or row.get("source_csv_path") != spec.source_csv_path:
            raise RuntimeError(
                "Token row/spec alignment mismatch: "
                f"row source={row.get('source_csv_path')}:{row.get('source_row_index')} "
                f"spec source={spec.source_csv_path}:{spec.source_row_index}"
            )
        windows.append(
            TokenizedInstructionWindow(
                spec=spec,
                imu_token_ids=[int(token_id) for token_id in row["imu_token_ids"]],
                imu_token_text=str(row["messages"][0]["content"]),
                sensor_context=format_sensor_context(_visible_segment_names_for_spec(spec, args)),
            )
        )
    return windows


def write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    utils.set_seed(args.seed)
    class_pool = load_class_pool(Path(args.class_pool_txt))
    records = token_export.data.exclude_held_out_subjects(
        token_export.data.load_summary_records(args.summary_csv, args.candidate_name, args.base_dir)
    )
    encoder = token_export.train_pqvae.load_frozen_encoder(args.encoder_ckpt, device=device, dropout=args.dropout)
    pqvae = token_export.load_frozen_pqvae(args.tokenizer_export, device=device)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_export.write_codebook_lookup_artifact(args.tokenizer_export, output_dir / "imu_codebook_lookup.pt")

    specs = build_instruction_window_specs(
        records=records,
        activity_label_filename=args.activity_label_filename,
        limit_windows=args.limit_windows,
    )
    record_by_sample = {record.sample_dir: record for record in records}
    windows = tokenize_windows(specs, args, record_by_sample, encoder, pqvae, device)
    if bool(args.only_text_to_imu_export):
        text_to_imu_rows = build_text_to_imu_records(
            windows=windows,
            seed=args.seed,
            text_to_imu_repeats=args.narration_repeats,
        )
        output_rows = [strip_record_metadata(row) for row in text_to_imu_rows]
        write_jsonl_records(output_dir / "train_text_to_imu.jsonl", output_rows)
        summary = {
            "train_rows": 0,
            "contrastive_rows": 0,
            "text_to_imu_rows": len(text_to_imu_rows),
            "skip_lm_export": True,
            "only_text_to_imu_export": True,
            "task_counts": {"text_to_imu": len(text_to_imu_rows)},
            "text_windows": len(specs),
            "labeled_windows": sum(1 for window in windows if clean_text(window.spec.activity_label)),
            "tokenizer_export": str(args.tokenizer_export),
            "encoder_ckpt": str(args.encoder_ckpt),
            "activity_label_filename": str(args.activity_label_filename),
            "max_visible_nodes": int(args.max_visible_nodes),
            "text_to_imu_repeats": int(args.narration_repeats),
        }
        utils.save_json(output_dir / "summary.json", summary)
        return summary

    rows: list[dict[str, Any]] = []
    if not bool(args.skip_lm_export):
        rows = build_instruction_records(
            windows=windows,
            class_pool=class_pool,
            seed=args.seed,
            narration_repeats=args.narration_repeats,
            mcq_repeats=args.mcq_repeats,
            mcq_min_choices=args.mcq_min_choices,
            mcq_max_choices=args.mcq_max_choices,
        )
        output_rows = [strip_record_metadata(row) for row in rows]
        write_jsonl_records(output_dir / "train.jsonl", output_rows)
    contrastive_rows = build_contrastive_records(
        windows=windows,
        seed=args.seed,
        contrastive_repeats=args.contrastive_repeats,
    )
    write_jsonl_records(output_dir / "train_contrastive.jsonl", contrastive_rows)

    task_counts: dict[str, int] = {}
    for row in rows:
        task = str(row["task"])
        task_counts[task] = task_counts.get(task, 0) + 1
    contrastive_positive_type_counts: dict[str, int] = {}
    for row in contrastive_rows:
        for item in row.get("positive_texts", []):
            positive_type = str(item.get("type", ""))
            contrastive_positive_type_counts[positive_type] = contrastive_positive_type_counts.get(positive_type, 0) + 1
    summary = {
        "train_rows": len(rows),
        "contrastive_rows": len(contrastive_rows),
        "skip_lm_export": bool(args.skip_lm_export),
        "task_counts": task_counts,
        "contrastive_positive_type_counts": contrastive_positive_type_counts,
        "text_windows": len(specs),
        "labeled_windows": sum(1 for window in windows if clean_text(window.spec.activity_label)),
        "class_pool_size": len(class_pool),
        "tokenizer_export": str(args.tokenizer_export),
        "encoder_ckpt": str(args.encoder_ckpt),
        "activity_label_filename": str(args.activity_label_filename),
        "max_visible_nodes": int(args.max_visible_nodes),
        "narration_repeats": int(args.narration_repeats),
        "mcq_repeats": int(args.mcq_repeats),
        "mcq_min_choices": int(args.mcq_min_choices),
        "mcq_max_choices": int(args.mcq_max_choices),
        "contrastive_repeats": int(args.contrastive_repeats),
    }
    utils.save_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = export_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
