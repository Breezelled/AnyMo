from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_EGOEXO4D_ROOT = Path(os.environ.get("EGOEXO4D_ROOT", "<PATH_TO_EGOEXO4D>"))
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("ANYMO_OUTPUT_ROOT", Path(__file__).resolve().parents[2] / "outputs")
) / "egoexo4d_prepared"
DEFAULT_PARENT_TASKS = (
    "Rock Climbing",
    "Basketball",
    "Dance",
    "Cooking",
    "Health",
    "Bike Repair",
    "Soccer",
    "Music",
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class AtomicExample:
    query_id: str
    take_uid: str
    take_name: str
    parent_task_name: str
    task_name: str
    text: str
    original_text: str
    timestamp_sec: float
    subject: str
    annotation_uid: str
    annotation_split: str
    imu_path: str
    timestamp_path: str | None
    window_index: int
    center_index: int
    window_start_index: int
    window_end_index: int
    pad_left: int
    pad_right: int
    window_start_sec: float
    window_end_sec: float
    original_sample_rate_hz: int
    window_seconds: float
    imu_window_npz: str = "imu_windows.npz"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a class-balanced EgoExo4D atomic-description zero-shot "
            "cross-modal retrieval/captioning dataset."
        )
    )
    parser.add_argument("--egoexo4d-root", type=Path, default=DEFAULT_EGOEXO4D_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotation-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--parent-tasks", nargs="+", default=list(DEFAULT_PARENT_TASKS))
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate-hz", type=int, default=200)
    parser.add_argument(
        "--retrieval-subset-size",
        type=int,
        default=100,
        help="Also write a class-balanced retrieval_<N>_seed<seed>.jsonl subset. Use 0 to disable.",
    )
    parser.add_argument("--no-normalize-text", action="store_true")
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_balanced_subset(
    examples: list[AtomicExample],
    subset_size: int,
    seed: int,
    class_order: list[str] | None = None,
) -> list[AtomicExample]:
    if subset_size <= 0 or not examples:
        return []
    rng = random.Random(int(seed))
    groups: dict[str, list[AtomicExample]] = {}
    for example in examples:
        groups.setdefault(example.parent_task_name, []).append(example)
    ordered_classes = [cls for cls in (class_order or sorted(groups)) if cls in groups]
    if not ordered_classes:
        return []

    base_quota, remainder = divmod(int(subset_size), len(ordered_classes))
    selected: list[AtomicExample] = []
    used_ids: set[str] = set()
    for class_index, class_name in enumerate(ordered_classes):
        class_examples = list(groups[class_name])
        rng.shuffle(class_examples)
        quota = base_quota + (1 if class_index < remainder else 0)
        for example in class_examples[:quota]:
            selected.append(example)
            used_ids.add(example.query_id)

    if len(selected) < subset_size:
        leftovers = [example for example in examples if example.query_id not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: subset_size - len(selected)])

    rng.shuffle(selected)
    return selected[:subset_size]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(str(text)))


def normalize_atomic_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\b[Cc]'s\b", "the person's", text)
    text = re.sub(r"\b[Mm]an\s+X\b", "another person", text)
    text = re.sub(r"\b[Ww]oman\s+X\b", "another person", text)
    text = re.sub(r"\b[Cc]\b", "the person", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("the person"):
        text = "the person" + text[len("the person") :]
    return text


def _load_take_maps(root: Path) -> dict[str, dict[str, Any]]:
    takes = read_json(Path(root) / "takes.json")
    if not isinstance(takes, list):
        raise ValueError(f"Expected takes.json to contain a list, got {type(takes)!r}")
    return {
        str(take["take_uid"]): dict(take)
        for take in takes
        if take.get("take_uid") and take.get("take_name")
    }


def _annotation_path(root: Path, split: str) -> Path:
    return Path(root) / "annotations" / f"atomic_descriptions_{split}.json"


def _nearest_index(timestamps_ns: np.ndarray, center_sec: float, fallback_sample_rate_hz: int) -> int:
    if timestamps_ns.size == 0:
        return max(0, int(round(float(center_sec) * float(fallback_sample_rate_hz))))
    rel_sec = (timestamps_ns.astype(np.float64) - float(timestamps_ns[0])) / 1e9
    index = int(np.searchsorted(rel_sec, float(center_sec), side="left"))
    if index <= 0:
        return 0
    if index >= len(rel_sec):
        return len(rel_sec) - 1
    before = float(rel_sec[index - 1])
    after = float(rel_sec[index])
    return index - 1 if abs(float(center_sec) - before) <= abs(after - float(center_sec)) else index


def extract_centered_window(
    imu: np.ndarray,
    timestamps_ns: np.ndarray | None,
    *,
    center_sec: float,
    sample_rate_hz: int,
    window_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(imu, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 6:
        raise ValueError(f"Expected EgoExo4D IMU shape [6, T], got {arr.shape}")
    total_steps = int(arr.shape[1])
    window_steps = int(round(float(window_seconds) * float(sample_rate_hz)))
    if window_steps <= 0:
        raise ValueError(f"window_seconds and sample_rate_hz imply non-positive window length: {window_steps}")
    timestamps_arr = np.asarray(timestamps_ns, dtype=np.float64) if timestamps_ns is not None else np.asarray([])
    center_index = _nearest_index(timestamps_arr, float(center_sec), int(sample_rate_hz))

    half = window_steps // 2
    unclamped_start = center_index - half
    if total_steps >= window_steps:
        start = min(max(0, unclamped_start), total_steps - window_steps)
        end = start + window_steps
        pad_left = 0
        pad_right = 0
        window = arr[:, start:end]
    else:
        start = 0
        end = total_steps
        pad_left = max(0, -unclamped_start)
        pad_left = min(pad_left, window_steps - total_steps)
        pad_right = window_steps - total_steps - pad_left
        window = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant")

    if window.shape != (6, window_steps):
        raise RuntimeError(f"Internal error: expected window shape (6, {window_steps}), got {window.shape}")

    return window.astype(np.float32, copy=False), {
        "center_index": int(center_index),
        "window_start_index": int(start),
        "window_end_index": int(end),
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
        "window_start_sec": float(start) / float(sample_rate_hz),
        "window_end_sec": float(end) / float(sample_rate_hz),
    }


def _iter_atomic_candidates(
    *,
    root: Path,
    annotation_splits: list[str],
    parent_tasks: set[str],
    min_words: int,
    normalize_text: bool,
) -> Iterable[dict[str, Any]]:
    take_by_uid = _load_take_maps(root)
    processed_dir = Path(root) / "processed_imu"
    for split in annotation_splits:
        annotation_file = _annotation_path(root, split)
        payload = read_json(annotation_file)
        annotations = payload.get("annotations", {})
        if not isinstance(annotations, dict):
            raise ValueError(f"Expected annotations dict in {annotation_file}")
        for take_uid, jobs in annotations.items():
            take = take_by_uid.get(str(take_uid))
            if not take:
                continue
            parent_task_name = str(take.get("parent_task_name") or "")
            if parent_task_name not in parent_tasks:
                continue
            take_name = str(take["take_name"])
            imu_path = processed_dir / f"{take_name}.npy"
            timestamp_path = processed_dir / f"{take_name}_timestamps.npy"
            if not imu_path.exists():
                continue
            for job in jobs or []:
                annotation_uid = str(job.get("annotation_uid", ""))
                for description in job.get("descriptions", []) or []:
                    if str(description.get("subject", "")) != "C":
                        continue
                    if description.get("unsure") is True:
                        continue
                    original_text = str(description.get("text", "")).strip()
                    if not original_text:
                        continue
                    text = normalize_atomic_text(original_text) if normalize_text else original_text
                    if word_count(text) < int(min_words):
                        continue
                    if description.get("timestamp") is None:
                        continue
                    yield {
                        "take_uid": str(take_uid),
                        "take_name": take_name,
                        "parent_task_name": parent_task_name,
                        "task_name": str(take.get("task_name") or parent_task_name),
                        "original_text": original_text,
                        "text": text,
                        "timestamp_sec": float(description["timestamp"]),
                        "subject": str(description.get("subject", "")),
                        "annotation_uid": annotation_uid,
                        "annotation_split": str(split),
                        "imu_path": str(imu_path),
                        "timestamp_path": str(timestamp_path) if timestamp_path.exists() else None,
                    }


def _sample_balanced(
    candidates: list[dict[str, Any]],
    *,
    parent_tasks: list[str],
    samples_per_class: int,
    seed: int,
    require_full_balanced: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    by_class: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in parent_tasks}
    for item in candidates:
        class_name = str(item["parent_task_name"])
        if class_name in by_class:
            by_class[class_name].append(item)

    class_counts_before = {class_name: len(rows) for class_name, rows in by_class.items()}
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    class_counts_after: dict[str, int] = {}
    for class_name in parent_tasks:
        rows = list(by_class[class_name])
        rng.shuffle(rows)
        if len(rows) < int(samples_per_class) and require_full_balanced:
            raise ValueError(
                f"Not enough EgoExo4D atomic rows for class {class_name!r}: "
                f"need {samples_per_class}, got {len(rows)}"
            )
        chosen = rows[: min(int(samples_per_class), len(rows))]
        class_counts_after[class_name] = len(chosen)
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected, class_counts_before, class_counts_after


def build_atomic_examples(
    *,
    root: Path,
    annotation_splits: list[str],
    samples_per_class: int,
    min_words: int,
    seed: int,
    normalize_text: bool,
    require_full_balanced: bool,
    parent_tasks: list[str] | None = None,
    window_seconds: float = 5.0,
    sample_rate_hz: int = 200,
) -> tuple[list[AtomicExample], dict[str, Any]]:
    root = Path(root)
    parent_tasks = list(parent_tasks or DEFAULT_PARENT_TASKS)
    candidates = list(
        _iter_atomic_candidates(
            root=root,
            annotation_splits=list(annotation_splits),
            parent_tasks=set(parent_tasks),
            min_words=int(min_words),
            normalize_text=bool(normalize_text),
        )
    )
    selected, counts_before, counts_after = _sample_balanced(
        candidates,
        parent_tasks=parent_tasks,
        samples_per_class=int(samples_per_class),
        seed=int(seed),
        require_full_balanced=bool(require_full_balanced),
    )

    examples: list[AtomicExample] = []
    for window_index, item in enumerate(selected):
        imu_path = Path(str(item["imu_path"]))
        timestamp_path = Path(str(item["timestamp_path"])) if item.get("timestamp_path") else None
        imu = np.asarray(np.load(imu_path), dtype=np.float32)
        timestamps = np.asarray(np.load(timestamp_path), dtype=np.float64) if timestamp_path and timestamp_path.exists() else None
        _, window_meta = extract_centered_window(
            imu,
            timestamps,
            center_sec=float(item["timestamp_sec"]),
            sample_rate_hz=int(sample_rate_hz),
            window_seconds=float(window_seconds),
        )
        query_id = f"{item['take_name']}:{item['annotation_uid']}:{window_index}"
        examples.append(
            AtomicExample(
                query_id=query_id,
                take_uid=str(item["take_uid"]),
                take_name=str(item["take_name"]),
                parent_task_name=str(item["parent_task_name"]),
                task_name=str(item["task_name"]),
                text=str(item["text"]),
                original_text=str(item["original_text"]),
                timestamp_sec=float(item["timestamp_sec"]),
                subject=str(item["subject"]),
                annotation_uid=str(item["annotation_uid"]),
                annotation_split=str(item["annotation_split"]),
                imu_path=str(imu_path),
                timestamp_path=str(timestamp_path) if timestamp_path else None,
                window_index=int(window_index),
                center_index=int(window_meta["center_index"]),
                window_start_index=int(window_meta["window_start_index"]),
                window_end_index=int(window_meta["window_end_index"]),
                pad_left=int(window_meta["pad_left"]),
                pad_right=int(window_meta["pad_right"]),
                window_start_sec=float(window_meta["window_start_sec"]),
                window_end_sec=float(window_meta["window_end_sec"]),
                original_sample_rate_hz=int(sample_rate_hz),
                window_seconds=float(window_seconds),
            )
        )

    summary = {
        "egoexo4d_root": str(root),
        "annotation_splits": list(annotation_splits),
        "parent_tasks": parent_tasks,
        "samples_per_class": int(samples_per_class),
        "min_words": int(min_words),
        "seed": int(seed),
        "normalize_text": bool(normalize_text),
        "window_seconds": float(window_seconds),
        "sample_rate_hz": int(sample_rate_hz),
        "candidate_rows": len(candidates),
        "num_samples": len(examples),
        "class_counts_before_sampling": counts_before,
        "class_counts_after_sampling": counts_after,
    }
    return examples, summary


def _retrieval_row(example: AtomicExample) -> dict[str, Any]:
    metadata = asdict(example)
    text_prompt = (
        "Represent the human motion described by the text.\n\n"
        f"Motion description:\n{example.text}\n\n"
        "Return a compact embedding of the motion."
    )
    return {
        "task": "retrieval_eval",
        "query_id": example.query_id,
        "positive_texts": [
            {
                "type": "narration",
                "text": example.text,
                "text_prompt": text_prompt,
            }
        ],
        "activity_label": example.parent_task_name,
        **metadata,
    }


def _captioning_row(example: AtomicExample) -> dict[str, Any]:
    metadata = asdict(example)
    return {
        "task": "captioning_eval",
        "references": [{"type": "narration", "text": example.text}],
        "activity_label": example.parent_task_name,
        **metadata,
    }


def export_dataset(
    *,
    root: Path,
    output_dir: Path,
    annotation_splits: list[str],
    samples_per_class: int,
    min_words: int,
    seed: int,
    normalize_text: bool,
    window_seconds: float,
    sample_rate_hz: int,
    overwrite: bool,
    parent_tasks: list[str] | None = None,
    allow_shortfall: bool = False,
    retrieval_subset_size: int = 100,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output dir already exists and is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, summary = build_atomic_examples(
        root=Path(root),
        annotation_splits=list(annotation_splits),
        samples_per_class=int(samples_per_class),
        min_words=int(min_words),
        seed=int(seed),
        normalize_text=bool(normalize_text),
        require_full_balanced=not bool(allow_shortfall),
        parent_tasks=parent_tasks,
        window_seconds=float(window_seconds),
        sample_rate_hz=int(sample_rate_hz),
    )

    windows: list[np.ndarray] = []
    for example in examples:
        imu = np.asarray(np.load(example.imu_path), dtype=np.float32)
        timestamps = (
            np.asarray(np.load(example.timestamp_path), dtype=np.float64)
            if example.timestamp_path and Path(example.timestamp_path).exists()
            else None
        )
        window, _ = extract_centered_window(
            imu,
            timestamps,
            center_sec=example.timestamp_sec,
            sample_rate_hz=int(sample_rate_hz),
            window_seconds=float(window_seconds),
        )
        windows.append(window)

    if windows:
        stacked_windows = np.stack(windows).astype(np.float32)
    else:
        window_steps = int(round(float(window_seconds) * float(sample_rate_hz)))
        stacked_windows = np.empty((0, 6, window_steps), dtype=np.float32)

    np.savez_compressed(output_dir / "imu_windows.npz", windows=stacked_windows)
    write_jsonl(output_dir / "selected_samples.jsonl", [asdict(example) for example in examples])
    write_jsonl(output_dir / "retrieval.jsonl", [_retrieval_row(example) for example in examples])
    write_jsonl(output_dir / "captioning.jsonl", [_captioning_row(example) for example in examples])
    retrieval_subset_jsonl: Path | None = None
    retrieval_subset_counts: dict[str, int] = {}
    if int(retrieval_subset_size) > 0:
        retrieval_subset = select_balanced_subset(
            examples,
            subset_size=int(retrieval_subset_size),
            seed=int(seed),
            class_order=list(parent_tasks) if parent_tasks else None,
        )
        retrieval_subset_jsonl = output_dir / f"retrieval_{len(retrieval_subset)}_seed{int(seed)}.jsonl"
        write_jsonl(retrieval_subset_jsonl, [_retrieval_row(example) for example in retrieval_subset])
        for example in retrieval_subset:
            retrieval_subset_counts[example.parent_task_name] = (
                retrieval_subset_counts.get(example.parent_task_name, 0) + 1
            )

    summary = {
        **summary,
        "output_dir": str(output_dir),
        "imu_windows_npz": str(output_dir / "imu_windows.npz"),
        "retrieval_jsonl": str(output_dir / "retrieval.jsonl"),
        "captioning_jsonl": str(output_dir / "captioning.jsonl"),
        "selected_samples_jsonl": str(output_dir / "selected_samples.jsonl"),
        "retrieval_subset_jsonl": str(retrieval_subset_jsonl) if retrieval_subset_jsonl else None,
        "retrieval_subset_counts": retrieval_subset_counts,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = export_dataset(
        root=Path(args.egoexo4d_root),
        output_dir=Path(args.output_dir),
        annotation_splits=list(args.annotation_splits),
        samples_per_class=int(args.samples_per_class),
        min_words=int(args.min_words),
        seed=int(args.seed),
        normalize_text=not bool(args.no_normalize_text),
        window_seconds=float(args.window_seconds),
        sample_rate_hz=int(args.sample_rate_hz),
        overwrite=bool(args.overwrite),
        parent_tasks=list(args.parent_tasks),
        allow_shortfall=bool(args.allow_shortfall),
        retrieval_subset_size=int(args.retrieval_subset_size),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
