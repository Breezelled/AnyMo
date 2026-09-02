from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import csv
from collections import Counter, OrderedDict


DEFAULT_NYMERIA_ROOT = Path(os.environ.get("ANYMO_DATA_ROOT", "<PATH_TO_NYMERIA>"))
DEFAULT_EXCLUDED_LABELS = ("writing with foot", "sit-ups")
ATOMIC_LABEL_CSV = "multimodal_sync_60hz/anymo_annotations.csv"
ATOMIC_TIME_CSV = "multimodal_sync_60hz/atomic_action_60hz.csv"
BENCHMARK_LABEL_COLUMNS = {
    "fine150": "activity_label",
    "core50": "superclass_50_label",
}
IMU_FEATURES_PER_DEVICE = 6
IMUS_PER_POSITION = 2
POSITION_FEATURE_STRIDE = IMU_FEATURES_PER_DEVICE * IMUS_PER_POSITION
IMU_CANDIDATES = (
    "multimodal_sync_60hz/sync_imu_60hz.npz",
    "synced_6imu_60hz.npz",
)


@dataclass(frozen=True)
class AnymoWindowRecord:
    imu_path: str
    start: int
    length: int
    start_time_sec: float
    end_time_sec: float
    start_t_ns_global: int
    end_t_ns_global: int
    label_id: int
    label_name: str
    subject: str
    session_name: str


def read_subjects(path: Path) -> set[str]:
    subjects: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            subjects.add(name)
    return subjects


def load_label_vocabulary(
    label_counts_csv: Path,
    excluded_labels=DEFAULT_EXCLUDED_LABELS,
    label_column: str = "activity_label",
) -> tuple[list[str], dict[str, int]]:
    excluded = {label.strip().lower() for label in excluded_labels}
    ordered_labels: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    order_column = label_column.replace("_label", "_id")
    with Path(label_counts_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if label_column not in fieldnames:
            raise ValueError(f"Expected {label_column} in {label_counts_csv}")
        use_order_column = order_column in fieldnames
        for row_idx, row in enumerate(reader):
            label = row[label_column].strip()
            if not label or label.lower() in excluded or label in seen:
                continue
            seen.add(label)
            order_id = int(row[order_column]) if use_order_column else row_idx
            ordered_labels.append((order_id, row_idx, label))
    class_names = [label for _, _, label in sorted(ordered_labels)]
    return class_names, {label: idx for idx, label in enumerate(class_names)}


def label_column_for_benchmark_type(benchmark_type: str) -> str:
    if benchmark_type not in BENCHMARK_LABEL_COLUMNS:
        choices = ", ".join(sorted(BENCHMARK_LABEL_COLUMNS))
        raise ValueError(f"Unknown benchmark_type={benchmark_type!r}; expected one of: {choices}")
    return BENCHMARK_LABEL_COLUMNS[benchmark_type]


def make_window_segments(num_samples: int, window_size: int = 300) -> list[tuple[int, int]]:
    if num_samples <= 0:
        return []
    if num_samples <= window_size:
        return [(0, int(num_samples))]
    return [(offset, window_size) for offset in range(0, num_samples - window_size + 1, window_size)]


def resolve_imu_path(session_dir: Path) -> Path | None:
    for rel in IMU_CANDIDATES:
        path = session_dir / rel
        if path.exists():
            return path
    return None


def load_atomic_time_lookup(session_dir: Path) -> dict[tuple[int, int], dict[str, float | int]]:
    time_csv = session_dir / ATOMIC_TIME_CSV
    if not time_csv.exists():
        return {}
    lookup: dict[tuple[int, int], dict[str, float | int]] = {}
    with time_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"start_idx", "end_idx", "start_t_ns_global", "end_t_ns_global", "start_time", "end_time"}
        if not required.issubset(set(reader.fieldnames or [])):
            return {}
        for row in reader:
            try:
                start_idx = int(row["start_idx"])
                end_idx = int(row["end_idx"])
                lookup[(start_idx, end_idx)] = {
                    "start_t_ns_global": int(row["start_t_ns_global"]),
                    "end_t_ns_global": int(row["end_t_ns_global"]),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                }
            except (TypeError, ValueError):
                continue
    return lookup


def interpolate_time_bounds(
    time_row: dict[str, float | int],
    source_start_idx: int,
    source_end_idx: int,
    window_start_idx: int,
    window_length: int,
) -> tuple[float, float, int, int]:
    source_len = max(1, int(source_end_idx) - int(source_start_idx) + 1)
    rel_start = (int(window_start_idx) - int(source_start_idx)) / source_len
    rel_end = (int(window_start_idx) + int(window_length) - int(source_start_idx)) / source_len
    rel_start = min(1.0, max(0.0, rel_start))
    rel_end = min(1.0, max(0.0, rel_end))

    start_sec = float(time_row["start_time"])
    end_sec = float(time_row["end_time"])
    start_ns = int(time_row["start_t_ns_global"])
    end_ns = int(time_row["end_t_ns_global"])

    window_start_sec = start_sec + (end_sec - start_sec) * rel_start
    window_end_sec = start_sec + (end_sec - start_sec) * rel_end
    window_start_ns = round(start_ns + (end_ns - start_ns) * rel_start)
    window_end_ns = round(start_ns + (end_ns - start_ns) * rel_end)
    return window_start_sec, window_end_sec, int(window_start_ns), int(window_end_ns)


def select_imu_features(x: np.ndarray, imu_index_within_position: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected IMU array shape [T, C], got {arr.shape}")
    if not 0 <= int(imu_index_within_position) < IMUS_PER_POSITION:
        raise ValueError(f"imu_index_within_position must be 0 or 1, got {imu_index_within_position}")
    if arr.shape[1] < POSITION_FEATURE_STRIDE:
        if int(imu_index_within_position) != 0:
            raise ValueError(f"Cannot select second IMU from single-IMU array shape {arr.shape}")
        return arr[:, :IMU_FEATURES_PER_DEVICE]
    chunks = []
    offset = int(imu_index_within_position) * IMU_FEATURES_PER_DEVICE
    for start in range(0, arr.shape[1], POSITION_FEATURE_STRIDE):
        col_start = start + offset
        stop = col_start + IMU_FEATURES_PER_DEVICE
        if stop <= arr.shape[1]:
            chunks.append(arr[:, col_start:stop])
    if not chunks:
        raise ValueError(f"Could not select IMU index {imu_index_within_position} from shape {arr.shape}")
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)


def load_imu_array(imu_path: Path, imu_index_within_position: int = 0) -> np.ndarray:
    with np.load(imu_path, allow_pickle=True) as npz:
        return select_imu_features(np.asarray(npz["x"], dtype=np.float32), imu_index_within_position)


class _ArrayCache:
    def __init__(self, max_entries: int = 4, imu_index_within_position: int = 0) -> None:
        self.max_entries = max(1, int(max_entries))
        self.imu_index_within_position = int(imu_index_within_position)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        arr = load_imu_array(Path(path), self.imu_index_within_position)
        self._cache[path] = arr
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return arr


def _iter_session_dirs(nymeria_root: Path) -> list[Path]:
    return sorted(p for p in Path(nymeria_root).iterdir() if p.is_dir() and (p / "metadata.json").exists())


def _session_subject(session_dir: Path) -> str | None:
    try:
        meta = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    subject = meta.get("fake_name")
    return str(subject).strip() if subject else None


def build_records_for_subjects(
    subjects: set[str],
    label_to_id: dict[str, int],
    nymeria_root: Path,
    window_size: int = 300,
    imu_index_within_position: int = 0,
    label_column: str = "activity_label",
) -> tuple[list[AnymoWindowRecord], dict[str, int]]:
    records: list[AnymoWindowRecord] = []
    stats: Counter[str] = Counter()
    for session_dir in _iter_session_dirs(nymeria_root):
        subject = _session_subject(session_dir)
        if subject not in subjects:
            continue
        stats["sessions_matched_subject_split"] += 1
        label_csv = session_dir / ATOMIC_LABEL_CSV
        imu_path = resolve_imu_path(session_dir)
        if imu_path is None:
            stats["sessions_missing_imu"] += 1
            continue
        if not label_csv.exists():
            stats["sessions_missing_label_csv"] += 1
            continue
        time_lookup = load_atomic_time_lookup(session_dir)
        if not time_lookup:
            stats["sessions_missing_time_csv"] += 1
            continue
        imu_len = int(load_imu_array(imu_path, imu_index_within_position).shape[0])
        with label_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if label_column not in (reader.fieldnames or []):
                stats["sessions_missing_label_column"] += 1
                continue
            for row in reader:
                stats["rows_seen"] += 1
                label = row.get(label_column, "").strip()
                if label not in label_to_id:
                    stats["rows_skipped_label_not_in_vocabulary"] += 1
                    continue
                try:
                    row_start = int(row["start_idx"])
                    row_end = int(row["end_idx"])
                except (KeyError, TypeError, ValueError):
                    stats["rows_skipped_bad_indices"] += 1
                    continue
                time_row = time_lookup.get((row_start, row_end))
                if time_row is None:
                    stats["rows_skipped_missing_time_metadata"] += 1
                    continue
                start = max(0, row_start)
                stop = min(row_end + 1, imu_len)
                num_samples = stop - start
                if num_samples <= 0:
                    stats["rows_skipped_out_of_bounds"] += 1
                    continue
                stats["rows_used"] += 1
                if num_samples < window_size:
                    stats["rows_shorter_than_window"] += 1
                elif num_samples == window_size:
                    stats["rows_exact_window"] += 1
                else:
                    stats["rows_longer_than_window"] += 1
                    stats["dropped_tail_samples"] += num_samples % window_size
                segments = make_window_segments(num_samples, window_size)
                if len(segments) > 1:
                    stats["rows_split_into_multiple_windows"] += 1
                for offset, length in segments:
                    if length < window_size:
                        stats["windows_padded"] += 1
                    window_start = start + offset
                    start_time_sec, end_time_sec, start_t_ns_global, end_t_ns_global = interpolate_time_bounds(
                        time_row,
                        source_start_idx=row_start,
                        source_end_idx=row_end,
                        window_start_idx=window_start,
                        window_length=length,
                    )
                    records.append(
                        AnymoWindowRecord(
                            imu_path=str(imu_path),
                            start=window_start,
                            length=length,
                            start_time_sec=start_time_sec,
                            end_time_sec=end_time_sec,
                            start_t_ns_global=start_t_ns_global,
                            end_t_ns_global=end_t_ns_global,
                            label_id=label_to_id[label],
                            label_name=label,
                            subject=subject,
                            session_name=session_dir.name,
                        )
                    )
                    stats["windows"] += 1
    return records, dict(stats)


DEFAULT_SPLIT_DIR = Path(os.environ.get("ANYMO_BENCH_SPLIT_DIR", "<PATH_TO_ANYMO_BENCH_SPLIT>"))
DEFAULT_LABEL_COUNTS_CSV = DEFAULT_SPLIT_DIR / "filtered_label_counts.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_NYMERIA_ROOT / "anymo_bench_hf"

SAMPLING_RATE_HZ = 60
WINDOW_SIZE = 300
BODY_POSITION = "head+left_wrist+right_wrist"
IMU_DEVICE_NAMES = {
    0: "imu_1202_1",
    1: "imu_1202_2",
}
CHANNEL_NAMES = [
    f"{position}_{sensor}_{axis}"
    for position in ("head", "left_wrist", "right_wrist")
    for sensor in ("acc", "gyro")
    for axis in ("x", "y", "z")
]


@dataclass(frozen=True)
class ExportConfig:
    name: str
    benchmark_type: str
    train_imu_index: int
    test_imu_index: int


CONFIGS = (
    ExportConfig("AnyMo-Bench-150-US", "fine150", 0, 0),
    ExportConfig("AnyMo-Bench-150-USCD", "fine150", 0, 1),
    ExportConfig("AnyMo-Bench-50-US", "core50", 0, 0),
    ExportConfig("AnyMo-Bench-50-USCD", "core50", 0, 1),
)

PARQUET_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("session_id", pa.string()),
        ("subject_id", pa.string()),
        ("body_position", pa.string()),
        ("device_id", pa.string()),
        ("start_time_sec", pa.float64()),
        ("end_time_sec", pa.float64()),
        ("start_t_ns_global", pa.int64()),
        ("end_t_ns_global", pa.int64()),
        ("label", pa.string()),
        ("label_id", pa.int32()),
        ("imu", pa.list_(pa.list_(pa.float32(), list_size=18))),
    ]
)


def make_sample_id(config_name: str, split: str, row_idx: int, rec) -> str:
    source = f"{rec.session_name}|{rec.start}|{rec.length}|{rec.label_id}|{split}"
    suffix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{config_name}:{split}:{row_idx:06d}:{suffix}"


def records_to_table(rows: list[dict], imu_windows: list[np.ndarray]) -> pa.Table:
    arrays = []
    for name in PARQUET_SCHEMA.names:
        if name == "imu":
            offsets = [0]
            flat_parts = []
            for window in imu_windows:
                if window.ndim != 2 or window.shape[1] != 18:
                    raise ValueError(f"Expected IMU window shape [T, 18], got {window.shape}")
                offsets.append(offsets[-1] + int(window.shape[0]))
                flat_parts.append(window.astype(np.float32, copy=False).reshape(-1))
            if flat_parts:
                flat_values = np.concatenate(flat_parts).astype(np.float32, copy=False)
            else:
                flat_values = np.empty((0,), dtype=np.float32)
            timestep_vectors = pa.FixedSizeListArray.from_arrays(pa.array(flat_values, type=pa.float32()), 18)
            arrays.append(pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), timestep_vectors))
        else:
            arrays.append(pa.array([row[name] for row in rows], type=PARQUET_SCHEMA.field(name).type))
    return pa.Table.from_arrays(arrays, schema=PARQUET_SCHEMA)


def write_parquet_shards(
    records,
    config_name: str,
    split: str,
    imu_index: int,
    out_dir: Path,
    shard_size: int,
    max_rows: int | None,
) -> dict[str, int]:
    split_dir = out_dir / "data" / config_name
    split_dir.mkdir(parents=True, exist_ok=True)
    cache = _ArrayCache(max_entries=4, imu_index_within_position=imu_index)
    device_id = IMU_DEVICE_NAMES.get(imu_index, f"imu_index_{imu_index}")

    batch: list[dict] = []
    batch_imu_windows: list[np.ndarray] = []
    shard_idx = 0
    written = 0
    total_timesteps = 0
    min_timesteps: int | None = None
    max_timesteps = 0

    def flush() -> None:
        nonlocal batch, batch_imu_windows, shard_idx
        if not batch:
            return
        table = records_to_table(batch, batch_imu_windows)
        shard_path = split_dir / f"{split}-{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        shard_idx += 1
        batch = []
        batch_imu_windows = []

    for row_idx, rec in enumerate(records):
        if max_rows is not None and row_idx >= max_rows:
            break
        arr = cache.get(rec.imu_path)
        window = arr[rec.start : rec.start + rec.length]
        window = np.nan_to_num(window.astype(np.float32, copy=True), copy=False)
        length = int(window.shape[0])
        total_timesteps += length
        min_timesteps = length if min_timesteps is None else min(min_timesteps, length)
        max_timesteps = max(max_timesteps, length)
        batch.append(
            {
                "sample_id": make_sample_id(config_name, split, row_idx, rec),
                "session_id": rec.session_name,
                "subject_id": rec.subject,
                "body_position": BODY_POSITION,
                "device_id": device_id,
                "start_time_sec": float(rec.start_time_sec),
                "end_time_sec": float(rec.end_time_sec),
                "start_t_ns_global": int(rec.start_t_ns_global),
                "end_t_ns_global": int(rec.end_t_ns_global),
                "label": rec.label_name,
                "label_id": int(rec.label_id),
                "imu": None,
            }
        )
        batch_imu_windows.append(window)
        written += 1
        if len(batch) >= shard_size:
            flush()
    flush()
    return {
        "rows": written,
        "shards": shard_idx,
        "min_timesteps": int(min_timesteps or 0),
        "max_timesteps": int(max_timesteps),
        "total_timesteps": int(total_timesteps),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_dataset_card(out_dir: Path, summaries: dict, train_subjects: set[str], test_subjects: set[str]) -> None:
    yaml_configs = []
    for config in CONFIGS:
        yaml_configs.append(
            f"""- config_name: {config.name}
  data_files:
  - split: train
    path: data/{config.name}/train-*.parquet
  - split: test
    path: data/{config.name}/test-*.parquet"""
        )
    yaml_block = "\n".join(yaml_configs)
    fine150 = summaries["AnyMo-Bench-150-US"]
    total_intervals = fine150["train_build_stats"]["rows_used"] + fine150["test_build_stats"]["rows_used"]
    total_segments = fine150["train"]["rows"] + fine150["test"]["rows"]
    readme = f"""---
license: cc-by-nc-4.0
pretty_name: AnyMo Bench
task_categories:
- other
tags:
- human-activity-recognition
- imu
- wearable-sensing
- time-series
- timeseries
- in-the-wild
- nymeria
configs:
{yaml_block}
---

# AnyMo Bench

AnyMo Bench is a challenging fine-grained in-the-wild HAR benchmark built from real wearable IMU streams in the Nymeria dataset. It provides unseen-subject and cross-device evaluation settings for wearable motion recognition.

The benchmark contains {total_intervals:,} eligible activity windows from {len(train_subjects) + len(test_subjects)} subjects, covering 211.6 hours of real in-the-wild IMU data. IMU streams are synchronized to a common {SAMPLING_RATE_HZ} Hz temporal grid. Each row contains one activity window with an `imu` array of shape `[T, 18]`, where `T <= {WINDOW_SIZE}` and the 18 channels concatenate one selected IMU from each of Head, Left Wrist, and Right Wrist.

## Configurations

- `AnyMo-Bench-150-US`: Fine150 label space, unseen-subject evaluation, train/test use the first co-located IMU at each body position.
- `AnyMo-Bench-150-USCD`: Fine150 label space, unseen-subject + cross-device evaluation, train uses the first co-located IMU and test uses the second co-located IMU at each body position.
- `AnyMo-Bench-50-US`: Core50 label space, unseen-subject evaluation.
- `AnyMo-Bench-50-USCD`: Core50 label space, unseen-subject + cross-device evaluation.

```python
from datasets import load_dataset

dataset = load_dataset("CRUISEResearchGroup/AnyMo-Bench", "AnyMo-Bench-150-US")
```

## Schema

- `sample_id`: unique row identifier within the exported benchmark.
- `session_id`: Nymeria sequence/session directory name.
- `subject_id`: Nymeria anonymized subject identifier.
- `body_position`: concatenated body-position group, currently `head+left_wrist+right_wrist`.
- `device_id`: selected co-located IMU unit, `imu_1202_1` or `imu_1202_2`.
- `start_time_sec`, `end_time_sec`: Nymeria time-code boundaries in seconds for the activity window.
- `start_t_ns_global`, `end_t_ns_global`: Nymeria global time-code boundaries in nanoseconds for the activity window.
- `label`: activity label for the selected label taxonomy.
- `label_id`: zero-based label index for the selected label taxonomy.
- `imu`: nested list with shape `[T, 18]`.

Channel order:

```text
{", ".join(CHANNEL_NAMES)}
```

The exported IMU values are the processed 60 Hz benchmark windows before model-specific normalization. Shorter-than-5-second windows are not zero-padded in the Parquet files. Windows longer than 300 timesteps are split into non-overlapping 300-timestep segments, matching the AnyMo Bench evaluation construction.

## Mapping Back to Nymeria

Each row includes the Nymeria `session_id` and global time-code fields (`start_t_ns_global`, `end_t_ns_global`). These fields let users map AnyMo Bench labels back to the original Nymeria sequence and align the labels with other Nymeria modalities, including RGB video and body motion. The official Nymeria tools are available at [facebookresearch/nymeria_dataset](https://github.com/facebookresearch/nymeria_dataset). See `examples/map_to_nymeria_modalities.py` for a minimal example.

## Split Summary

| Config | Train rows | Test rows | Train device | Test device |
|---|---:|---:|---|---|
"""
    for config in CONFIGS:
        train = summaries[config.name]["train"]
        test = summaries[config.name]["test"]
        readme += (
            f"| `{config.name}` | {train['rows']:,} | {test['rows']:,} | "
            f"{IMU_DEVICE_NAMES[config.train_imu_index]} | {IMU_DEVICE_NAMES[config.test_imu_index]} |\n"
        )
    readme += """
## License and Source Dataset

AnyMo Bench is curated from Nymeria and follows the Nymeria non-commercial research-use terms. Please cite the Nymeria dataset and AnyMo when using this data.

```bibtex
@article{chen2026anymo,
  title={AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild},
  author={Chen, Baiyu and Li, Zechen and Wongso, Wilson and Li, Lihuan and Lin, Xiachong and Xue, Hao and Tag, Benjamin and Salim, Flora},
  journal={arXiv preprint arXiv:2605.22715},
  year={2026}
}

@inproceedings{ma2024nymeria,
  title={Nymeria: A massive collection of multimodal egocentric daily motion in the wild},
  author={Ma, Lingni and Ye, Yuting and Hong, Fangzhou and Guzov, Vladimir and Jiang, Yifeng and Postyeni, Rowan and Pesqueira, Luis and Gamino, Alexander and Baiyya, Vijay and Kim, Hyo Jin and others},
  booktitle={European Conference on Computer Vision},
  pages={445--465},
  year={2024},
  organization={Springer}
}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def write_helper_examples(out_dir: Path) -> None:
    helper = '"""Map an AnyMo Bench row back to raw Nymeria modalities.\n\nInstall or clone the official Nymeria tools first:\nhttps://github.com/facebookresearch/nymeria_dataset\n\nThis example uses the row\'s Nymeria session_id and global TIME_CODE\nnanosecond timestamps to query the corresponding original sequence.\n"""\nfrom pathlib import Path\n\nfrom datasets import load_dataset\nfrom nymeria.data_provider import NymeriaDataProvider\n\nNYMERIA_ROOT = Path("<PATH_TO_NYMERIA>")\n\nbenchmark = load_dataset(\n    "CRUISEResearchGroup/AnyMo-Bench",\n    "AnyMo-Bench-150-US",\n    split="train",\n    streaming=True,\n)\nrow = next(iter(benchmark))\n\nsequence_root = NYMERIA_ROOT / row["session_id"]\nprovider = NymeriaDataProvider(sequence_rootdir=sequence_root)\n\nstart_t_ns = int(row["start_t_ns_global"])\nend_t_ns = int(row["end_t_ns_global"])\nmid_t_ns = (start_t_ns + end_t_ns) // 2\n\n# Query modalities around the benchmark label window.\nrgb_frames = provider.get_synced_rgb_videos(mid_t_ns)\nposes = provider.get_synced_poses(mid_t_ns)\n\nprint(row["label"], row["session_id"], start_t_ns, end_t_ns)\nprint(rgb_frames.keys())\nprint(poses.keys())\n'
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "map_to_nymeria_modalities.py").write_text(helper, encoding="utf-8")


def write_gitattributes(out_dir: Path) -> None:
    (out_dir / ".gitattributes").write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )


def export(args: argparse.Namespace) -> None:
    nymeria_root = Path(args.nymeria_root)
    label_counts_csv = Path(args.label_counts_csv)
    train_subjects_path = Path(args.train_subjects_path)
    test_subjects_path = Path(args.test_subjects_path)
    out_dir = Path(args.output_dir)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_subjects = read_subjects(train_subjects_path)
    test_subjects = read_subjects(test_subjects_path)
    summaries: dict[str, dict] = {}
    label_metadata: dict[str, list[dict]] = {}

    for config in CONFIGS:
        label_column = label_column_for_benchmark_type(config.benchmark_type)
        class_names, label_to_id = load_label_vocabulary(
            label_counts_csv,
            excluded_labels=DEFAULT_EXCLUDED_LABELS,
            label_column=label_column,
        )
        labels = [{"label_id": idx, "label": label} for idx, label in enumerate(class_names)]
        label_metadata[config.benchmark_type] = labels

        train_records, train_stats = build_records_for_subjects(
            train_subjects,
            label_to_id,
            nymeria_root=nymeria_root,
            window_size=WINDOW_SIZE,
            imu_index_within_position=config.train_imu_index,
            label_column=label_column,
        )
        test_records, test_stats = build_records_for_subjects(
            test_subjects,
            label_to_id,
            nymeria_root=nymeria_root,
            window_size=WINDOW_SIZE,
            imu_index_within_position=config.test_imu_index,
            label_column=label_column,
        )
        config_summary = {
            "benchmark_type": config.benchmark_type,
            "label_column": label_column,
            "num_classes": len(class_names),
            "train_imu_index_within_position": config.train_imu_index,
            "test_imu_index_within_position": config.test_imu_index,
            "train_build_stats": train_stats,
            "test_build_stats": test_stats,
        }
        config_summary["train"] = write_parquet_shards(
            train_records,
            config.name,
            "train",
            config.train_imu_index,
            out_dir,
            args.shard_size,
            args.max_rows_per_split,
        )
        config_summary["test"] = write_parquet_shards(
            test_records,
            config.name,
            "test",
            config.test_imu_index,
            out_dir,
            args.shard_size,
            args.max_rows_per_split,
        )
        summaries[config.name] = config_summary
        print(
            f"{config.name}: train={config_summary['train']['rows']} rows, "
            f"test={config_summary['test']['rows']} rows"
        )

    for benchmark_type, labels in sorted(label_metadata.items()):
        write_json(out_dir / "metadata" / f"{benchmark_type}_labels.json", labels)
    write_json(
        out_dir / "metadata" / "split_subjects.json",
        {
            "train_subjects": sorted(train_subjects),
            "test_subjects": sorted(test_subjects),
        },
    )
    write_json(
        out_dir / "metadata" / "dataset_info.json",
        {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "window_size_timesteps": WINDOW_SIZE,
            "body_position": BODY_POSITION,
            "channel_names": CHANNEL_NAMES,
            "summaries": summaries,
        },
    )
    write_dataset_card(out_dir, summaries, train_subjects, test_subjects)
    write_helper_examples(out_dir)
    write_gitattributes(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AnyMo Bench as a Hugging Face-ready Parquet dataset.")
    parser.add_argument("--nymeria-root", type=Path, default=DEFAULT_NYMERIA_ROOT)
    parser.add_argument("--label-counts-csv", type=Path, default=DEFAULT_LABEL_COUNTS_CSV)
    parser.add_argument("--train-subjects-path", type=Path, default=DEFAULT_SPLIT_DIR / "train_subjects.txt")
    parser.add_argument("--test-subjects-path", type=Path, default=DEFAULT_SPLIT_DIR / "test_subjects.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    export(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
