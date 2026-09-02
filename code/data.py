"""Data loading and preprocessing for AnyMo training and paper evaluation."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import threading
import zipfile
import zlib
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import numpy.core as numpy_core
import torch
from scipy.signal import resample
from torch.utils.data import Dataset
from tqdm import tqdm

import body_part_mapping


sys.modules.setdefault("numpy._core", numpy_core)
sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)


SEGMENT_NAMES = [
    "Pelvis",
    "L5",
    "L3",
    "T12",
    "T8",
    "Neck",
    "Head",
    "R_Shoulder",
    "R_UpperArm",
    "R_Forearm",
    "R_Hand",
    "L_Shoulder",
    "L_UpperArm",
    "L_Forearm",
    "L_Hand",
    "R_UpperLeg",
    "R_LowerLeg",
    "R_Foot",
    "R_Toe",
    "L_UpperLeg",
    "L_LowerLeg",
    "L_Foot",
    "L_Toe",
]
SITE_NAMES = SEGMENT_NAMES
SITE_TO_INDEX = {name: idx for idx, name in enumerate(SITE_NAMES)}
REAL_SITE_NAMES = ["head", "lwrist", "rwrist"]
REAL_SITE_TO_SEGMENT = {"head": "Head", "lwrist": "L_Forearm", "rwrist": "R_Forearm"}
REAL_SEGMENT_NAMES = [REAL_SITE_TO_SEGMENT[name] for name in REAL_SITE_NAMES]
REAL_SEGMENT_INDICES = [SITE_TO_INDEX[name] for name in REAL_SEGMENT_NAMES]
REAL_FEATS = ["ax", "ay", "az", "gx", "gy", "gz"]
HELD_OUT_SUBJECTS = {
    "alec_meza",
    "bradley_herman",
    "dominique_frye",
    "justin_ramirez",
    "kyle_parker",
}
SUBJECT_RE = re.compile(r"^\d{8}_s\d+_(?P<subject>.+?)_act\d+_")


@dataclass(frozen=True)
class SampleRecord:
    sample_dir: str
    sample_path: str
    synth_npz: str
    real_npz: str
    label_name: str


@dataclass(frozen=True)
class ClipKey:
    sample_dir: str
    clip_start: int
    label_name: str


def subject_from_sample_dir(sample_dir: str) -> str:
    match = SUBJECT_RE.match(str(sample_dir))
    return match.group("subject") if match else ""


def exclude_held_out_subjects(records: list[SampleRecord]) -> list[SampleRecord]:
    """Remove the five subject-disjoint Nymeria evaluation participants."""
    return [
        record
        for record in records
        if subject_from_sample_dir(record.sample_dir) not in HELD_OUT_SUBJECTS
    ]


def load_summary_records(
    summary_csv: Path, candidate_name: str, base_dir: Path
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with summary_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("candidate_name") != candidate_name:
                continue
            if row.get("status") not in {"ok", "skipped_existing"}:
                continue
            output_path = Path(row["output_path"])
            if not output_path.exists() and not output_path.with_suffix("").is_dir():
                continue
            sample_dir = row["sample_dir"]
            sample_path = base_dir / sample_dir
            meta = json.loads(
                (sample_path / "metadata.json").read_text(encoding="utf-8")
            )
            records.append(
                SampleRecord(
                    sample_dir=sample_dir,
                    sample_path=str(sample_path),
                    synth_npz=str(output_path),
                    real_npz=str(output_path.parent / "sync_imu_60hz.npz"),
                    label_name=str(meta["script"]),
                )
            )
    records.sort(key=lambda x: x.sample_dir)
    return records


def load_geometry_aware_imu_archive(npz_path: Path) -> dict[str, Any]:
    bundle_dir = npz_path.with_suffix("")
    if bundle_dir.is_dir():
        x_zarr_path = bundle_dir / "x.zarr"
        archive: dict[str, Any] = {}
        if x_zarr_path.exists():
            try:
                import zarr
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Found x.zarr synthetic bundle, but zarr is not installed in the active environment."
                ) from exc
            archive["x"] = zarr.open(str(x_zarr_path), mode="r")
        for p in sorted(bundle_dir.glob("*.npy")):
            if p.stem == "x" and "x" in archive:
                continue
            try:
                archive[p.stem] = np.load(p, mmap_mode="r", allow_pickle=True)
            except ValueError:
                archive[p.stem] = np.load(p, allow_pickle=True)
        return archive
    with np.load(npz_path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def resolve_real_layout(
    real_npz_path: Path, device_suffix: str
) -> tuple[list[int], list[int]]:
    with np.load(real_npz_path, allow_pickle=True) as npz:
        stream_order = [str(x) for x in npz["stream_order"].tolist()]
        feature_cols = [str(x) for x in npz["feature_cols"].tolist()]

    stream_indices: list[int] = []
    feature_indices: list[int] = []
    for site in REAL_SITE_NAMES:
        stream_name = f"{site}:{device_suffix}"
        stream_indices.append(stream_order.index(stream_name))
        feature_prefix = stream_name.replace(":", "_")
        for feat in REAL_FEATS:
            feature_indices.append(feature_cols.index(f"{feature_prefix}_{feat}"))
    return stream_indices, feature_indices


def load_selected_real_array(real_npz_path: Path, device_suffix: str) -> np.ndarray:
    _, feature_indices = resolve_real_layout(real_npz_path, device_suffix)
    with np.load(real_npz_path, allow_pickle=True) as npz:
        return np.asarray(npz["x"][:, feature_indices], dtype=np.float32)


class RealArrayStore:
    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, real_npz_path: str, device_suffix: str) -> np.ndarray:
        key = (real_npz_path, device_suffix)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = load_selected_real_array(Path(real_npz_path), device_suffix)
        self._cache[key] = arr
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return arr


def build_real_frame_valid_mask(real_npz_path: Path, device_suffix: str) -> np.ndarray:
    stream_indices, feature_indices = resolve_real_layout(real_npz_path, device_suffix)
    with np.load(real_npz_path, allow_pickle=True) as npz:
        x = np.asarray(npz["x"][:, feature_indices], dtype=np.float32)
        stream_valid = np.asarray(npz["acc_valid"], dtype=bool) & np.asarray(
            npz["gyro_valid"], dtype=bool
        )
    return np.all(stream_valid[:, stream_indices], axis=1) & np.isfinite(x).all(axis=1)


def build_synthetic_frame_valid_mask(synth_npz_path: Path) -> np.ndarray:
    x = load_geometry_aware_imu_archive(synth_npz_path)["x"]
    return np.ones(int(x.shape[1]), dtype=bool)


def load_real_time_axis(real_npz_path: Path) -> np.ndarray | None:
    with np.load(real_npz_path, allow_pickle=True) as npz:
        if "t_ns_global_timecode" not in npz:
            return None
        return np.asarray(npz["t_ns_global_timecode"], dtype=np.int64)


def load_synthetic_time_axis(synth_npz_path: Path) -> np.ndarray | None:
    bundle_dir = synth_npz_path.with_suffix("")
    time_axis_path = bundle_dir / "t_ns_global_timecode.npy"
    if time_axis_path.exists():
        return np.asarray(np.load(time_axis_path, mmap_mode="r"), dtype=np.int64)
    if synth_npz_path.exists():
        with np.load(synth_npz_path, allow_pickle=True) as npz:
            if "t_ns_global_timecode" in npz:
                return np.asarray(npz["t_ns_global_timecode"], dtype=np.int64)
    return None


def align_frame_mask_to_timestamps(
    source_mask: np.ndarray,
    source_t_ns: np.ndarray,
    target_t_ns: np.ndarray,
) -> np.ndarray:
    source_mask = np.asarray(source_mask, dtype=bool)
    source_t_ns = np.asarray(source_t_ns, dtype=np.int64)
    target_t_ns = np.asarray(target_t_ns, dtype=np.int64)
    if source_mask.shape[0] != source_t_ns.shape[0]:
        raise ValueError("source mask and timestamp lengths differ")
    if source_t_ns.size == 0:
        return np.zeros(target_t_ns.shape[0], dtype=bool)
    right = np.searchsorted(source_t_ns, target_t_ns, side="left")
    right = np.clip(right, 0, source_t_ns.shape[0] - 1)
    left = np.clip(right - 1, 0, source_t_ns.shape[0] - 1)
    choose_left = np.abs(target_t_ns - source_t_ns[left]) <= np.abs(
        source_t_ns[right] - target_t_ns
    )
    nearest = np.where(choose_left, left, right)
    return source_mask[nearest]


def build_frame_valid_mask(
    record: SampleRecord, real1_device_suffix: str, real2_device_suffix: str
) -> np.ndarray:
    real1 = build_real_frame_valid_mask(Path(record.real_npz), real1_device_suffix)
    real2 = build_real_frame_valid_mask(Path(record.real_npz), real2_device_suffix)
    synth = build_synthetic_frame_valid_mask(Path(record.synth_npz))
    real_t_ns = load_real_time_axis(Path(record.real_npz))
    synth_t_ns = load_synthetic_time_axis(Path(record.synth_npz))
    if real_t_ns is not None and synth_t_ns is not None:
        real1_aligned = align_frame_mask_to_timestamps(real1, real_t_ns, synth_t_ns)
        real2_aligned = align_frame_mask_to_timestamps(real2, real_t_ns, synth_t_ns)
        shared_len = min(len(real1_aligned), len(real2_aligned), len(synth))
        return (
            real1_aligned[:shared_len] & real2_aligned[:shared_len] & synth[:shared_len]
        )
    shared_len = min(len(real1), len(real2), len(synth))
    return real1[:shared_len] & real2[:shared_len] & synth[:shared_len]


def split_counts_for_class(num_items: int) -> tuple[int, int, int]:
    n_train = int(np.floor(num_items * 0.7))
    n_val = int(np.floor(num_items * 0.1))
    n_test = num_items - n_train - n_val
    if n_val == 0:
        n_val = 1
        n_train -= 1
    if n_test == 0:
        n_test = 1
        n_train -= 1
    return n_train, n_val, n_test


def build_split_index(
    records: list[SampleRecord],
    window_size: int,
    seed: int,
    real1_device_suffix: str,
    real2_device_suffix: str,
) -> tuple[list[ClipKey], list[ClipKey], list[ClipKey], list[str]]:
    rng = np.random.default_rng(seed)
    class_to_keys: dict[str, list[ClipKey]] = defaultdict(list)
    for record in records:
        shared_valid = build_frame_valid_mask(
            record, real1_device_suffix, real2_device_suffix
        )
        for clip_start in range(
            0, shared_valid.shape[0] - window_size + 1, window_size
        ):
            if bool(np.all(shared_valid[clip_start : clip_start + window_size])):
                class_to_keys[record.label_name].append(
                    ClipKey(
                        sample_dir=record.sample_dir,
                        clip_start=clip_start,
                        label_name=record.label_name,
                    )
                )

    class_names = sorted(class_to_keys)
    train_keys: list[ClipKey] = []
    val_keys: list[ClipKey] = []
    test_keys: list[ClipKey] = []
    for class_name in class_names:
        keys = class_to_keys[class_name]
        idx = np.arange(len(keys))
        rng.shuffle(idx)
        n_train, n_val, n_test = split_counts_for_class(len(keys))
        train_keys.extend(keys[i] for i in idx[:n_train])
        val_keys.extend(keys[i] for i in idx[n_train : n_train + n_val])
        test_keys.extend(
            keys[i] for i in idx[n_train + n_val : n_train + n_val + n_test]
        )
    rng.shuffle(train_keys)
    rng.shuffle(val_keys)
    rng.shuffle(test_keys)
    return train_keys, val_keys, test_keys, class_names


def get_or_build_split_index(
    records: list[SampleRecord],
    window_size: int,
    seed: int,
    real1_device_suffix: str,
    real2_device_suffix: str,
    cache_path: Path,
) -> tuple[list[ClipKey], list[ClipKey], list[ClipKey], list[str]]:
    sample_dirs = [record.sample_dir for record in records]
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            obj["sample_dirs"] == sample_dirs
            and int(obj["window_size"]) == window_size
            and int(obj["seed"]) == seed
        ):
            return (
                [ClipKey(**x) for x in obj["train_keys"]],
                [ClipKey(**x) for x in obj["val_keys"]],
                [ClipKey(**x) for x in obj["test_keys"]],
                obj["class_names"],
            )
    train_keys, val_keys, test_keys, class_names = build_split_index(
        records=records,
        window_size=window_size,
        seed=seed,
        real1_device_suffix=real1_device_suffix,
        real2_device_suffix=real2_device_suffix,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "sample_dirs": sample_dirs,
                "window_size": window_size,
                "seed": seed,
                "class_names": class_names,
                "train_keys": [asdict(x) for x in train_keys],
                "val_keys": [asdict(x) for x in val_keys],
                "test_keys": [asdict(x) for x in test_keys],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_keys, val_keys, test_keys, class_names


def build_full_graph_input_from_ct(arr_ct: np.ndarray) -> np.ndarray:
    t = int(arr_ct.shape[0])
    out = np.zeros((6, t, len(SITE_NAMES), 1), dtype=np.float32)
    for site_idx, segment_idx in enumerate(REAL_SEGMENT_INDICES):
        start = site_idx * 6
        out[:, :, segment_idx, 0] = arr_ct[:, start : start + 6].T
    return out


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def sample_surface_rotation_matrix(
    rng: np.random.Generator,
    inplane_max_deg: float,
    tilt_max_deg: float,
) -> np.ndarray:
    inplane_deg = float(rng.uniform(-inplane_max_deg, inplane_max_deg))
    tilt_x_deg = float(rng.uniform(-tilt_max_deg, tilt_max_deg))
    tilt_y_deg = float(rng.uniform(-tilt_max_deg, tilt_max_deg))
    return (
        rotation_matrix_y(np.deg2rad(tilt_y_deg))
        @ rotation_matrix_x(np.deg2rad(tilt_x_deg))
        @ rotation_matrix_z(np.deg2rad(inplane_deg))
    ).astype(np.float32)


def apply_surface_rotation_to_placement_window(
    placement: np.ndarray, rot: np.ndarray
) -> np.ndarray:
    out = placement.copy().astype(np.float32)
    out[:, 0:3] = placement[:, 0:3] @ rot.T
    out[:, 3:6] = placement[:, 3:6] @ rot.T
    return out


class SiteArrayView:
    def __init__(self, x: Any, start: int, count: int):
        self._x = x
        self._start = int(start)
        self._count = int(count)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._count, int(self._x.shape[1]), int(self._x.shape[2]))

    def __len__(self) -> int:
        return self._count

    def _placement_index(self, key: Any) -> Any:
        base = np.arange(self._start, self._start + self._count, dtype=np.int64)
        return base[key]

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        placement_key, time_key, feat_key = key[:3]
        return self._x[self._placement_index(placement_key), time_key, feat_key]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        out = np.asarray(self._x[self._start : self._start + self._count, :, :])
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out


class DominantTop2Store:
    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_bundle(self, npz_path: str) -> dict[str, Any]:
        with self._lock:
            if npz_path in self._cache:
                self._cache.move_to_end(npz_path)
                return self._cache[npz_path]
        archive = load_geometry_aware_imu_archive(Path(npz_path))
        x = archive["x"]
        site_order = [
            str(x) for x in np.asarray(archive["site_order"], dtype=object).tolist()
        ]
        site_offsets = np.asarray(archive["site_offsets"], dtype=np.int32)
        out: dict[str, Any] = {
            "x": x,
            "site_order": site_order,
            "site_offsets": site_offsets,
        }
        for site_name, (start, count) in zip(site_order, site_offsets.tolist()):
            out[site_name] = SiteArrayView(x, start, count)
        with self._lock:
            cached = self._cache.get(npz_path)
            if cached is not None:
                self._cache.move_to_end(npz_path)
                return cached
            self._cache[npz_path] = out
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return out

    def get(self, npz_path: str) -> dict[str, Any]:
        return self._load_bundle(npz_path)

    def get_window(
        self, npz_path: str, clip_start: int, window_size: int
    ) -> dict[str, np.ndarray]:
        bundle = self._load_bundle(npz_path)
        clip_end = clip_start + window_size
        x_window = np.asarray(bundle["x"][:, clip_start:clip_end, :], dtype=np.float32)
        out: dict[str, np.ndarray] = {}
        for site_name, (start, count) in zip(
            bundle["site_order"], bundle["site_offsets"].tolist()
        ):
            out[site_name] = x_window[start : start + count]
        return out


def sample_dual_view_window(
    site_arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    surface_rotation_augment: bool = True,
    inplane_max_deg: float = 180.0,
    tilt_max_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    selections_a: dict[str, np.ndarray] = {}
    selections_b: dict[str, np.ndarray] = {}
    for site_name in SITE_NAMES:
        site_x = np.asarray(site_arrays[site_name], dtype=np.float32)
        idx_a = int(rng.integers(0, site_x.shape[0]))
        idx_b = int(rng.integers(0, site_x.shape[0]))
        placement_a = site_x[idx_a]
        placement_b = site_x[idx_b]
        if surface_rotation_augment:
            placement_a = apply_surface_rotation_to_placement_window(
                placement_a,
                sample_surface_rotation_matrix(rng, inplane_max_deg, tilt_max_deg),
            )
            placement_b = apply_surface_rotation_to_placement_window(
                placement_b,
                sample_surface_rotation_matrix(rng, inplane_max_deg, tilt_max_deg),
            )
        selections_a[site_name] = placement_a
        selections_b[site_name] = placement_b
    return build_graph_view(selections_a), build_graph_view(selections_b)


def sample_single_view_window(
    site_arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    surface_rotation_augment: bool = True,
    inplane_max_deg: float = 180.0,
    tilt_max_deg: float = 10.0,
) -> np.ndarray:
    selections: dict[str, np.ndarray] = {}
    for site_name in SITE_NAMES:
        site_x = np.asarray(site_arrays[site_name], dtype=np.float32)
        idx = int(rng.integers(0, site_x.shape[0]))
        placement = site_x[idx]
        if surface_rotation_augment:
            placement = apply_surface_rotation_to_placement_window(
                placement,
                sample_surface_rotation_matrix(rng, inplane_max_deg, tilt_max_deg),
            )
        selections[site_name] = placement
    return build_graph_view(selections)


def build_graph_view(selected_by_site: dict[str, np.ndarray]) -> np.ndarray:
    t = int(selected_by_site[SITE_NAMES[0]].shape[0])
    x = np.zeros((6, t, len(SITE_NAMES), 1), dtype=np.float32)
    for site_name in SITE_NAMES:
        x[:, :, SITE_TO_INDEX[site_name], 0] = selected_by_site[site_name].T
    return x


def build_all_clip_index(
    records: list[SampleRecord],
    window_size: int,
    real1_device_suffix: str,
    real2_device_suffix: str,
) -> list[ClipKey]:
    keys: list[ClipKey] = []
    for record in records:
        shared_valid = build_frame_valid_mask(
            record, real1_device_suffix, real2_device_suffix
        )
        for clip_start in range(
            0, shared_valid.shape[0] - window_size + 1, window_size
        ):
            if bool(np.all(shared_valid[clip_start : clip_start + window_size])):
                keys.append(
                    ClipKey(
                        sample_dir=record.sample_dir,
                        clip_start=clip_start,
                        label_name=record.label_name,
                    )
                )
    return keys


class PretrainDataset(Dataset):
    def __init__(
        self,
        clip_keys: list[ClipKey],
        record_by_sample: dict[str, SampleRecord],
        synth_store: DominantTop2Store,
        window_size: int,
        seed: int,
        surface_rotation_augment: bool = True,
        inplane_max_deg: float = 180.0,
        tilt_max_deg: float = 10.0,
    ):
        self.clip_keys = clip_keys
        self.record_by_sample = record_by_sample
        self.synth_store = synth_store
        self.window_size = window_size
        self.seed = seed
        self.surface_rotation_augment = surface_rotation_augment
        self.inplane_max_deg = inplane_max_deg
        self.tilt_max_deg = tilt_max_deg

    def __len__(self) -> int:
        return len(self.clip_keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip_key = self.clip_keys[idx]
        record = self.record_by_sample[clip_key.sample_dir]
        site_arrays = self.synth_store.get_window(
            record.synth_npz, clip_key.clip_start, self.window_size
        )
        seed = (
            self.seed
            + clip_key.clip_start
            + int(zlib.crc32(clip_key.sample_dir.encode("utf-8")))
        )
        rng = np.random.default_rng(seed)
        view_a, view_b = sample_dual_view_window(
            site_arrays,
            rng,
            surface_rotation_augment=self.surface_rotation_augment,
            inplane_max_deg=self.inplane_max_deg,
            tilt_max_deg=self.tilt_max_deg,
        )
        return {
            "view_a_full": torch.tensor(view_a, dtype=torch.float32),
            "view_b_full": torch.tensor(view_b, dtype=torch.float32),
        }


class SyntheticClassificationDataset(Dataset):
    def __init__(
        self,
        clip_keys: list[ClipKey],
        record_by_sample: dict[str, SampleRecord],
        label_to_id: dict[str, int],
        synth_store: DominantTop2Store,
        window_size: int,
        seed: int,
    ):
        self.clip_keys = clip_keys
        self.record_by_sample = record_by_sample
        self.label_to_id = label_to_id
        self.synth_store = synth_store
        self.window_size = window_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clip_keys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_key = self.clip_keys[idx]
        record = self.record_by_sample[clip_key.sample_dir]
        site_arrays = self.synth_store.get_window(
            record.synth_npz, clip_key.clip_start, self.window_size
        )
        seed = (
            self.seed
            + clip_key.clip_start
            + int(zlib.crc32(clip_key.sample_dir.encode("utf-8")))
        )
        view = sample_single_view_window(
            site_arrays, np.random.default_rng(seed), surface_rotation_augment=False
        )
        return torch.tensor(view, dtype=torch.float32), torch.tensor(
            self.label_to_id[clip_key.label_name], dtype=torch.long
        )


class MaskedMotionTokenizerDataset(Dataset):
    def __init__(
        self,
        clip_keys: list[ClipKey],
        record_by_sample: dict[str, SampleRecord],
        synth_store: DominantTop2Store,
        window_size: int,
        seed: int,
        surface_rotation_augment: bool = True,
        inplane_max_deg: float = 180.0,
        tilt_max_deg: float = 10.0,
        preload_all_views: bool = False,
        preload_workers: int = 1,
    ):
        self.clip_keys = clip_keys
        self.record_by_sample = record_by_sample
        self.synth_store = synth_store
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.surface_rotation_augment = bool(surface_rotation_augment)
        self.inplane_max_deg = float(inplane_max_deg)
        self.tilt_max_deg = float(tilt_max_deg)
        self.preload_all_views = bool(preload_all_views)
        self.preload_workers = max(1, int(preload_workers))
        self._preloaded_views: list[torch.Tensor] | None = None
        if self.preload_all_views:
            indices = range(len(self.clip_keys))
            if self.preload_workers > 1:
                with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
                    self._preloaded_views = list(
                        tqdm(
                            executor.map(self._build_view_full, indices),
                            total=len(self.clip_keys),
                            desc="preload view_full",
                            leave=False,
                        )
                    )
            else:
                self._preloaded_views = [
                    self._build_view_full(idx)
                    for idx in tqdm(indices, desc="preload view_full", leave=False)
                ]

    def __len__(self) -> int:
        return len(self.clip_keys)

    def _build_view_full(self, idx: int) -> torch.Tensor:
        clip_key = self.clip_keys[idx]
        record = self.record_by_sample[clip_key.sample_dir]
        site_arrays = self.synth_store.get_window(
            record.synth_npz, clip_key.clip_start, self.window_size
        )
        seed = (
            self.seed
            + clip_key.clip_start
            + int(zlib.crc32(clip_key.sample_dir.encode("utf-8")))
        )
        view = sample_single_view_window(
            site_arrays,
            np.random.default_rng(seed),
            surface_rotation_augment=self.surface_rotation_augment,
            inplane_max_deg=self.inplane_max_deg,
            tilt_max_deg=self.tilt_max_deg,
        )
        return torch.tensor(view, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._preloaded_views is not None:
            return {"view_full": self._preloaded_views[idx]}
        return {"view_full": self._build_view_full(idx)}


class RealClassificationDataset(Dataset):
    def __init__(
        self,
        clip_keys: list[ClipKey],
        record_by_sample: dict[str, SampleRecord],
        label_to_id: dict[str, int],
        device_suffix: str,
        window_size: int,
        real_store: RealArrayStore | None = None,
    ):
        self.clip_keys = clip_keys
        self.record_by_sample = record_by_sample
        self.label_to_id = label_to_id
        self.device_suffix = device_suffix
        self.window_size = window_size
        self.real_store = real_store if real_store is not None else RealArrayStore()

    def __len__(self) -> int:
        return len(self.clip_keys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_key = self.clip_keys[idx]
        record = self.record_by_sample[clip_key.sample_dir]
        arr = self.real_store.get(record.real_npz, self.device_suffix)
        clip = arr[clip_key.clip_start : clip_key.clip_start + self.window_size]
        x = build_full_graph_input_from_ct(clip)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            self.label_to_id[clip_key.label_name], dtype=torch.long
        )


# -----------------------------------------------------------------------------
# External zero-shot HAR benchmark datasets
# -----------------------------------------------------------------------------

DEFAULT_EGO4D_ROOT = Path(os.environ.get("EGO4D_ROOT", "<PATH_TO_EGO4D>"))
DEFAULT_MMEA_ROOT = Path(os.environ.get("MMEA_ROOT", "<PATH_TO_MMEA>"))
DEFAULT_EGOEXO4D_ROOT = Path(os.environ.get("EGOEXO4D_ROOT", "<PATH_TO_EGOEXO4D>"))
DEFAULT_TARGET_SAMPLE_RATE_HZ = 60
EGO4D_SAMPLE_RATE = 200
EGO4D_WINDOW_SECONDS = 5
EGO4D_WINDOW_STEPS = EGO4D_SAMPLE_RATE * EGO4D_WINDOW_SECONDS
MMEA_SAMPLE_RATE = 25
EGOEXO4D_SAMPLE_RATE = 200
EGOEXO4D_WINDOW_SECONDS = 5
EGOEXO4D_WINDOW_STEPS = EGOEXO4D_SAMPLE_RATE * EGOEXO4D_WINDOW_SECONDS
ALL_IMU_CHANNEL_SLICES = (((0, 3), (3, 6)),)


@dataclass(frozen=True)
class SequenceDataset:
    name: str
    x_train: list[np.ndarray]
    y_train: np.ndarray
    x_test: list[np.ndarray]
    y_test: np.ndarray
    label_dictionary: dict[str, str]
    channel_axis: int
    original_sample_rate_hz: int
    target_sample_rate_hz: int
    num_classes: int
    selected_channel_slices: tuple[tuple[tuple[int, int], ...], ...]


def _contiguous_label_map(labels: list[str]) -> dict[str, int]:
    unique_labels = sorted(
        set(labels), key=lambda label: int(label) if label.isdigit() else label
    )
    return {label: idx for (idx, label) in enumerate(unique_labels)}


def read_ego4d_split(split_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in Path(split_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        (sample_id, label) = line.split("\t", 1)
        rows.append((sample_id.strip(), label.strip()))
    return rows


def _resample_sequence(
    sequence: np.ndarray,
    *,
    original_sample_rate_hz: int,
    target_sample_rate_hz: int,
    channel_axis: int,
) -> np.ndarray:
    if original_sample_rate_hz <= 0:
        raise ValueError(
            f"original_sample_rate_hz must be positive, got {original_sample_rate_hz}"
        )
    if target_sample_rate_hz <= 0:
        raise ValueError(
            f"target_sample_rate_hz must be positive, got {target_sample_rate_hz}"
        )
    arr = np.asarray(sequence, dtype=np.float32)
    if original_sample_rate_hz == target_sample_rate_hz:
        return arr.astype(np.float32, copy=False)
    time_axis = 1 if channel_axis == 0 else 0
    target_length = max(
        1,
        int(
            arr.shape[time_axis]
            / float(original_sample_rate_hz)
            * target_sample_rate_hz
        ),
    )
    if target_length == arr.shape[time_axis]:
        return arr.astype(np.float32, copy=False)
    return np.asarray(resample(arr, target_length, axis=time_axis), dtype=np.float32)


def load_ego4d_dataset(
    root: Path = DEFAULT_EGO4D_ROOT,
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
) -> SequenceDataset:
    root = Path(root)
    train_rows = read_ego4d_split(root / "train.txt")
    test_rows = read_ego4d_split(root / "test.txt")
    label_to_id = _contiguous_label_map(
        [label for (_, label) in train_rows + test_rows]
    )
    imu_dir = root / "v2" / "processed_imu"

    def load_rows(rows: list[tuple[str, str]]) -> tuple[list[np.ndarray], np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for sample_id, label in rows:
            arr = np.asarray(np.load(imu_dir / f"{sample_id}.npy"), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != 6:
                raise ValueError(
                    f"Expected Ego4D IMU shape [6, T], got {arr.shape} for {sample_id}"
                )
            num_windows = arr.shape[1] // EGO4D_WINDOW_STEPS
            for window_idx in range(num_windows):
                start = window_idx * EGO4D_WINDOW_STEPS
                end = start + EGO4D_WINDOW_STEPS
                xs.append(
                    _resample_sequence(
                        arr[:, start:end],
                        original_sample_rate_hz=EGO4D_SAMPLE_RATE,
                        target_sample_rate_hz=target_sample_rate_hz,
                        channel_axis=0,
                    )
                )
                ys.append(label_to_id[label])
        return (xs, np.asarray(ys, dtype=np.int64))

    (x_train, y_train) = load_rows(train_rows)
    (x_test, y_test) = load_rows(test_rows)
    id_to_label = {str(idx): label for (label, idx) in label_to_id.items()}
    return SequenceDataset(
        name="ego4d",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        label_dictionary=id_to_label,
        channel_axis=0,
        original_sample_rate_hz=EGO4D_SAMPLE_RATE,
        target_sample_rate_hz=target_sample_rate_hz,
        num_classes=len(label_to_id),
        selected_channel_slices=ALL_IMU_CHANNEL_SLICES,
    )


def read_mmea_split(split_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in Path(split_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append((parts[0], int(parts[-1])))
    return rows


def mmea_sensor_path(root: Path, split_sample_path: str) -> Path:
    rel = split_sample_path.replace("yourdataset_path/data/", "", 1)
    return Path(root) / "sensor" / f"{rel}.csv"


def _load_mmea_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(arr, dtype=np.float32)


def _preprocess_mmea_sequence(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected MMEA sensor shape [T, 6], got {arr.shape}")
    acc = arr[:, :3] / 16384.0 * 9.80665
    gyro = arr[:, 3:6] / 16.4 * np.pi / 180.0
    return np.concatenate((acc, gyro), axis=1).astype(np.float32, copy=False)


def _mmea_label_text(split_sample_path: str) -> str:
    rel = split_sample_path.replace("yourdataset_path/data/", "", 1)
    action_name = Path(rel).parent.name
    parts = action_name.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    text = " ".join(parts).strip()
    return text if text else action_name


def load_mmea_dataset(
    root: Path = DEFAULT_MMEA_ROOT,
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
) -> SequenceDataset:
    root = Path(root)
    train_rows = read_mmea_split(root / "train.txt")
    test_rows = read_mmea_split(root / "test.txt")
    label_to_id = _contiguous_label_map(
        [str(label) for (_, label) in train_rows + test_rows]
    )
    label_texts: dict[str, str] = {}
    for sample_path, label in train_rows + test_rows:
        label_texts.setdefault(str(label), _mmea_label_text(sample_path))

    def load_rows(rows: list[tuple[str, int]]) -> tuple[list[np.ndarray], np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for sample_path, label in rows:
            xs.append(
                _resample_sequence(
                    _preprocess_mmea_sequence(
                        _load_mmea_csv(mmea_sensor_path(root, sample_path))
                    ),
                    original_sample_rate_hz=MMEA_SAMPLE_RATE,
                    target_sample_rate_hz=target_sample_rate_hz,
                    channel_axis=1,
                )
            )
            ys.append(label_to_id[str(label)])
        return (xs, np.asarray(ys, dtype=np.int64))

    (x_train, y_train) = load_rows(train_rows)
    (x_test, y_test) = load_rows(test_rows)
    id_to_label = {str(idx): label_texts[label] for (label, idx) in label_to_id.items()}
    return SequenceDataset(
        name="mmea",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        label_dictionary=id_to_label,
        channel_axis=1,
        original_sample_rate_hz=MMEA_SAMPLE_RATE,
        target_sample_rate_hz=target_sample_rate_hz,
        num_classes=len(label_to_id),
        selected_channel_slices=ALL_IMU_CHANNEL_SLICES,
    )


def load_egoexo4d_dataset(
    root: Path = DEFAULT_EGOEXO4D_ROOT,
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
) -> SequenceDataset:
    root = Path(root)
    train_rows = read_ego4d_split(root / "train.txt")
    test_rows = read_ego4d_split(root / "test.txt")
    label_to_id = _contiguous_label_map(
        [label for (_, label) in train_rows + test_rows]
    )
    imu_dir = root / "processed_imu"

    def load_rows(rows: list[tuple[str, str]]) -> tuple[list[np.ndarray], np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for sample_id, label in rows:
            arr = np.asarray(np.load(imu_dir / f"{sample_id}.npy"), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != 6:
                raise ValueError(
                    f"Expected EgoExo4D IMU shape [6, T], got {arr.shape} for {sample_id}"
                )
            num_windows = arr.shape[1] // EGOEXO4D_WINDOW_STEPS
            for window_idx in range(num_windows):
                start = window_idx * EGOEXO4D_WINDOW_STEPS
                end = start + EGOEXO4D_WINDOW_STEPS
                xs.append(
                    _resample_sequence(
                        arr[:, start:end],
                        original_sample_rate_hz=EGOEXO4D_SAMPLE_RATE,
                        target_sample_rate_hz=target_sample_rate_hz,
                        channel_axis=0,
                    )
                )
                ys.append(label_to_id[label])
        return (xs, np.asarray(ys, dtype=np.int64))

    (x_train, y_train) = load_rows(train_rows)
    (x_test, y_test) = load_rows(test_rows)
    id_to_label = {str(idx): label for (label, idx) in label_to_id.items()}
    return SequenceDataset(
        name="egoexo4d",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        label_dictionary=id_to_label,
        channel_axis=0,
        original_sample_rate_hz=EGOEXO4D_SAMPLE_RATE,
        target_sample_rate_hz=target_sample_rate_hz,
        num_classes=len(label_to_id),
        selected_channel_slices=ALL_IMU_CHANNEL_SLICES,
    )


def load_ego_dataset(
    name: str,
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    *,
    ego4d_root: Path = DEFAULT_EGO4D_ROOT,
    mmea_root: Path = DEFAULT_MMEA_ROOT,
    egoexo4d_root: Path = DEFAULT_EGOEXO4D_ROOT,
) -> SequenceDataset:
    if name == "ego4d":
        return load_ego4d_dataset(
            ego4d_root, target_sample_rate_hz=target_sample_rate_hz
        )
    if name == "mmea":
        return load_mmea_dataset(mmea_root, target_sample_rate_hz=target_sample_rate_hz)
    if name == "egoexo4d":
        return load_egoexo4d_dataset(
            egoexo4d_root, target_sample_rate_hz=target_sample_rate_hz
        )
    raise ValueError(f"Unknown ego dataset: {name}")


DEFAULT_OPENPACK_ROOT = Path(os.environ.get("OPENPACK_ROOT", "<PATH_TO_OPENPACK>"))
OPENPACK_ORIGINAL_SAMPLE_RATE_HZ = 30
OPENPACK_PREPROCESSED_ZIP = "preprocessed-IMU-with-operation-labels.zip"
OPENPACK_PREPROCESSED_DIR = "imuWithOperationLabel"
G_TO_M_PER_S2 = 9.80665
DPS_TO_RAD_PER_S = np.pi / 180.0
OPENPACK_SENSOR_TO_PROJECT_SEGMENT = OrderedDict(
    (
        (sensor_id, str(entry["project_segment_name"]))
        for (
            sensor_id,
            entry,
        ) in body_part_mapping.OPENPACK_SENSOR_TO_BODY_PARTS.items()
    )
)
OPENPACK_SENSOR_CONTEXT = "right wrist, left wrist, right upper arm, and left upper arm"
OPENPACK_LABEL_DICTIONARY = OrderedDict(
    {
        "0": "Picking",
        "1": "Relocate Item Label",
        "2": "Assemble Box",
        "3": "Insert Items",
        "4": "Close Box",
        "5": "Attach Box Label",
        "6": "Scan Label",
        "7": "Attach Shipping Label",
        "8": "Put on Back Table",
        "9": "Fill out Order",
    }
)
OPENPACK_ENCODED_NULL_LABEL = 10
OPENPACK_OFFICIAL_OPERATION_ID_TO_LABEL_ID = {
    100: 0,
    200: 1,
    300: 2,
    400: 3,
    500: 4,
    600: 5,
    700: 6,
    800: 7,
    900: 8,
    1000: 9,
    8100: OPENPACK_ENCODED_NULL_LABEL,
}
OPENPACK_CHALLENGE_2022_SPLIT: dict[str, tuple[tuple[str, str], ...]] = {
    "train": tuple(
        (
            (user, session)
            for user in (
                "U0101",
                "U0102",
                "U0103",
                "U0105",
                "U0106",
                "U0107",
                "U0109",
                "U0111",
                "U0202",
                "U0205",
                "U0210",
            )
            for session in ("S0100", "S0200", "S0400", "S0500")
        )
    ),
    "val": (
        ("U0101", "S0300"),
        ("U0103", "S0300"),
        ("U0105", "S0300"),
        ("U0107", "S0300"),
        ("U0109", "S0300"),
        ("U0111", "S0300"),
        ("U0205", "S0300"),
    ),
    "test": (
        ("U0102", "S0300"),
        ("U0106", "S0300"),
        ("U0202", "S0300"),
        ("U0210", "S0300"),
    ),
}


@dataclass(frozen=True)
class OpenPackSession:
    user: str
    session: str
    features: np.ndarray
    operations: np.ndarray
    sensor_segments: list[str]


@dataclass(frozen=True)
class OpenPackDataset:
    name: str
    x_train: np.ndarray | list[np.ndarray]
    y_train: np.ndarray
    x_val: np.ndarray | list[np.ndarray]
    y_val: np.ndarray
    x_test: np.ndarray | list[np.ndarray]
    y_test: np.ndarray
    label_dictionary: dict[str, str]
    channel_axis: int | None
    original_sample_rate_hz: int
    target_sample_rate_hz: int
    num_classes: int
    selected_channel_slices: tuple[tuple[tuple[int, int], ...], ...]
    sensor_to_segment: dict[str, str]
    sensor_context: str
    split_sessions: dict[str, list[str]]
    sample_mode: str
    min_segment_seconds: float
    max_segment_seconds: float | None


def ensure_openpack_preprocessed_extracted(root: Path = DEFAULT_OPENPACK_ROOT) -> Path:
    root = Path(root)
    data_dir = root / OPENPACK_PREPROCESSED_DIR
    if data_dir.is_dir():
        return data_dir
    zip_path = root / OPENPACK_PREPROCESSED_ZIP
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Missing OpenPack preprocessed directory or zip under {root}"
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Extracted {zip_path}, but {data_dir} was not created")
    return data_dir


def session_key(user: str, session: str) -> str:
    return f"{str(user)}-{str(session)}"


def _parse_session_file_name(path: Path) -> tuple[str, str]:
    stem = Path(path).stem
    if "-" not in stem:
        raise ValueError(
            f"Expected OpenPack session file name like U0101-S0100.csv, got {path.name}"
        )
    (user, session) = stem.split("-", maxsplit=1)
    return (user, session)


def _selected_channel_slices(
    sensor_count: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    groups: list[tuple[tuple[int, int], ...]] = []
    for sensor_index in range(int(sensor_count)):
        offset = sensor_index * 6
        groups.append(((offset, offset + 3), (offset + 3, offset + 6)))
    return tuple(groups)


def _feature_column_names(sensors: Iterable[str]) -> list[str]:
    names: list[str] = []
    for sensor in sensors:
        names.extend(
            [
                f"{sensor}/acc_x",
                f"{sensor}/acc_y",
                f"{sensor}/acc_z",
                f"{sensor}/gyro_x",
                f"{sensor}/gyro_y",
                f"{sensor}/gyro_z",
            ]
        )
    return names


def _operation_to_encoded_label(operation: int) -> int:
    operation = int(operation)
    if operation in OPENPACK_OFFICIAL_OPERATION_ID_TO_LABEL_ID:
        return int(OPENPACK_OFFICIAL_OPERATION_ID_TO_LABEL_ID[operation])
    return operation


def load_openpack_session_csv(
    path: Path,
    *,
    sensor_to_segment: OrderedDict[str, str] = OPENPACK_SENSOR_TO_PROJECT_SEGMENT,
) -> OpenPackSession:
    path = Path(path)
    (user, session) = _parse_session_file_name(path)
    sensors = tuple(sensor_to_segment.keys())
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    index = {name: idx for (idx, name) in enumerate(header)}
    column_names = ["operation", *_feature_column_names(sensors)]
    missing = [name for name in column_names if name not in index]
    if missing:
        raise KeyError(f"{path} is missing expected OpenPack columns: {missing[:8]}")
    raw = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=[index[name] for name in column_names],
        dtype=np.float64,
    )
    if raw.ndim == 1:
        raw = raw[None, :]
    operations = np.asarray(
        [_operation_to_encoded_label(value) for value in raw[:, 0]], dtype=np.int64
    )
    features = np.asarray(raw[:, 1:], dtype=np.float32)
    scale = np.asarray([G_TO_M_PER_S2] * 3 + [DPS_TO_RAD_PER_S] * 3, dtype=np.float32)
    features = features * np.tile(scale, len(sensors))
    return OpenPackSession(
        user=user,
        session=session,
        features=features.astype(np.float32, copy=False),
        operations=operations,
        sensor_segments=list(sensor_to_segment.values()),
    )


def _resample_openpack_time_axis(
    x: np.ndarray, original_sample_rate_hz: int, target_sample_rate_hz: int
) -> np.ndarray:
    if int(original_sample_rate_hz) == int(target_sample_rate_hz):
        return x.astype(np.float32, copy=False)
    target_length = max(
        1,
        int(
            round(
                x.shape[0] / float(original_sample_rate_hz) * int(target_sample_rate_hz)
            )
        ),
    )
    if target_length == x.shape[0]:
        return x.astype(np.float32, copy=False)
    return np.asarray(resample(x, target_length, axis=0), dtype=np.float32)


def _pad_window(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.shape[0] >= int(window_size):
        return values[: int(window_size)]
    pad_count = int(window_size) - int(values.shape[0])
    if values.shape[0] == 0:
        return np.zeros((int(window_size), values.shape[1]), dtype=np.float32)
    pad = np.repeat(values[-1:, :], pad_count, axis=0)
    return np.concatenate([values, pad], axis=0).astype(np.float32, copy=False)


def window_openpack_arrays(
    features: np.ndarray,
    operations: np.ndarray,
    *,
    window_size: int,
    stride_size: int,
    include_short_segments: bool = True,
    min_segment_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    operations = np.asarray(operations, dtype=np.int64)
    if features.shape[0] != operations.shape[0]:
        raise ValueError(
            f"features and operations must have equal length, got {features.shape[0]} and {operations.shape[0]}"
        )
    window_size = max(1, int(window_size))
    stride_size = max(1, int(stride_size))
    min_segment_size = max(1, int(min_segment_size))
    windows: list[np.ndarray] = []
    labels: list[int] = []
    start = 0
    while start < operations.shape[0]:
        op = int(operations[start])
        end = start + 1
        while end < operations.shape[0] and int(operations[end]) == op:
            end += 1
        label_id = _operation_to_encoded_label(op)
        segment = features[start:end]
        if (
            0 <= label_id < len(OPENPACK_LABEL_DICTIONARY)
            and segment.shape[0] >= min_segment_size
        ):
            if segment.shape[0] < window_size:
                if include_short_segments:
                    windows.append(_pad_window(segment, window_size))
                    labels.append(label_id)
            else:
                for window_start in range(
                    0, segment.shape[0] - window_size + 1, stride_size
                ):
                    windows.append(segment[window_start : window_start + window_size])
                    labels.append(label_id)
        start = end
    if not windows:
        return (
            np.empty((0, window_size, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
    )


def segment_openpack_arrays(
    features: np.ndarray,
    operations: np.ndarray,
    *,
    min_segment_size: int = 1,
    max_segment_size: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    operations = np.asarray(operations, dtype=np.int64)
    if features.shape[0] != operations.shape[0]:
        raise ValueError(
            f"features and operations must have equal length, got {features.shape[0]} and {operations.shape[0]}"
        )
    min_segment_size = max(1, int(min_segment_size))
    max_segment_size = (
        None if max_segment_size is None else max(1, int(max_segment_size))
    )
    segments: list[np.ndarray] = []
    labels: list[int] = []
    start = 0
    while start < operations.shape[0]:
        op = int(operations[start])
        end = start + 1
        while end < operations.shape[0] and int(operations[end]) == op:
            end += 1
        label_id = _operation_to_encoded_label(op)
        segment = features[start:end]
        is_valid_label = 0 <= label_id < len(OPENPACK_LABEL_DICTIONARY)
        is_long_enough = segment.shape[0] >= min_segment_size
        is_short_enough = (
            max_segment_size is None or segment.shape[0] <= max_segment_size
        )
        if is_valid_label and is_long_enough and is_short_enough:
            segments.append(segment.astype(np.float32, copy=False))
            labels.append(label_id)
        start = end
    return (segments, np.asarray(labels, dtype=np.int64))


def _load_split_arrays(
    data_dir: Path,
    sessions: tuple[tuple[str, str], ...],
    *,
    window_size: int,
    stride_size: int,
    target_sample_rate_hz: int,
    require_all_sessions: bool,
    include_short_segments: bool,
    min_segment_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    loaded_sessions: list[str] = []
    for user, session in sessions:
        path = data_dir / f"{session_key(user, session)}.csv"
        if not path.is_file():
            if require_all_sessions:
                raise FileNotFoundError(f"Missing OpenPack split session file: {path}")
            continue
        session_data = load_openpack_session_csv(path)
        (windows, labels) = window_openpack_arrays(
            session_data.features,
            session_data.operations,
            window_size=window_size,
            stride_size=stride_size,
            include_short_segments=include_short_segments,
            min_segment_size=min_segment_size,
        )
        if target_sample_rate_hz != OPENPACK_ORIGINAL_SAMPLE_RATE_HZ and windows.size:
            windows = np.stack(
                [
                    _resample_openpack_time_axis(
                        window,
                        OPENPACK_ORIGINAL_SAMPLE_RATE_HZ,
                        int(target_sample_rate_hz),
                    )
                    for window in windows
                ]
            ).astype(np.float32, copy=False)
        x_rows.append(windows)
        y_rows.append(labels)
        loaded_sessions.append(session_key(user, session))
    if not x_rows:
        target_length = max(
            1,
            int(
                round(
                    window_size
                    / OPENPACK_ORIGINAL_SAMPLE_RATE_HZ
                    * int(target_sample_rate_hz)
                )
            ),
        )
        return (
            np.empty(
                (0, target_length, len(OPENPACK_SENSOR_TO_PROJECT_SEGMENT) * 6),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
            loaded_sessions,
        )
    return (
        np.concatenate(x_rows, axis=0).astype(np.float32, copy=False),
        np.concatenate(y_rows, axis=0).astype(np.int64, copy=False),
        loaded_sessions,
    )


def _load_split_segments(
    data_dir: Path,
    sessions: tuple[tuple[str, str], ...],
    *,
    target_sample_rate_hz: int,
    require_all_sessions: bool,
    min_segment_size: int,
    max_segment_size: int | None,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    loaded_sessions: list[str] = []
    for user, session in sessions:
        path = data_dir / f"{session_key(user, session)}.csv"
        if not path.is_file():
            if require_all_sessions:
                raise FileNotFoundError(f"Missing OpenPack split session file: {path}")
            continue
        session_data = load_openpack_session_csv(path)
        (segments, labels) = segment_openpack_arrays(
            session_data.features,
            session_data.operations,
            min_segment_size=min_segment_size,
            max_segment_size=max_segment_size,
        )
        if target_sample_rate_hz != OPENPACK_ORIGINAL_SAMPLE_RATE_HZ:
            segments = [
                _resample_openpack_time_axis(
                    segment,
                    OPENPACK_ORIGINAL_SAMPLE_RATE_HZ,
                    int(target_sample_rate_hz),
                )
                for segment in segments
            ]
        x_rows.extend((segment.astype(np.float32, copy=False) for segment in segments))
        y_rows.append(labels)
        loaded_sessions.append(session_key(user, session))
    if not y_rows:
        return (x_rows, np.empty((0,), dtype=np.int64), loaded_sessions)
    return (
        x_rows,
        np.concatenate(y_rows, axis=0).astype(np.int64, copy=False),
        loaded_sessions,
    )


def load_openpack_dataset(
    root: Path = DEFAULT_OPENPACK_ROOT,
    *,
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    sample_mode: str = "segment",
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    require_all_sessions: bool = True,
    include_short_segments: bool = True,
    min_segment_seconds: float = 1.0,
    max_segment_seconds: float | None = 30.0,
) -> OpenPackDataset:
    data_dir = ensure_openpack_preprocessed_extracted(Path(root))
    sample_mode = str(sample_mode).lower()
    if sample_mode not in {"segment", "window"}:
        raise ValueError(
            f"sample_mode must be 'segment' or 'window', got {sample_mode!r}"
        )
    window_size = max(
        1, int(round(float(window_seconds) * OPENPACK_ORIGINAL_SAMPLE_RATE_HZ))
    )
    stride_size = max(
        1, int(round(float(stride_seconds) * OPENPACK_ORIGINAL_SAMPLE_RATE_HZ))
    )
    min_segment_size = max(
        1, int(round(float(min_segment_seconds) * OPENPACK_ORIGINAL_SAMPLE_RATE_HZ))
    )
    max_segment_size = (
        None
        if max_segment_seconds is None
        else max(
            1, int(round(float(max_segment_seconds) * OPENPACK_ORIGINAL_SAMPLE_RATE_HZ))
        )
    )
    if sample_mode == "segment":
        (x_train, y_train, train_sessions) = _load_split_segments(
            data_dir,
            OPENPACK_CHALLENGE_2022_SPLIT["train"],
            target_sample_rate_hz=int(target_sample_rate_hz),
            require_all_sessions=bool(require_all_sessions),
            min_segment_size=min_segment_size,
            max_segment_size=max_segment_size,
        )
        (x_val, y_val, val_sessions) = _load_split_segments(
            data_dir,
            OPENPACK_CHALLENGE_2022_SPLIT["val"],
            target_sample_rate_hz=int(target_sample_rate_hz),
            require_all_sessions=bool(require_all_sessions),
            min_segment_size=min_segment_size,
            max_segment_size=max_segment_size,
        )
        (x_test, y_test, test_sessions) = _load_split_segments(
            data_dir,
            OPENPACK_CHALLENGE_2022_SPLIT["test"],
            target_sample_rate_hz=int(target_sample_rate_hz),
            require_all_sessions=bool(require_all_sessions),
            min_segment_size=min_segment_size,
            max_segment_size=max_segment_size,
        )
    else:
        (x_train, y_train, train_sessions) = _load_split_arrays(
            data_dir,
            OPENPACK_CHALLENGE_2022_SPLIT["train"],
            window_size=window_size,
            stride_size=stride_size,
            target_sample_rate_hz=int(target_sample_rate_hz),
            require_all_sessions=bool(require_all_sessions),
            include_short_segments=bool(include_short_segments),
            min_segment_size=min_segment_size,
        )
        (x_val, y_val, val_sessions) = _load_split_arrays(
            data_dir,
            OPENPACK_CHALLENGE_2022_SPLIT["val"],
            window_size=window_size,
            stride_size=stride_size,
            target_sample_rate_hz=int(target_sample_rate_hz),
            require_all_sessions=bool(require_all_sessions),
            include_short_segments=bool(include_short_segments),
            min_segment_size=min_segment_size,
        )
        (x_test, y_test, test_sessions) = _load_split_arrays(
            data_dir,
            OPENPACK_CHALLENGE_2022_SPLIT["test"],
            window_size=window_size,
            stride_size=stride_size,
            target_sample_rate_hz=int(target_sample_rate_hz),
            require_all_sessions=bool(require_all_sessions),
            include_short_segments=bool(include_short_segments),
            min_segment_size=min_segment_size,
        )
    return OpenPackDataset(
        name="OpenPack",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        label_dictionary=dict(OPENPACK_LABEL_DICTIONARY),
        channel_axis=None,
        original_sample_rate_hz=OPENPACK_ORIGINAL_SAMPLE_RATE_HZ,
        target_sample_rate_hz=int(target_sample_rate_hz),
        num_classes=len(OPENPACK_LABEL_DICTIONARY),
        selected_channel_slices=_selected_channel_slices(
            len(OPENPACK_SENSOR_TO_PROJECT_SEGMENT)
        ),
        sensor_to_segment=dict(OPENPACK_SENSOR_TO_PROJECT_SEGMENT),
        sensor_context=OPENPACK_SENSOR_CONTEXT,
        split_sessions={
            "train": train_sessions,
            "val": val_sessions,
            "test": test_sessions,
        },
        sample_mode=sample_mode,
        min_segment_seconds=float(min_segment_seconds),
        max_segment_seconds=None
        if max_segment_seconds is None
        else float(max_segment_seconds),
    )


DEFAULT_ARRAY_HAR_ROOT = Path(
    os.environ.get("HAR_DATA_ROOT", "<PATH_TO_PREPROCESSED_HAR_DATASETS>")
)


@dataclass(frozen=True)
class ChannelSelection:
    start: int
    end: int
    scale: float = 1.0
    bias: float = 0.0


@dataclass(frozen=True)
class DatasetPreprocessingSpec:
    original_sample_rate_hz: int
    num_classes: int
    selected_channels: tuple[tuple[ChannelSelection, ...], ...]

    @property
    def selected_channel_slices(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(
            (
                tuple(((selection.start, selection.end) for selection in group))
                for group in self.selected_channels
            )
        )


@dataclass(frozen=True)
class ArrayHARDataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    label_dictionary: dict[str, str]
    original_sample_rate_hz: int
    target_sample_rate_hz: int
    num_classes: int
    selected_channel_slices: tuple[tuple[tuple[int, int], ...], ...]


def _group(*selections: ChannelSelection) -> tuple[ChannelSelection, ...]:
    return selections


ARRAY_HAR_PREPROCESSING_SPECS: dict[str, DatasetPreprocessingSpec] = {
    "PAMAP": DatasetPreprocessingSpec(
        original_sample_rate_hz=100,
        num_classes=12,
        selected_channels=(
            _group(ChannelSelection(0, 3), ChannelSelection(3, 6)),
            _group(ChannelSelection(18, 21), ChannelSelection(21, 24)),
            _group(ChannelSelection(9, 12), ChannelSelection(12, 15)),
        ),
    ),
    "USCHAD": DatasetPreprocessingSpec(
        original_sample_rate_hz=100,
        num_classes=12,
        selected_channels=(
            _group(
                ChannelSelection(0, 3, scale=9.80665),
                ChannelSelection(3, 6, scale=np.pi / 180.0),
            ),
        ),
    ),
    "UCIHAR": DatasetPreprocessingSpec(
        original_sample_rate_hz=50,
        num_classes=6,
        selected_channels=(
            _group(ChannelSelection(6, 9, scale=9.80665), ChannelSelection(3, 6)),
        ),
    ),
    "Opp_g": DatasetPreprocessingSpec(
        original_sample_rate_hz=30,
        num_classes=4,
        selected_channels=(
            _group(
                ChannelSelection(0, 3, scale=9.8 / 1000.0),
                ChannelSelection(3, 6, scale=1.0 / 1000.0),
            ),
            _group(
                ChannelSelection(9, 12, scale=9.8 / 1000.0),
                ChannelSelection(12, 15, scale=1.0 / 1000.0),
            ),
            _group(
                ChannelSelection(18, 21, scale=9.8 / 1000.0),
                ChannelSelection(21, 24, scale=1.0 / 1000.0),
            ),
            _group(
                ChannelSelection(27, 30, scale=9.8 / 1000.0),
                ChannelSelection(30, 33, scale=1.0 / 1000.0),
            ),
            _group(
                ChannelSelection(36, 39, scale=9.8 / 1000.0),
                ChannelSelection(39, 42, scale=1.0 / 1000.0),
            ),
        ),
    ),
    "WISDM": DatasetPreprocessingSpec(
        original_sample_rate_hz=20,
        num_classes=18,
        selected_channels=(_group(ChannelSelection(0, 3), ChannelSelection(3, 6)),),
    ),
    "DSADS": DatasetPreprocessingSpec(
        original_sample_rate_hz=25,
        num_classes=19,
        selected_channels=(
            _group(ChannelSelection(0, 3), ChannelSelection(3, 6)),
            _group(ChannelSelection(9, 12), ChannelSelection(12, 15)),
            _group(ChannelSelection(18, 21), ChannelSelection(21, 24)),
            _group(ChannelSelection(27, 30), ChannelSelection(30, 33)),
            _group(ChannelSelection(36, 39), ChannelSelection(39, 42)),
        ),
    ),
    "UTD-MHAD": DatasetPreprocessingSpec(
        original_sample_rate_hz=50,
        num_classes=27,
        selected_channels=(
            _group(
                ChannelSelection(0, 3, scale=9.80665),
                ChannelSelection(3, 6, scale=np.pi / 180.0),
            ),
        ),
    ),
    "w-HAR": DatasetPreprocessingSpec(
        original_sample_rate_hz=250,
        num_classes=7,
        selected_channels=(
            _group(
                ChannelSelection(0, 3, scale=9.80665),
                ChannelSelection(3, 6, scale=np.pi / 180.0),
            ),
        ),
    ),
    "realworld": DatasetPreprocessingSpec(
        original_sample_rate_hz=50,
        num_classes=8,
        selected_channels=(
            _group(ChannelSelection(0, 3), ChannelSelection(3, 6)),
            _group(ChannelSelection(6, 9), ChannelSelection(9, 12)),
            _group(ChannelSelection(12, 15), ChannelSelection(15, 18)),
            _group(ChannelSelection(18, 21), ChannelSelection(21, 24)),
            _group(ChannelSelection(24, 27), ChannelSelection(27, 30)),
            _group(ChannelSelection(30, 33), ChannelSelection(33, 36)),
            _group(ChannelSelection(36, 39), ChannelSelection(39, 42)),
        ),
    ),
    "TNDA-HAR": DatasetPreprocessingSpec(
        original_sample_rate_hz=50,
        num_classes=8,
        selected_channels=(
            _group(ChannelSelection(0, 3), ChannelSelection(3, 6)),
            _group(ChannelSelection(6, 9), ChannelSelection(9, 12)),
            _group(ChannelSelection(12, 15), ChannelSelection(15, 18)),
            _group(ChannelSelection(18, 21), ChannelSelection(21, 24)),
            _group(ChannelSelection(24, 27), ChannelSelection(27, 30)),
        ),
    ),
}


def _normalize_label_dictionary(meta: dict[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in meta.get("label_dictionary", {}).items():
        if isinstance(value, (list, tuple)):
            normalized[str(key)] = " ".join((str(token) for token in value))
        else:
            normalized[str(key)] = str(value)
    return normalized


def _apply_preprocessing(x: np.ndarray, spec: DatasetPreprocessingSpec) -> np.ndarray:
    groups: list[np.ndarray] = []
    for group in spec.selected_channels:
        parts = [
            x[:, :, selection.start : selection.end] * selection.scale + selection.bias
            for selection in group
        ]
        groups.append(np.concatenate(parts, axis=-1))
    return np.concatenate(groups, axis=-1).astype(np.float32, copy=False)


def _resample_array_har_time_axis(
    x: np.ndarray, original_sample_rate_hz: int, target_sample_rate_hz: int
) -> np.ndarray:
    if original_sample_rate_hz <= 0:
        raise ValueError(
            f"original_sample_rate_hz must be positive, got {original_sample_rate_hz}"
        )
    if target_sample_rate_hz <= 0:
        raise ValueError(
            f"target_sample_rate_hz must be positive, got {target_sample_rate_hz}"
        )
    if original_sample_rate_hz == target_sample_rate_hz:
        return x.astype(np.float32, copy=False)
    target_length = max(
        1, int(x.shape[1] / float(original_sample_rate_hz) * target_sample_rate_hz)
    )
    if target_length == x.shape[1]:
        return x.astype(np.float32, copy=False)
    return np.asarray(resample(x, target_length, axis=1), dtype=np.float32)


def list_array_har_datasets(root: Path = DEFAULT_ARRAY_HAR_ROOT) -> list[str]:
    names: list[str] = []
    for dataset_dir in sorted((path for path in Path(root).iterdir() if path.is_dir())):
        if dataset_dir.name.startswith("few_shot"):
            continue
        if (
            dataset_dir.name in ARRAY_HAR_PREPROCESSING_SPECS
            and (dataset_dir / f"{dataset_dir.name}.json").is_file()
        ):
            names.append(dataset_dir.name)
    return names


def load_array_har_dataset(
    root: Path = DEFAULT_ARRAY_HAR_ROOT,
    name: str = "DSADS",
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
) -> ArrayHARDataset:
    dataset_dir = Path(root) / name
    if name not in ARRAY_HAR_PREPROCESSING_SPECS:
        supported = ", ".join(sorted(ARRAY_HAR_PREPROCESSING_SPECS))
        raise ValueError(f"Unsupported HAR dataset {name!r}. Paper datasets: {supported}")
    spec = ARRAY_HAR_PREPROCESSING_SPECS[name]
    meta = json.loads((dataset_dir / f"{name}.json").read_text(encoding="utf-8"))
    x_train_raw = np.asarray(np.load(dataset_dir / "X_train.npy"), dtype=np.float32)
    x_test_raw = np.asarray(np.load(dataset_dir / "X_test.npy"), dtype=np.float32)
    x_train = _resample_array_har_time_axis(
        _apply_preprocessing(x_train_raw, spec),
        spec.original_sample_rate_hz,
        target_sample_rate_hz,
    )
    x_test = _resample_array_har_time_axis(
        _apply_preprocessing(x_test_raw, spec),
        spec.original_sample_rate_hz,
        target_sample_rate_hz,
    )
    return ArrayHARDataset(
        name=name,
        x_train=x_train,
        y_train=np.asarray(np.load(dataset_dir / "y_train.npy"), dtype=np.int64),
        x_test=x_test,
        y_test=np.asarray(np.load(dataset_dir / "y_test.npy"), dtype=np.int64),
        label_dictionary=_normalize_label_dictionary(meta),
        original_sample_rate_hz=spec.original_sample_rate_hz,
        target_sample_rate_hz=int(target_sample_rate_hz),
        num_classes=spec.num_classes,
        selected_channel_slices=spec.selected_channel_slices,
    )


# The 14 unseen HAR datasets reported in the paper.
ARRAY_HAR_DATASETS = tuple(ARRAY_HAR_PREPROCESSING_SPECS)
EGO_HAR_DATASETS = ("ego4d", "mmea", "egoexo4d")
OPENPACK_HAR_DATASETS = ("OpenPack",)
PAPER_HAR_DATASETS = ARRAY_HAR_DATASETS + EGO_HAR_DATASETS + OPENPACK_HAR_DATASETS


def load_har_dataset(
    name: str,
    *,
    target_sample_rate_hz: int = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    array_root: Path = DEFAULT_ARRAY_HAR_ROOT,
    ego4d_root: Path = DEFAULT_EGO4D_ROOT,
    mmea_root: Path = DEFAULT_MMEA_ROOT,
    egoexo4d_root: Path = DEFAULT_EGOEXO4D_ROOT,
    openpack_root: Path = DEFAULT_OPENPACK_ROOT,
    openpack_sample_mode: str = "segment",
    openpack_window_seconds: float = 4.0,
    openpack_stride_seconds: float = 2.0,
    openpack_min_segment_seconds: float = 1.0,
    openpack_max_segment_seconds: float | None = 30.0,
    openpack_require_all_sessions: bool = True,
) -> SequenceDataset | ArrayHARDataset | OpenPackDataset:
    """Load one of the 14 zero-shot HAR datasets evaluated in the paper."""
    if name in ARRAY_HAR_DATASETS:
        return load_array_har_dataset(
            root=Path(array_root),
            name=name,
            target_sample_rate_hz=target_sample_rate_hz,
        )
    if name in EGO_HAR_DATASETS:
        return load_ego_dataset(
            name=name,
            target_sample_rate_hz=target_sample_rate_hz,
            ego4d_root=Path(ego4d_root),
            mmea_root=Path(mmea_root),
            egoexo4d_root=Path(egoexo4d_root),
        )
    if name == "OpenPack":
        return load_openpack_dataset(
            root=Path(openpack_root),
            target_sample_rate_hz=target_sample_rate_hz,
            sample_mode=openpack_sample_mode,
            window_seconds=openpack_window_seconds,
            stride_seconds=openpack_stride_seconds,
            min_segment_seconds=openpack_min_segment_seconds,
            max_segment_seconds=openpack_max_segment_seconds,
            require_all_sessions=openpack_require_all_sessions,
        )
    supported = ", ".join(PAPER_HAR_DATASETS)
    raise ValueError(f"Unknown HAR dataset {name!r}. Paper datasets: {supported}")
