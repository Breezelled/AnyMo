#!/usr/bin/env python3
"""Generate AnyMo geometry-aware IMU signals over body-surface placements."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_BASE_DIR = Path(os.environ.get("ANYMO_DATA_ROOT", "<PATH_TO_NYMERIA>"))
DEFAULT_SELECTION_NPZ = (
    DEFAULT_BASE_DIR / "body_surface_candidates_with_local_frames.npz"
)
DEFAULT_OUTPUT_NAME = "geometry_aware_imu_60hz.npz"
DEFAULT_OUTPUT_SUMMARY_CSV = (
    DEFAULT_BASE_DIR / "geometry_aware_imu_summary.csv"
)
DEFAULT_OUTPUT_SUMMARY_JSON = (
    DEFAULT_BASE_DIR / "geometry_aware_imu_summary.json"
)
DEFAULT_CANDIDATE_NAME = "body_surface_placements"
DEFAULT_SAMPLE_RATE = 60.0

SELECTION_MODE = "body_surface"
SELECTION_TOP_K = 2
HARDWARE_NOISE_SOURCE = "mean_head_lwrist_rwrist"
REAL_NOISE_PRIOR_SITES = ("head", "lwrist", "rwrist")


@lru_cache(maxsize=1)
def _load_base_module():
    from simulation import wimusim_utils as base

    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate geometry-aware WIMUSim signals over body-surface placements for 23 anatomical segments."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--selection-npz", type=Path, default=DEFAULT_SELECTION_NPZ)
    parser.add_argument("--candidate-name", type=str, default=DEFAULT_CANDIDATE_NAME)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-range",
        type=str,
        default=None,
        help="Zero-based half-open slice over the sorted eligible samples, e.g. 0:3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real1-style-ratio", type=float, default=0.5)
    parser.add_argument("--surface-jitter-inplane-max-deg", type=float, default=180.0)
    parser.add_argument("--surface-jitter-tilt-max-deg", type=float, default=0.0)
    parser.add_argument("--output-summary-csv", type=Path, default=DEFAULT_OUTPUT_SUMMARY_CSV)
    parser.add_argument("--output-summary-json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON)
    args = parser.parse_args()

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive when provided")
    if args.sample_range is not None:
        try:
            start_text, end_text = args.sample_range.split(":", 1)
            start = int(start_text)
            end = int(end_text)
        except Exception:
            parser.error("--sample-range must be in start:end format, e.g. 0:3")
        if start < 0 or end < 0 or end <= start:
            parser.error("--sample-range must satisfy 0 <= start < end")
        args.sample_range = (start, end)
    if not 0.0 <= float(args.real1_style_ratio) <= 1.0:
        parser.error("--real1-style-ratio must be in [0, 1]")
    if float(args.surface_jitter_inplane_max_deg) < 0.0:
        parser.error("--surface-jitter-inplane-max-deg must be non-negative")
    if float(args.surface_jitter_tilt_max_deg) < 0.0:
        parser.error("--surface-jitter-tilt-max-deg must be non-negative")
    return args


def quaternion_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_vec = q[..., 1:]
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quaternion_conjugate_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).copy()
    q[..., 1:] *= -1.0
    return q


def quaternion_inverse_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    conj = quaternion_conjugate_np(q)
    denom = np.sum(q * q, axis=-1, keepdims=True)
    return conj / np.clip(denom, 1e-8, None)


def standardize_quaternion_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    signs = np.where(q[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    q = q * signs
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def vector_in_parent_frame_np(
    p_parent: np.ndarray,
    q_parent: np.ndarray,
    p_child: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(p_child, dtype=np.float32) - np.asarray(p_parent, dtype=np.float32)
    return quaternion_apply_np(quaternion_inverse_np(q_parent), delta)


def build_site_vertex_records(selection_payload: dict[str, np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    segment_names = [str(x) for x in selection_payload["segment_names"].tolist()]
    selected_vertex_ids = np.asarray(selection_payload["selected_vertex_ids"], dtype=np.int32)
    selected_vertex_mask = np.asarray(selection_payload["selected_vertex_mask"], dtype=bool)
    selected_vertex_weights = np.asarray(selection_payload["selected_vertex_weights"], dtype=np.float32)
    template_vertices = np.asarray(selection_payload["template_vertices"], dtype=np.float32)
    top1_masks = np.asarray(selection_payload["candidate_dominant_masks"], dtype=bool)
    rotation_templates = np.asarray(selection_payload["selected_vertex_rotation_matrix"], dtype=np.float32)
    surface_normals = np.asarray(selection_payload["selected_vertex_normal"], dtype=np.float32)
    surface_tangents = np.asarray(selection_payload["selected_vertex_tangent"], dtype=np.float32)

    site_rows: dict[str, list[dict[str, Any]]] = {segment_name: [] for segment_name in segment_names}
    for seg_idx, segment_name in enumerate(segment_names):
        valid = selected_vertex_mask[seg_idx]
        ids = selected_vertex_ids[seg_idx][valid]
        weights = selected_vertex_weights[seg_idx][valid]
        for placement_index, (vertex_id, weight) in enumerate(zip(ids.tolist(), weights.tolist())):
            site_rows[segment_name].append(
                {
                    "site_name": segment_name,
                    "segment_name": segment_name,
                    "parent_segment": segment_name,
                    "vertex_id": int(vertex_id),
                    "placement_index_within_site": int(placement_index),
                    "template_vertex_xyz": template_vertices[int(vertex_id)].astype(np.float32),
                    "selection_weight": float(weight),
                    "is_dominant_top1": bool(top1_masks[seg_idx, int(vertex_id)]),
                    "rotation_matrix_template": rotation_templates[seg_idx, placement_index].astype(np.float32),
                    "surface_normal": surface_normals[seg_idx, placement_index].astype(np.float32),
                    "surface_tangent": surface_tangents[seg_idx, placement_index].astype(np.float32),
                }
            )
    return site_rows


def load_selection_payload(selection_npz: Path) -> dict[str, np.ndarray]:
    with np.load(selection_npz, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def discover_sample_records(base_dir: Path, max_samples: int | None, sample_range: tuple[int, int] | None):
    base = _load_base_module()
    records = []
    for sample_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        multimodal_dir = sample_dir / "multimodal_sync_60hz"
        xsens_npz = multimodal_dir / "xsens_60hz.npz"
        imu_npz = multimodal_dir / "sync_imu_60hz.npz"
        mesh_npz = multimodal_dir / "momentum_mesh_60hz.npz"
        if not (sample_dir / "metadata.json").exists():
            continue
        if not xsens_npz.exists() or not imu_npz.exists() or not mesh_npz.exists():
            continue
        fake_name, script = base.read_metadata(sample_dir)
        records.append(
            base.SampleRecord(
                sample_dir=sample_dir.name,
                fake_name=fake_name,
                script=script,
                multimodal_dir=str(multimodal_dir),
                xsens_npz=str(xsens_npz),
                imu_npz=str(imu_npz),
                output_npz=str(multimodal_dir / DEFAULT_OUTPUT_NAME),
            )
        )
    records = sorted(records, key=lambda r: (r.fake_name, r.sample_dir))
    if sample_range is not None:
        start, end = sample_range
        records = records[start:end]
    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        raise RuntimeError("No eligible samples found after discovery and slicing")
    return records


def load_mesh_sample(record) -> dict[str, np.ndarray]:
    mesh_path = Path(record.multimodal_dir) / "momentum_mesh_60hz.npz"
    with np.load(mesh_path, allow_pickle=False) as data:
        return {
            "t_ns": np.asarray(data["t_ns_global_timecode"], dtype=np.int64),
            "posed_vertices": np.asarray(data["posed_vertices_60hz"], dtype=np.float32),
            "template_vertices": np.asarray(data["template_vertices"], dtype=np.float32),
        }


def estimate_local_offset_from_vertex(
    vertex_world: np.ndarray,
    parent_position: np.ndarray,
    parent_quaternion: np.ndarray,
) -> np.ndarray:
    local = vector_in_parent_frame_np(
        np.asarray(parent_position, dtype=np.float32),
        standardize_quaternion_np(parent_quaternion),
        np.asarray(vertex_world, dtype=np.float32),
    )
    rp = np.mean(local, axis=0).astype(np.float32)
    if not np.isfinite(rp).all():
        raise ValueError("estimated rp contains non-finite values")
    return rp


def build_balanced_style_labels(
    num_items: int,
    real1_style_ratio: float,
    seed: int,
) -> np.ndarray:
    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if num_items == 0:
        return np.empty((0,), dtype="<U5")
    real1_count = int(math.floor(num_items * float(real1_style_ratio) + 0.5))
    real1_count = min(max(real1_count, 0), num_items)
    if num_items >= 2 and 0.0 < real1_style_ratio < 1.0:
        real1_count = min(max(real1_count, 1), num_items - 1)

    labels = np.full(num_items, "real2", dtype="<U5")
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_items)
    labels[order[:real1_count]] = "real1"
    return labels


def build_sample_output_payload(
    *,
    x: np.ndarray,
    t_ns: np.ndarray,
    sample_rate_hz: float,
    placement_rows: list[dict[str, Any]],
    sample_dir: str,
) -> dict[str, np.ndarray]:
    if x.shape[0] != len(placement_rows):
        raise ValueError("x first dimension must match number of placement rows")

    site_names = np.asarray([row["site_name"] for row in placement_rows], dtype=object)
    segment_names = np.asarray([row["segment_name"] for row in placement_rows], dtype=object)
    vertex_ids = np.asarray([row["vertex_id"] for row in placement_rows], dtype=np.int32)
    placement_index = np.asarray(
        [row["placement_index_within_site"] for row in placement_rows], dtype=np.int32
    )
    style_prior = np.asarray([row["style_name"] for row in placement_rows], dtype=object)
    hardware_noise_source = np.asarray(
        [row.get("hardware_noise_source", HARDWARE_NOISE_SOURCE) for row in placement_rows],
        dtype=object,
    )
    template_vertex_xyz = np.asarray(
        [row["template_vertex_xyz"] for row in placement_rows], dtype=np.float32
    )
    selection_weight = np.asarray(
        [row["selection_weight"] for row in placement_rows], dtype=np.float32
    )
    is_dominant_top1 = np.asarray(
        [row["is_dominant_top1"] for row in placement_rows], dtype=bool
    )
    rotation_matrix_template = np.asarray(
        [row["rotation_matrix_template"] for row in placement_rows], dtype=np.float32
    )
    rotation_matrix_final = np.asarray(
        [row["rotation_matrix_final"] for row in placement_rows], dtype=np.float32
    )
    surface_normal = np.asarray(
        [row["surface_normal"] for row in placement_rows], dtype=np.float32
    )
    surface_tangent = np.asarray(
        [row["surface_tangent"] for row in placement_rows], dtype=np.float32
    )
    surface_jitter_inplane_deg = np.asarray(
        [row["surface_jitter_inplane_deg"] for row in placement_rows], dtype=np.float32
    )
    surface_jitter_tilt_x_deg = np.asarray(
        [row["surface_jitter_tilt_x_deg"] for row in placement_rows], dtype=np.float32
    )
    surface_jitter_tilt_y_deg = np.asarray(
        [row["surface_jitter_tilt_y_deg"] for row in placement_rows], dtype=np.float32
    )

    offsets: list[list[int]] = []
    start = 0
    site_order = list(dict.fromkeys(site_names.tolist()))
    for site_name in site_order:
        count = int(np.sum(site_names == site_name))
        offsets.append([start, count])
        start += count

    return {
        "x": np.asarray(x, dtype=np.float32),
        "t_ns_global_timecode": np.asarray(t_ns, dtype=np.int64),
        "sample_rate_hz": np.asarray(sample_rate_hz, dtype=np.float32),
        "site_order": np.asarray(site_order, dtype=object),
        "site_offsets": np.asarray(offsets, dtype=np.int32),
        "site_names": site_names,
        "segment_names": segment_names,
        "vertex_ids": vertex_ids,
        "placement_index_within_site": placement_index,
        "style_prior": style_prior,
        "hardware_noise_source": hardware_noise_source,
        "selection_mode": np.asarray(SELECTION_MODE, dtype=object),
        "selection_top_k": np.asarray(SELECTION_TOP_K, dtype=np.int32),
        "template_vertex_xyz": template_vertex_xyz,
        "selection_weight": selection_weight,
        "is_dominant_top1": is_dominant_top1,
        "rotation_matrix_template": rotation_matrix_template,
        "rotation_matrix_final": rotation_matrix_final,
        "surface_normal": surface_normal,
        "surface_tangent": surface_tangent,
        "surface_jitter_inplane_deg": surface_jitter_inplane_deg,
        "surface_jitter_tilt_x_deg": surface_jitter_tilt_x_deg,
        "surface_jitter_tilt_y_deg": surface_jitter_tilt_y_deg,
        "source": np.asarray("wimusim", dtype=object),
        "alignment_target": np.asarray(
            "geometry_aware_body_surface_imu",
            dtype=object,
        ),
        "sample_dir": np.asarray(sample_dir, dtype=object),
    }


def save_output_payload(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_path.with_suffix("")
    if output_path.exists():
        output_path.unlink()
    if bundle_dir.exists():
        for old_path in bundle_dir.glob("*.npy"):
            old_path.unlink()
    else:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    for key, value in payload.items():
        arr = np.asarray(value)
        np.save(bundle_dir / f"{key}.npy", arr, allow_pickle=(arr.dtype == object))


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def rotation_matrix_to_euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_euler("XYZ", degrees=False).astype(np.float32)


def apply_surface_constrained_jitter(
    rotation_matrix_template: np.ndarray,
    rng: np.random.Generator,
    inplane_max_deg: float,
    tilt_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    inplane_deg = float(rng.uniform(-float(inplane_max_deg), float(inplane_max_deg)))
    tilt_x_deg = float(rng.uniform(-float(tilt_max_deg), float(tilt_max_deg)))
    tilt_y_deg = float(rng.uniform(-float(tilt_max_deg), float(tilt_max_deg)))
    r_inplane = rotation_matrix_z(np.deg2rad(inplane_deg))
    r_tilt_x = rotation_matrix_x(np.deg2rad(tilt_x_deg))
    r_tilt_y = rotation_matrix_y(np.deg2rad(tilt_y_deg))
    rotation_matrix_final = (r_tilt_y @ r_tilt_x @ r_inplane @ np.asarray(rotation_matrix_template, dtype=np.float32)).astype(np.float32)
    angles_deg = np.asarray([inplane_deg, tilt_x_deg, tilt_y_deg], dtype=np.float32)
    return rotation_matrix_final, angles_deg


def average_style_noise_prior(style_prior: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    missing = [site_name for site_name in REAL_NOISE_PRIOR_SITES if site_name not in style_prior]
    if missing:
        raise KeyError(f"style_prior missing real noise source sites: {missing}")
    sa_values = [np.asarray(style_prior[site_name]["sa_mean"], dtype=np.float32) for site_name in REAL_NOISE_PRIOR_SITES]
    sg_values = [np.asarray(style_prior[site_name]["sg_mean"], dtype=np.float32) for site_name in REAL_NOISE_PRIOR_SITES]
    return {
        "sa_mean": np.mean(np.stack(sa_values, axis=0), axis=0).astype(np.float32),
        "sg_mean": np.mean(np.stack(sg_values, axis=0), axis=0).astype(np.float32),
    }


def build_hardware_for_placements(
    *,
    placement_rows: list[dict[str, Any]],
    style_noise_priors: dict[str, dict[str, np.ndarray]],
    generate_default_H_configs,
    rng: np.random.Generator,
) -> dict[str, Any]:
    configs = generate_default_H_configs([row["imu_name"] for row in placement_rows])
    for row in placement_rows:
        imu_name = row["imu_name"]
        site_prior = style_noise_priors[row["style_name"]]
        sa_mean = np.asarray(site_prior["sa_mean"], dtype=np.float32)
        sg_mean = np.asarray(site_prior["sg_mean"], dtype=np.float32)
        sa_scale = np.maximum(sa_mean * 0.20, np.full(3, 1e-3, dtype=np.float32))
        sg_scale = np.maximum(sg_mean * 0.20, np.full(3, 5e-4, dtype=np.float32))
        configs["ba"][imu_name] = np.zeros(3, dtype=np.float32)
        configs["bg"][imu_name] = np.zeros(3, dtype=np.float32)
        configs["sa"][imu_name] = np.clip(
            rng.normal(loc=sa_mean, scale=sa_scale, size=3),
            1e-3,
            0.3,
        ).astype(np.float32)
        configs["sg"][imu_name] = np.clip(
            rng.normal(loc=sg_mean, scale=sg_scale, size=3),
            1e-3,
            0.3,
        ).astype(np.float32)
    return configs


def stack_simulated_output(
    simulated_dict: dict[str, tuple[Any, Any]],
    placement_rows: list[dict[str, Any]],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for row in placement_rows:
        acc, gyro = simulated_dict[row["imu_name"]]
        parts.append(
            np.concatenate(
                [
                    acc.detach().cpu().numpy().astype(np.float32),
                    gyro.detach().cpu().numpy().astype(np.float32),
                ],
                axis=1,
            )
        )
    return np.stack(parts, axis=0)


def build_site_style_counts(placement_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    site_order = list(dict.fromkeys(row["site_name"] for row in placement_rows))
    counts = {site_name: {"real1": 0, "real2": 0} for site_name in site_order}
    for row in placement_rows:
        counts[row["site_name"]][row["style_name"]] += 1
    return counts


def process_record(
    *,
    record,
    args: argparse.Namespace,
    site_vertex_records: dict[str, list[dict[str, Any]]],
    subject_body_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base = _load_base_module()
    WIMUSim, generate_default_H_configs, xsens_consts = base.load_wimusim_modules()

    output_path = Path(record.multimodal_dir) / DEFAULT_OUTPUT_NAME
    bundle_dir = output_path.with_suffix("")
    metrics_path = output_path.with_suffix(".metrics.json")
    if (output_path.exists() or bundle_dir.is_dir()) and not args.overwrite:
        row = {
            "candidate_name": args.candidate_name,
            "sample_dir": record.sample_dir,
            "status": "skipped_existing",
            "output_path": str(output_path),
            "num_placements": None,
            "real1_count": None,
            "real2_count": None,
            "runtime_sec": 0.0,
        }
        return row, None

    started = time.time()
    failed_placements: list[dict[str, Any]] = []
    try:
        import torch

        imu_sample = base.load_real_imu_sample(record)
        xsens_sample = base.load_xsens_sample(record)
        mesh_sample = load_mesh_sample(record)
        shared_valid = base.build_shared_valid_mask(imu_sample, xsens_sample)
        if int(shared_valid.sum()) < 32:
            raise RuntimeError(f"Too few shared valid frames ({int(shared_valid.sum())}) after filtering.")

        body_state = subject_body_cache.get(record.fake_name)
        if body_state is None:
            body_state = base.build_body_from_xsens(xsens_sample, shared_valid, xsens_consts)
            subject_body_cache[record.fake_name] = {
                "rp": {k: v.copy() for k, v in body_state["rp"].items()},
                "rom_dict": {k: v.copy() for k, v in body_state["rom_dict"].items()},
            }

        dynamics_state = base.build_dynamics_from_xsens(
            xsens_sample=xsens_sample,
            mask=shared_valid,
            sample_rate=args.sample_rate,
            xsens_consts=xsens_consts,
        )

        valid_rows: list[dict[str, Any]] = []
        for site_name, site_rows in site_vertex_records.items():
            for src_row in site_rows:
                row = dict(src_row)
                try:
                    parent_idx = xsens_sample["part_index"][row["parent_segment"]]
                    vertex_world = mesh_sample["posed_vertices"][shared_valid, row["vertex_id"]]
                    parent_pos = xsens_sample["segment_t"][shared_valid, parent_idx]
                    parent_quat = xsens_sample["segment_q"][shared_valid, parent_idx]
                    row["rp"] = estimate_local_offset_from_vertex(
                        vertex_world=vertex_world,
                        parent_position=parent_pos,
                        parent_quaternion=parent_quat,
                    )
                    row["ro"] = np.zeros(3, dtype=np.float32)
                    row["imu_name"] = f"{site_name}_{row['placement_index_within_site']:03d}"
                    valid_rows.append(row)
                except Exception as exc:  # pragma: no cover - exercised in integration only
                    failed_placements.append(
                        {
                            "site_name": row["site_name"],
                            "vertex_id": row["vertex_id"],
                            "error": str(exc),
                        }
                    )

        if not valid_rows:
            raise RuntimeError("No valid placements remained after rp estimation.")
        for site_name in site_vertex_records:
            if not any(row["site_name"] == site_name for row in valid_rows):
                raise RuntimeError(f"All placements failed for site {site_name}.")

        seed_base = int(zlib.crc32(record.sample_dir.encode("utf-8")))
        style_labels = build_balanced_style_labels(
            num_items=len(valid_rows),
            real1_style_ratio=float(args.real1_style_ratio),
            seed=args.seed + seed_base,
        )
        for row, style_name in zip(valid_rows, style_labels.tolist()):
            row["style_name"] = style_name
            row["hardware_noise_source"] = HARDWARE_NOISE_SOURCE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        style_priors = {
            "real1": base.estimate_hardware_style_prior(imu_sample, shared_valid, device_suffix="imu_1202_1"),
            "real2": base.estimate_hardware_style_prior(imu_sample, shared_valid, device_suffix="imu_1202_2"),
        }
        style_noise_priors = {
            style_name: average_style_noise_prior(style_prior)
            for style_name, style_prior in style_priors.items()
        }
        hardware_rng = np.random.default_rng(args.seed + seed_base + 1)
        hardware_state = build_hardware_for_placements(
            placement_rows=valid_rows,
            style_noise_priors=style_noise_priors,
            generate_default_H_configs=generate_default_H_configs,
            rng=hardware_rng,
        )
        orientation_rng = np.random.default_rng(args.seed + seed_base + 2)
        for row in valid_rows:
            rotation_matrix_final, angles_deg = apply_surface_constrained_jitter(
                row["rotation_matrix_template"],
                rng=orientation_rng,
                inplane_max_deg=float(args.surface_jitter_inplane_max_deg),
                tilt_max_deg=float(args.surface_jitter_tilt_max_deg),
            )
            row["rotation_matrix_final"] = rotation_matrix_final
            row["ro"] = rotation_matrix_to_euler_xyz(rotation_matrix_final)
            row["surface_jitter_inplane_deg"] = float(angles_deg[0])
            row["surface_jitter_tilt_x_deg"] = float(angles_deg[1])
            row["surface_jitter_tilt_y_deg"] = float(angles_deg[2])
        placement_rp = {(row["parent_segment"], row["imu_name"]): row["rp"] for row in valid_rows}
        placement_ro = {(row["parent_segment"], row["imu_name"]): row["ro"] for row in valid_rows}

        env = WIMUSim(
            B=WIMUSim.Body(
                rp=subject_body_cache[record.fake_name]["rp"],
                rom_dict=subject_body_cache[record.fake_name]["rom_dict"],
                device=device,
                requires_grad=False,
            ),
            D=WIMUSim.Dynamics(
                orientation=dynamics_state["orientation"],
                translation=dynamics_state["translation"],
                sample_rate=float(args.sample_rate),
                data_type="tensor",
                device=device,
                requires_grad=False,
            ),
            P=WIMUSim.Placement(
                rp=placement_rp,
                ro=placement_ro,
                device=device,
                requires_grad=False,
            ),
            H=WIMUSim.Hardware(
                **hardware_state,
                device=device,
                requires_grad=False,
            ),
            device=device,
            dataset_name="XSENS",
        )
        simulated_dict = env.simulate(mode="parameterise")
        x = stack_simulated_output(simulated_dict, valid_rows)
        mask_t_ns = imu_sample["t_ns"][: len(shared_valid)][shared_valid]
        payload = build_sample_output_payload(
            x=x,
            t_ns=mask_t_ns,
            sample_rate_hz=float(args.sample_rate),
            placement_rows=valid_rows,
            sample_dir=record.sample_dir,
        )
        save_output_payload(output_path, payload)

        metrics = {
            "sample_dir": record.sample_dir,
            "fake_name": record.fake_name,
            "script": record.script,
            "output_path": str(output_path),
            "num_total_placements": int(len(valid_rows)),
            "site_style_counts": build_site_style_counts(valid_rows),
            "hardware_noise_source": HARDWARE_NOISE_SOURCE,
            "hardware_noise_priors": {
                style_name: {key: value.tolist() for key, value in prior.items()}
                for style_name, prior in style_noise_priors.items()
            },
            "failed_placements": failed_placements,
            "runtime_sec": float(time.time() - started),
        }
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        real1_count = sum(1 for row in valid_rows if row["style_name"] == "real1")
        real2_count = len(valid_rows) - real1_count
        row = {
            "candidate_name": args.candidate_name,
            "sample_dir": record.sample_dir,
            "status": "ok",
            "output_path": str(output_path),
            "num_placements": int(len(valid_rows)),
            "real1_count": int(real1_count),
            "real2_count": int(real2_count),
            "failed_placements": int(len(failed_placements)),
            "runtime_sec": float(time.time() - started),
        }
        return row, metrics
    except Exception as exc:  # pragma: no cover - exercised in integration only
        row = {
            "candidate_name": args.candidate_name,
            "sample_dir": record.sample_dir,
            "status": "error",
            "output_path": str(output_path),
            "num_placements": 0,
            "real1_count": 0,
            "real2_count": 0,
            "failed_placements": int(len(failed_placements)),
            "runtime_sec": float(time.time() - started),
            "error": str(exc),
        }
        return row, {
            "sample_dir": record.sample_dir,
            "output_path": str(output_path),
            "failed_placements": failed_placements,
            "error": str(exc),
            "runtime_sec": float(time.time() - started),
        }


def write_summary(rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(
        json.dumps({"rows": rows, "metrics": metrics_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    selection_payload = load_selection_payload(args.selection_npz)
    site_vertex_records = build_site_vertex_records(selection_payload)
    records = discover_sample_records(
        base_dir=args.base_dir,
        max_samples=args.max_samples,
        sample_range=args.sample_range,
    )

    subject_body_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    record_iter = records
    if tqdm is not None:
        record_iter = tqdm(records, desc="Geometry-aware IMU simulation", unit="sample")

    for record in record_iter:
        row, metrics = process_record(
            record=record,
            args=args,
            site_vertex_records=site_vertex_records,
            subject_body_cache=subject_body_cache,
        )
        rows.append(row)
        if metrics is not None:
            metrics_rows.append(metrics)
        message = (
            f"{record.sample_dir}: status={row['status']} "
            f"placements={row.get('num_placements')} output={row['output_path']}"
        )
        if tqdm is not None and hasattr(record_iter, "write"):
            record_iter.write(message)
        else:
            print(message)

    write_summary(rows, metrics_rows, args.output_summary_csv, args.output_summary_json)
    ok = sum(1 for row in rows if row["status"] == "ok")
    skipped = sum(1 for row in rows if row["status"] == "skipped_existing")
    failed = sum(1 for row in rows if row["status"] == "error")
    print(f"ok={ok} skipped_existing={skipped} failed={failed}")
    print(f"summary_csv={args.output_summary_csv}")
    print(f"summary_json={args.output_summary_json}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
