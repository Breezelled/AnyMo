"""Shared Nymeria-to-WIMUSim loading and kinematics utilities.

These helpers are used by the final dense body-surface simulator. They are kept
separate from the earlier joint/placement exploration entry points.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REAL_STREAM_ORDER = [
    "head:imu_1202_1",
    "head:imu_1202_2",
    "lwrist:imu_1202_1",
    "lwrist:imu_1202_2",
    "rwrist:imu_1202_1",
    "rwrist:imu_1202_2",
]
VIRTUAL_STREAM_ORDER = ["head", "lwrist", "rwrist"]
REAL_FEAT_ORDER = ["ax", "ay", "az", "gx", "gy", "gz"]
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class SampleRecord:
    sample_dir: str
    fake_name: str
    script: str
    multimodal_dir: str
    xsens_npz: str
    imu_npz: str
    output_npz: str


def load_wimusim_modules():
    wimusim_root = Path(__file__).resolve().parents[1] / "WIMUSim"
    if str(wimusim_root) not in sys.path:
        sys.path.insert(0, str(wimusim_root))
    try:
        from wimusim.wimusim import WIMUSim
        from wimusim.utils import generate_default_H_configs
        from dataset_configs.xsens import consts as xsens_consts
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Install the bundled WIMUSim package and its PyBullet/PyTorch3D dependencies first."
        ) from exc
    return WIMUSim, generate_default_H_configs, xsens_consts


def read_metadata(sample_dir: Path) -> tuple[str, str]:
    meta_path = sample_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fake_name = meta.get("fake_name")
    script = meta.get("script")
    if not isinstance(fake_name, str) or not fake_name:
        raise ValueError(f"Missing fake_name in {meta_path}")
    if not isinstance(script, str) or not script:
        raise ValueError(f"Missing script in {meta_path}")
    return fake_name, script


def _stream_feature_indices(feature_cols: list[str], stream_order: list[str]) -> dict[str, list[int]]:
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    result: dict[str, list[int]] = {}
    for stream_name in stream_order:
        prefix = stream_name.replace(":", "_")
        result[stream_name] = [feature_index[f"{prefix}_{feat}"] for feat in REAL_FEAT_ORDER]
    return result


def load_real_imu_sample(record: SampleRecord) -> dict[str, Any]:
    with np.load(record.imu_npz, allow_pickle=True) as npz:
        x = np.asarray(npz["x"], dtype=np.float32)
        feature_cols = [str(value) for value in npz["feature_cols"].tolist()]
        stream_order = [str(value) for value in npz["stream_order"].tolist()]
        acc_valid = np.asarray(npz["acc_valid"], dtype=bool)
        gyro_valid = np.asarray(npz["gyro_valid"], dtype=bool)
        t_ns = np.asarray(npz["t_ns_global_timecode"], dtype=np.int64)
    if stream_order != REAL_STREAM_ORDER:
        raise RuntimeError(f"{record.imu_npz} stream_order mismatch: {stream_order}")
    feat_idx = _stream_feature_indices(feature_cols, stream_order)
    stream_values = {stream: x[:, feat_idx[stream]] for stream in stream_order}
    stream_valid = {
        stream: (
            acc_valid[:, stream_order.index(stream)]
            & gyro_valid[:, stream_order.index(stream)]
            & np.isfinite(stream_values[stream]).all(axis=1)
        )
        for stream in stream_order
    }
    return {
        "x": x,
        "t_ns": t_ns,
        "feature_cols": feature_cols,
        "stream_order": stream_order,
        "stream_values": stream_values,
        "stream_valid": stream_valid,
    }


def load_xsens_sample(record: SampleRecord) -> dict[str, Any]:
    with np.load(record.xsens_npz, allow_pickle=True) as npz:
        part_names = [str(value) for value in npz["part_names"].tolist()]
        segment_t = np.asarray(npz["segment_tXYZ_60hz"], dtype=np.float32)
        segment_q = np.asarray(npz["segment_qWXYZ_60hz"], dtype=np.float32)
        t_ns = np.asarray(npz["t_ns_global_timecode"], dtype=np.int64)
    part_index = {name: idx for idx, name in enumerate(part_names)}
    expected_parts = {
        "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
        "R_Shoulder", "R_UpperArm", "R_Forearm", "R_Hand",
        "L_Shoulder", "L_UpperArm", "L_Forearm", "L_Hand",
        "R_UpperLeg", "R_LowerLeg", "R_Foot", "R_Toe",
        "L_UpperLeg", "L_LowerLeg", "L_Foot", "L_Toe",
    }
    missing = sorted(expected_parts - set(part_index))
    if missing:
        raise RuntimeError(f"{record.xsens_npz} missing Xsens parts: {missing}")
    return {
        "part_names": part_names,
        "part_index": part_index,
        "segment_t": segment_t,
        "segment_q": segment_q,
        "t_ns": t_ns,
    }


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
    denom = np.sum(q * q, axis=-1, keepdims=True)
    return quaternion_conjugate_np(q) / np.clip(denom, 1e-8, None)


def quaternion_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(np.asarray(q1, dtype=np.float32), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(q2, dtype=np.float32), -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def standardize_quaternion_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q * np.where(q[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return (q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def relative_quaternion_np(q_parent: np.ndarray, q_child: np.ndarray) -> np.ndarray:
    return standardize_quaternion_np(quaternion_multiply_np(quaternion_inverse_np(q_parent), q_child))


def vector_in_parent_frame_np(p_parent: np.ndarray, q_parent: np.ndarray, p_child: np.ndarray) -> np.ndarray:
    delta = np.asarray(p_child, dtype=np.float32) - np.asarray(p_parent, dtype=np.float32)
    return quaternion_apply_np(quaternion_inverse_np(q_parent), delta)


def build_shared_valid_mask(imu_sample: dict[str, Any], xsens_sample: dict[str, Any]) -> np.ndarray:
    imu_len = min(len(imu_sample["t_ns"]), xsens_sample["segment_t"].shape[0])
    mask = np.ones(imu_len, dtype=bool)
    for stream_name in REAL_STREAM_ORDER:
        mask &= imu_sample["stream_valid"][stream_name][:imu_len]
    xsens_indices = [xsens_sample["part_index"][name] for name in xsens_sample["part_names"]]
    mask &= np.isfinite(xsens_sample["segment_t"][:imu_len, xsens_indices]).all(axis=(1, 2))
    mask &= np.isfinite(xsens_sample["segment_q"][:imu_len, xsens_indices]).all(axis=(1, 2))
    return mask


def build_body_from_xsens(xsens_sample: dict[str, Any], mask: np.ndarray, xsens_consts) -> dict[str, Any]:
    part_index = xsens_sample["part_index"]
    segment_t = xsens_sample["segment_t"][mask]
    segment_q = standardize_quaternion_np(xsens_sample["segment_q"][mask])
    rp: dict[tuple[str, str], np.ndarray] = {("BASE", "Pelvis"): np.zeros(3, dtype=np.float32)}
    for parent_joint, child_joint in xsens_consts.JOINT_PARENT_CHILD_PAIRS:
        if parent_joint == "BASE":
            continue
        rel_vec = vector_in_parent_frame_np(
            segment_t[:, part_index[parent_joint]],
            segment_q[:, part_index[parent_joint]],
            segment_t[:, part_index[child_joint]],
        )
        rp[(parent_joint, child_joint)] = rel_vec.mean(axis=0).astype(np.float32)
    return {"rp": rp, "rom_dict": xsens_consts.JOINT_ROM_DICT}


def build_dynamics_from_xsens(
    xsens_sample: dict[str, Any], mask: np.ndarray, sample_rate: float, xsens_consts
) -> dict[str, Any]:
    part_index = xsens_sample["part_index"]
    segment_t = xsens_sample["segment_t"][mask]
    segment_q = standardize_quaternion_np(xsens_sample["segment_q"][mask])
    orientation: dict[str, np.ndarray] = {
        "BASE": segment_q[:, part_index["Pelvis"]].astype(np.float32),
        "Pelvis": np.repeat(IDENTITY_QUAT[None, :], mask.sum(), axis=0),
    }
    for parent_joint, child_joint in xsens_consts.JOINT_PARENT_CHILD_PAIRS:
        if parent_joint != "BASE":
            orientation[child_joint] = relative_quaternion_np(
                segment_q[:, part_index[parent_joint]], segment_q[:, part_index[child_joint]]
            )
    return {
        "translation": {"XYZ": segment_t[:, part_index["Pelvis"]].astype(np.float32)},
        "orientation": orientation,
        "sample_rate": float(sample_rate),
        "data_type": "tensor",
    }


def _select_quiet_mask(acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    if acc.shape[0] < 8:
        return np.ones(acc.shape[0], dtype=bool)
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    acc_dev = np.abs(acc_mag - float(np.median(acc_mag)))
    quiet = (gyro_mag <= float(np.quantile(gyro_mag, 0.35))) & (acc_dev <= float(np.quantile(acc_dev, 0.45)))
    if int(quiet.sum()) < max(16, acc.shape[0] // 10):
        quiet = gyro_mag <= float(np.quantile(gyro_mag, 0.50))
    if int(quiet.sum()) < 8:
        quiet = np.ones(acc.shape[0], dtype=bool)
    return quiet


def _noise_std_from_diff(values: np.ndarray, min_std: float, max_std: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.full(3, min_std, dtype=np.float32)
    std = np.std(np.diff(values, axis=0), axis=0) / math.sqrt(2.0)
    return np.clip(std.astype(np.float32), min_std, max_std)


def estimate_hardware_style_prior(
    imu_sample: dict[str, Any], mask: np.ndarray, device_suffix: str
) -> dict[str, dict[str, np.ndarray]]:
    prior: dict[str, dict[str, np.ndarray]] = {}
    for site_name in VIRTUAL_STREAM_ORDER:
        arr = np.asarray(imu_sample["stream_values"][f"{site_name}:{device_suffix}"][mask], dtype=np.float32)
        acc, gyro = arr[:, :3], arr[:, 3:]
        quiet = _select_quiet_mask(acc, gyro)
        quiet_acc, quiet_gyro = acc[quiet], gyro[quiet]
        gyro_bias_mean = quiet_gyro.mean(axis=0).astype(np.float32)
        gyro_bias_scale = np.maximum(quiet_gyro.std(axis=0).astype(np.float32) * 0.25, 1e-3)
        acc_noise = _noise_std_from_diff(quiet_acc, min_std=0.002, max_std=0.15)
        gyro_noise = _noise_std_from_diff(quiet_gyro, min_std=0.001, max_std=0.15)
        prior[site_name] = {
            "ba_mean": np.zeros(3, dtype=np.float32),
            "ba_scale": np.maximum(acc_noise * 0.25, 0.002).astype(np.float32),
            "bg_mean": gyro_bias_mean,
            "bg_scale": gyro_bias_scale.astype(np.float32),
            "sa_mean": acc_noise,
            "sg_mean": gyro_noise,
        }
    return prior
