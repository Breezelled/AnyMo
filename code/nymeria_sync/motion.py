from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_BASE_DIR = Path(os.environ.get("ANYMO_DATA_ROOT", "<PATH_TO_NYMERIA>"))
DEFAULT_NYMERIA_PKG_ROOT = Path(os.environ.get("NYMERIA_TOOLS_ROOT", "<PATH_TO_NYMERIA_TOOLS>"))
PART_NAMES = [
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
ATOMIC_TEXT_COL = "Describe my atomic actions"
XSENS_NUM_PARTS = 23
XSENS_DT_NOMINAL_US = 1.0e6 / 240.0
XSENS_DT_TOLERANCE_US = 1000
XSENS_TCORRECT_TOLERANCE_US = 10_000


def _correct_xsens_timestamps(timestamps_us: np.ndarray) -> np.ndarray:
    timestamps_us = timestamps_us.astype(np.int64, copy=True)
    if timestamps_us.size < 2:
        return timestamps_us
    dt_original = timestamps_us[1:] - timestamps_us[:-1]
    invalid = np.abs(dt_original - XSENS_DT_NOMINAL_US) > XSENS_DT_TOLERANCE_US
    if not np.any(invalid):
        return timestamps_us
    dt_corrected = dt_original.copy()
    dt_corrected[invalid] = int(XSENS_DT_NOMINAL_US)
    dt_corrected = np.insert(dt_corrected, 0, 0)
    corrected = timestamps_us[0] + np.cumsum(dt_corrected)
    if np.abs(corrected - timestamps_us)[-1] > XSENS_TCORRECT_TOLERANCE_US:
        raise RuntimeError('corrected Xsens timestamps exceed tolerance')
    return corrected.astype(np.int64, copy=False)


def _correct_xsens_quaternions(segment_qwxyz: np.ndarray) -> np.ndarray:
    q = segment_qwxyz.reshape(-1, XSENS_NUM_PARTS, 4).astype(np.float32, copy=True)
    qn = np.linalg.norm(q, axis=-1)
    invalid = qn < 0.1
    if not np.any(invalid):
        return q.reshape(-1, XSENS_NUM_PARTS * 4)
    for part_idx in range(XSENS_NUM_PARTS):
        if qn[0, part_idx] < 0.5:
            q[0, part_idx] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for frame_idx in range(1, q.shape[0]):
        for part_idx in range(XSENS_NUM_PARTS):
            if qn[frame_idx, part_idx] < 0.5:
                q[frame_idx, part_idx] = q[frame_idx - 1, part_idx]
    return q.reshape(-1, XSENS_NUM_PARTS * 4)


def _ensure_imports(nymeria_pkg_root: str) -> None:
    pkg_root = str(Path(nymeria_pkg_root))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _load_xsens_arrays(sample_dir: Path, nymeria_pkg_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xdata_path = sample_dir / "body" / "xdata.npz"
    if not xdata_path.is_file():
        raise FileNotFoundError(f"missing {xdata_path}")

    # Prefer Nymeria's BodyDataProvider because it already applies the dataset's
    # timestamp and quaternion corrections on load. Fall back to local correction
    # logic if the body provider import path is unavailable.
    try:
        _ensure_imports(nymeria_pkg_root)
        from nymeria.body_motion_provider import BodyDataProvider

        body_dp = BodyDataProvider(
            npzfile=str(xdata_path),
            glbfile=str(sample_dir / "body" / "__skip_glb__.glb"),
        )
        xsens_data = body_dp.xsens_data
        timestamps_us = np.asarray(xsens_data["timestamps_us"], dtype=np.int64)
        segment_txyz = np.asarray(xsens_data["segment_tXYZ"], dtype=np.float32)
        segment_qwxyz = np.asarray(xsens_data["segment_qWXYZ"], dtype=np.float32)
        return timestamps_us, segment_txyz, segment_qwxyz
    except Exception:
        with np.load(xdata_path, allow_pickle=True) as data:
            timestamps_us = _correct_xsens_timestamps(np.asarray(data["timestamps_us"], dtype=np.int64))
            segment_txyz = np.asarray(data["segment_tXYZ"], dtype=np.float32)
            segment_qwxyz = _correct_xsens_quaternions(np.asarray(data["segment_qWXYZ"], dtype=np.float32))
        return timestamps_us, segment_txyz, segment_qwxyz


def _load_head_recording(sample_dir: Path, nymeria_pkg_root: str):
    _ensure_imports(nymeria_pkg_root)
    from nymeria.recording_data_provider import create_recording_data_provider

    rec = create_recording_data_provider(sample_dir / "recording_head")
    if rec is None or rec.vrs_dp is None:
        raise RuntimeError("recording_head VRS provider not available")
    return rec


def _device_time_ns_to_global_time_ns(vrs_dp, t_ns_device: int) -> int:
    candidates = [
        "convert_from_device_time_to_timecode_ns",
        "convert_from_device_time_ns_to_timecode_ns",
        "convert_from_device_time_to_time_code_ns",
        "convert_from_device_time_ns_to_time_code_ns",
        "convert_from_device_to_timecode_ns",
        "convert_from_device_to_time_code_ns",
    ]
    for name in candidates:
        fn = getattr(vrs_dp, name, None)
        if fn is None:
            continue
        return int(fn(int(t_ns_device)))
    raise AttributeError(
        "VRS provider does not expose a supported device->timecode conversion method"
    )


def _crop_imu_payload(imu_npz: Any, overlap_mask: np.ndarray, grid_ns: np.ndarray) -> dict[str, Any]:
    cropped: dict[str, Any] = {}
    raw_len = int(overlap_mask.shape[0])
    new_len = int(grid_ns.shape[0])
    for key in imu_npz.files:
        value = imu_npz[key]
        if key == "t_ns_global_timecode":
            cropped[key] = grid_ns.astype(np.int64, copy=False)
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == raw_len:
            cropped[key] = value[overlap_mask]
        else:
            cropped[key] = value
    return cropped


def _find_nearest_indices(sorted_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    if sorted_values.ndim != 1:
        raise ValueError("sorted_values must be 1D")
    if sorted_values.size == 0:
        raise ValueError("sorted_values must be non-empty")
    idx_rr = np.searchsorted(sorted_values, query_values, side="left")
    idx_rr = np.clip(idx_rr, 0, sorted_values.size - 1)
    idx_ll = np.clip(idx_rr - 1, 0, sorted_values.size - 1)
    choose_left = np.abs(sorted_values[idx_ll] - query_values) <= np.abs(
        sorted_values[idx_rr] - query_values
    )
    return np.where(choose_left, idx_ll, idx_rr).astype(np.int64, copy=False)


def _build_xsens_payload(
    timestamps_us: np.ndarray,
    segment_txyz: np.ndarray,
    segment_qwxyz: np.ndarray,
    grid_ns: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if segment_txyz.ndim != 2 or segment_txyz.shape[1] != XSENS_NUM_PARTS * 3:
        raise RuntimeError(f"unexpected segment_tXYZ shape: {segment_txyz.shape}")
    if segment_qwxyz.ndim != 2 or segment_qwxyz.shape[1] != XSENS_NUM_PARTS * 4:
        raise RuntimeError(f"unexpected segment_qWXYZ shape: {segment_qwxyz.shape}")
    if timestamps_us.shape[0] != segment_txyz.shape[0] or timestamps_us.shape[0] != segment_qwxyz.shape[0]:
        raise RuntimeError("xdata length mismatch among timestamps/segment_tXYZ/segment_qWXYZ")

    timestamps_ns = timestamps_us.astype(np.int64, copy=False) * 1000
    src_idx = _find_nearest_indices(timestamps_ns, grid_ns)
    src_ts_ns = timestamps_ns[src_idx]
    tdiff_ns = (grid_ns - src_ts_ns).astype(np.int64, copy=False)

    payload = {
        "t_ns_global_timecode": grid_ns.astype(np.int64, copy=False),
        "part_names": np.array(PART_NAMES, dtype=object),
        "segment_tXYZ_60hz": segment_txyz[src_idx].reshape(-1, XSENS_NUM_PARTS, 3).astype(np.float32, copy=False),
        "segment_qWXYZ_60hz": segment_qwxyz[src_idx].reshape(-1, XSENS_NUM_PARTS, 4).astype(np.float32, copy=False),
        "timestamps_us_src": timestamps_us[src_idx].astype(np.int64, copy=False),
        "xsens_src_idx_for_60hz": src_idx,
        "xsens_tdiff_ns": tdiff_ns,
    }
    meta = {
        "xsens_frame_count_raw": int(timestamps_us.shape[0]),
        "xsens_sync_tdiff_abs_ms_mean": float(np.mean(np.abs(tdiff_ns) / 1e6)),
        "xsens_sync_tdiff_abs_ms_p95": float(np.percentile(np.abs(tdiff_ns) / 1e6, 95)),
        "xsens_global_start_ns": int(timestamps_ns[0]),
        "xsens_global_end_ns": int(timestamps_ns[-1]),
    }
    return payload, meta


def _sync_atomic_action_to_grid(
    sample_dir: Path,
    grid_ns: np.ndarray,
    nymeria_pkg_root: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = sample_dir / "narration" / "atomic_action.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")

    df = pd.read_csv(csv_path)
    if "start_time" not in df.columns or "end_time" not in df.columns:
        raise RuntimeError("atomic_action.csv missing start_time/end_time")
    if ATOMIC_TEXT_COL not in df.columns:
        raise RuntimeError(f"atomic_action.csv missing {ATOMIC_TEXT_COL!r}")

    rec = _load_head_recording(sample_dir, nymeria_pkg_root)
    out_rows: list[dict[str, Any]] = []
    out_of_range = 0
    empty_after_mapping = 0

    grid_start = int(grid_ns[0])
    grid_end = int(grid_ns[-1])
    grid_len = int(grid_ns.shape[0])

    for row in df.to_dict(orient="records"):
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        start_t_ns_global = _device_time_ns_to_global_time_ns(rec.vrs_dp, int(round(start_time * 1e9)))
        end_t_ns_global = _device_time_ns_to_global_time_ns(rec.vrs_dp, int(round(end_time * 1e9)))

        if end_t_ns_global < grid_start or start_t_ns_global > grid_end:
            out_of_range += 1
            continue

        start_idx = int(np.searchsorted(grid_ns, start_t_ns_global, side="left"))
        end_idx = int(np.searchsorted(grid_ns, end_t_ns_global, side="right") - 1)
        start_idx = int(np.clip(start_idx, 0, grid_len - 1))
        end_idx = int(np.clip(end_idx, 0, grid_len - 1))

        if start_idx > end_idx:
            empty_after_mapping += 1
            continue

        out_rows.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_t_ns_global": int(start_t_ns_global),
                "end_t_ns_global": int(end_t_ns_global),
                "start_time": start_time,
                "end_time": end_time,
                ATOMIC_TEXT_COL: row[ATOMIC_TEXT_COL],
            }
        )

    out_df = pd.DataFrame(out_rows)
    meta = {
        "atomic_action_num_rows_raw": int(len(df)),
        "atomic_action_num_rows_kept": int(len(out_df)),
        "atomic_action_num_rows_out_of_range": int(out_of_range),
        "atomic_action_num_rows_empty_after_mapping": int(empty_after_mapping),
    }
    return out_df, meta


def sync_sample_to_multimodal_60hz(
    sample_dir: str | Path,
    *,
    nymeria_pkg_root: str | Path = DEFAULT_NYMERIA_PKG_ROOT,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    imu_npz_path = sample_dir / "synced_6imu_60hz.npz"
    atomic_csv_path = sample_dir / "narration" / "atomic_action.csv"
    xdata_path = sample_dir / "body" / "xdata.npz"

    if not imu_npz_path.is_file():
        raise FileNotFoundError(f"missing {imu_npz_path}")
    if not atomic_csv_path.is_file():
        raise FileNotFoundError(f"missing {atomic_csv_path}")
    if not xdata_path.is_file():
        raise FileNotFoundError(f"missing {xdata_path}")

    with np.load(imu_npz_path, allow_pickle=True) as imu_npz:
        imu_grid_ns = np.asarray(imu_npz["t_ns_global_timecode"], dtype=np.int64)

        if imu_grid_ns.ndim != 1 or imu_grid_ns.size == 0:
            raise RuntimeError(f"invalid IMU canonical grid shape: {imu_grid_ns.shape}")

        timestamps_us, segment_txyz, segment_qwxyz = _load_xsens_arrays(sample_dir, str(nymeria_pkg_root))
        timestamps_ns = timestamps_us.astype(np.int64, copy=False) * 1000
        overlap_mask = (imu_grid_ns >= timestamps_ns[0]) & (imu_grid_ns <= timestamps_ns[-1])
        grid_ns = imu_grid_ns[overlap_mask]
        if grid_ns.size == 0:
            raise RuntimeError("no overlap between synced_6imu_60hz grid and Xsens timestamps")

        imu_payload = _crop_imu_payload(imu_npz, overlap_mask, grid_ns)

    xsens_payload, xsens_meta = _build_xsens_payload(timestamps_us, segment_txyz, segment_qwxyz, grid_ns)
    atomic_df, atomic_meta = _sync_atomic_action_to_grid(sample_dir, grid_ns, str(nymeria_pkg_root))

    out_dir = sample_dir / "multimodal_sync_60hz"
    out_dir.mkdir(parents=True, exist_ok=True)
    xsens_out = out_dir / "xsens_60hz.npz"
    atomic_out = out_dir / "atomic_action_60hz.csv"
    imu_out = out_dir / "sync_imu_60hz.npz"
    meta_out = out_dir / "sync_meta.json"

    np.savez_compressed(xsens_out, **xsens_payload)
    np.savez_compressed(imu_out, **imu_payload)
    atomic_df.to_csv(atomic_out, index=False)

    meta: dict[str, Any] = {
        "sample_dir": sample_dir.name,
        "target_hz": 60,
        "canonical_grid_start_ns": int(grid_ns[0]),
        "canonical_grid_end_ns": int(grid_ns[-1]),
        "seq_len_60hz": int(grid_ns.shape[0]),
        "imu_grid_start_ns_raw": int(imu_grid_ns[0]),
        "imu_grid_end_ns_raw": int(imu_grid_ns[-1]),
        "imu_seq_len_60hz_raw": int(imu_grid_ns.shape[0]),
        "saved_xsens_npz": str(xsens_out),
        "saved_atomic_csv": str(atomic_out),
        "saved_imu_npz": str(imu_out),
        "status": "ok",
        **xsens_meta,
        **atomic_meta,
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _build_summary_row(sample_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_dir": sample_dir.name,
        "sample_path": str(sample_dir),
        "seq_len_60hz": meta.get("seq_len_60hz", np.nan),
        "xsens_sync_tdiff_abs_ms_mean": meta.get("xsens_sync_tdiff_abs_ms_mean", np.nan),
        "xsens_sync_tdiff_abs_ms_p95": meta.get("xsens_sync_tdiff_abs_ms_p95", np.nan),
        "atomic_action_num_rows_kept": meta.get("atomic_action_num_rows_kept", np.nan),
        "status": meta.get("status", "unknown"),
        "saved_xsens_npz": meta.get("saved_xsens_npz", ""),
        "saved_atomic_csv": meta.get("saved_atomic_csv", ""),
        "saved_imu_npz": meta.get("saved_imu_npz", ""),
    }


def _process_one_sample(task: dict[str, Any]) -> dict[str, Any]:
    sample_dir = Path(task["sample_dir"])
    out_dir = sample_dir / "multimodal_sync_60hz"
    xsens_out = out_dir / "xsens_60hz.npz"
    atomic_out = out_dir / "atomic_action_60hz.csv"
    imu_out = out_dir / "sync_imu_60hz.npz"
    meta_out = out_dir / "sync_meta.json"
    try:
        if (not task["overwrite"]) and xsens_out.exists() and atomic_out.exists() and imu_out.exists() and meta_out.exists():
            meta = json.loads(meta_out.read_text(encoding="utf-8"))
        else:
            meta = sync_sample_to_multimodal_60hz(
                sample_dir=sample_dir,
                nymeria_pkg_root=task["nymeria_pkg_root"],
            )
        return {"ok": True, "row": _build_summary_row(sample_dir, meta), "error": None}
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "row": None,
            "error": {
                "sample_dir": str(sample_dir),
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync Nymeria Xsens pose and atomic action to the existing 60Hz 6IMU grid")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--nymeria-pkg-root", type=Path, default=DEFAULT_NYMERIA_PKG_ROOT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir

    sample_dirs: list[Path] = []
    for imu_npz in sorted(base_dir.rglob("synced_6imu_60hz.npz")):
        sample_dir = imu_npz.parent
        if (sample_dir / "body" / "xdata.npz").is_file() and (sample_dir / "narration" / "atomic_action.csv").is_file():
            sample_dirs.append(sample_dir)

    if args.limit_samples is not None:
        sample_dirs = sample_dirs[: int(args.limit_samples)]

    print(
        f"Samples to process: {len(sample_dirs)} | overwrite={args.overwrite} | workers={args.workers}"
    )
    if not sample_dirs:
        print("No eligible samples found.")
        return 0

    tasks = [
        {
            "sample_dir": str(sample_dir),
            "overwrite": bool(args.overwrite),
            "nymeria_pkg_root": str(args.nymeria_pkg_root),
        }
        for sample_dir in sample_dirs
    ]

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if args.workers == 1:
        pbar = tqdm(tasks, desc="Nymeria multimodal sync@60Hz", unit="sample")
        for task in pbar:
            result = _process_one_sample(task)
            if result["ok"]:
                summary_rows.append(result["row"])
            else:
                errors.append(result["error"])
            pbar.set_postfix(success=len(summary_rows), fail=len(errors))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(_process_one_sample, task) for task in tasks]
            pbar = tqdm(total=len(futures), desc="Nymeria multimodal sync@60Hz", unit="sample")
            for fut in as_completed(futures):
                result = fut.result()
                if result["ok"]:
                    summary_rows.append(result["row"])
                else:
                    errors.append(result["error"])
                pbar.update(1)
                pbar.set_postfix(success=len(summary_rows), fail=len(errors))
            pbar.close()

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["sample_dir", "sample_path"]).reset_index(drop=True)

    summary_csv = base_dir / "nymeria_multimodal_60hz_summary.csv"
    errors_json = base_dir / "nymeria_multimodal_60hz_errors.json"
    summary_df.to_csv(summary_csv, index=False)
    errors_json.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nProcessing complete")
    print("Successful samples:", len(summary_df))
    print("Failed samples:", len(errors))
    print("\nSaved summary:")
    print("  ", summary_csv)
    print("  ", errors_json)
    if errors:
        print("\nFirst 10 failed samples:")
        for err in errors[:10]:
            print("  ", err["sample_dir"], err["message"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
