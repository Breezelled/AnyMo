
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

IMU_STREAMS = {
    "imu_1202_1": "1202-1",
    "imu_1202_2": "1202-2",
}
POS_ORDER = ["head", "lwrist", "rwrist"]
STREAM_ORDER = [
    ("head", "imu_1202_1"),
    ("head", "imu_1202_2"),
    ("lwrist", "imu_1202_1"),
    ("lwrist", "imu_1202_2"),
    ("rwrist", "imu_1202_1"),
    ("rwrist", "imu_1202_2"),
]
FEAT_ORDER = ["ax", "ay", "az", "gx", "gy", "gz"]


def _ensure_imports(nymeria_pkg_root: str) -> None:
    pkg_root = str(Path(nymeria_pkg_root))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _query_imu_motiondata_at_global_time(rec, t_ns_global: int, stream_id_str: str):
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    if rec is None or rec.vrs_dp is None:
        return None, None
    sid = StreamId(stream_id_str)
    if not rec.vrs_dp.check_stream_is_active(sid):
        return None, None
    t_dev = rec.vrs_dp.convert_from_timecode_to_device_time_ns(int(t_ns_global))
    motion = rec.vrs_dp.get_imu_data_by_time_ns(
        sid, int(t_dev), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
    )
    tdiff_ns = int(t_dev) - int(motion.capture_timestamp_ns)
    return motion, tdiff_ns


def _get_raw_imu_len_from_rec(rec, stream_id_str: str):
    from projectaria_tools.core.sensor_data import TimeDomain
    from projectaria_tools.core.stream_id import StreamId

    if rec is None or rec.vrs_dp is None:
        return np.nan
    sid = StreamId(stream_id_str)
    if not rec.vrs_dp.check_stream_is_active(sid):
        return np.nan
    ts = rec.vrs_dp.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME)
    return int(len(ts))


def sync_sample_to_60hz_npz(
    sample_dir: str | Path,
    *,
    target_hz: int = 60,
    save_csv_copy: bool = False,
    nymeria_pkg_root: str | Path = DEFAULT_NYMERIA_PKG_ROOT,
    show_inner_progress: bool = False,
) -> dict[str, Any]:
    _ensure_imports(str(nymeria_pkg_root))
    from nymeria.data_provider import NymeriaDataProvider

    sample_dir = Path(sample_dir)
    dp = NymeriaDataProvider(
        sequence_rootdir=sample_dir,
        load_head=True,
        load_wrist=True,
        load_observer=False,
        load_body=False,
    )

    rec_map = {
        "head": dp.recording_head,
        "lwrist": dp.recording_lwrist,
        "rwrist": dp.recording_rwrist,
    }

    raw_lens: dict[str, float] = {}
    for pos in POS_ORDER:
        rec = rec_map[pos]
        raw_lens[f"{pos}_1202_1_raw_len"] = _get_raw_imu_len_from_rec(rec, "1202-1")
        raw_lens[f"{pos}_1202_2_raw_len"] = _get_raw_imu_len_from_rec(rec, "1202-2")
        raw_lens[f"{pos}_raw_len_sum2"] = float(
            np.nansum([raw_lens[f"{pos}_1202_1_raw_len"], raw_lens[f"{pos}_1202_2_raw_len"]])
        )

    t0_ns, t1_ns = map(int, dp.timespan_ns)
    step_ns = int(round(1e9 / float(target_hz)))
    grid_ns = np.arange(t0_ns, t1_ns + 1, step_ns, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    time_iter = (
        tqdm(grid_ns, desc=f"{sample_dir.name}", leave=False, unit="tick")
        if show_inner_progress
        else grid_ns
    )
    for t_ns_global in time_iter:
        row = {"t_ns_global_timecode": int(t_ns_global)}
        for pos_name in POS_ORDER:
            rec = rec_map[pos_name]
            for imu_name, sid in IMU_STREAMS.items():
                prefix = f"{pos_name}_{imu_name}"
                motion, tdiff_ns = _query_imu_motiondata_at_global_time(rec, int(t_ns_global), sid)
                if motion is None:
                    for k in [
                        "ax",
                        "ay",
                        "az",
                        "gx",
                        "gy",
                        "gz",
                        "capture_t_ns",
                        "query_tdiff_ns",
                    ]:
                        row[f"{prefix}_{k}"] = np.nan
                    row[f"{prefix}_acc_valid"] = False
                    row[f"{prefix}_gyro_valid"] = False
                    continue
                a = list(motion.accel_msec2)
                g = list(motion.gyro_radsec)
                row[f"{prefix}_ax"], row[f"{prefix}_ay"], row[f"{prefix}_az"] = a
                row[f"{prefix}_gx"], row[f"{prefix}_gy"], row[f"{prefix}_gz"] = g
                row[f"{prefix}_capture_t_ns"] = int(motion.capture_timestamp_ns)
                row[f"{prefix}_query_tdiff_ns"] = (
                    int(tdiff_ns) if tdiff_ns is not None else np.nan
                )
                row[f"{prefix}_acc_valid"] = bool(motion.accel_valid)
                row[f"{prefix}_gyro_valid"] = bool(motion.gyro_valid)
        rows.append(row)

    synced_df = pd.DataFrame(rows)

    feature_cols: list[str] = []
    for pos_name, imu_name in STREAM_ORDER:
        prefix = f"{pos_name}_{imu_name}"
        for feat in FEAT_ORDER:
            feature_cols.append(f"{prefix}_{feat}")
    X = synced_df[feature_cols].to_numpy(dtype=np.float32, copy=True)

    query_tdiff_cols = [f"{pos}_{imu}_query_tdiff_ns" for pos, imu in STREAM_ORDER]
    capture_t_cols = [f"{pos}_{imu}_capture_t_ns" for pos, imu in STREAM_ORDER]
    acc_valid_cols = [f"{pos}_{imu}_acc_valid" for pos, imu in STREAM_ORDER]
    gyro_valid_cols = [f"{pos}_{imu}_gyro_valid" for pos, imu in STREAM_ORDER]

    qdiff = synced_df[query_tdiff_cols].to_numpy(dtype=np.float32, copy=True)
    cap_t = synced_df[capture_t_cols].to_numpy(dtype=np.float64, copy=True)
    acc_valid = synced_df[acc_valid_cols].to_numpy(dtype=np.bool_, copy=True)
    gyro_valid = synced_df[gyro_valid_cols].to_numpy(dtype=np.bool_, copy=True)

    out_npz = sample_dir / "synced_6imu_60hz.npz"
    out_meta = sample_dir / "synced_6imu_60hz.meta.json"
    out_csv = sample_dir / "synced_6imu_60hz.csv.gz"

    np.savez_compressed(
        out_npz,
        x=X,
        t_ns_global_timecode=grid_ns,
        query_tdiff_ns=qdiff,
        capture_t_ns=cap_t,
        acc_valid=acc_valid,
        gyro_valid=gyro_valid,
        feature_cols=np.array(feature_cols, dtype=object),
        stream_order=np.array([f"{p}:{i}" for p, i in STREAM_ORDER], dtype=object),
        feat_order=np.array(FEAT_ORDER, dtype=object),
    )

    if save_csv_copy:
        synced_df.to_csv(out_csv, index=False, compression="gzip")

    meta: dict[str, Any] = {
        "sample_dir": sample_dir.name,
        "sync_method": "NymeriaDataProvider_TIME_CODE_plus_CLOSEST_on_target_grid",
        "target_hz": int(target_hz),
        "t0_ns": int(t0_ns),
        "t1_ns": int(t1_ns),
        "overlap_sec": float((t1_ns - t0_ns) / 1e9),
        "step_ns": int(step_ns),
        "synced_seq_len": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "x_shape": [int(X.shape[0]), int(X.shape[1])],
        "saved_npz": str(out_npz),
        "saved_csv_gz": str(out_csv) if save_csv_copy else None,
        "raw_3pos_equal_sum2": bool(
            raw_lens["head_raw_len_sum2"]
            == raw_lens["lwrist_raw_len_sum2"]
            == raw_lens["rwrist_raw_len_sum2"]
        ),
        "synced_6streams_equal": True,
        **raw_lens,
    }

    for idx, (pos_name, imu_name) in enumerate(STREAM_ORDER):
        vals_ms = np.abs(qdiff[:, idx]) / 1e6
        meta[f"{pos_name}_{imu_name}_query_tdiff_abs_ms_mean"] = float(np.nanmean(vals_ms))
        meta[f"{pos_name}_{imu_name}_query_tdiff_abs_ms_p95"] = float(
            np.nanpercentile(vals_ms, 95)
        )

    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _build_summary_row(sample_dir: Path, meta: dict[str, Any], out_npz: Path, out_meta: Path):
    return {
        "sample_dir": sample_dir.name,
        "sample_path": str(sample_dir),
        "head_raw_len_sum2": meta.get("head_raw_len_sum2", np.nan),
        "lwrist_raw_len_sum2": meta.get("lwrist_raw_len_sum2", np.nan),
        "rwrist_raw_len_sum2": meta.get("rwrist_raw_len_sum2", np.nan),
        "head_1202_1_raw_len": meta.get("head_1202_1_raw_len", np.nan),
        "head_1202_2_raw_len": meta.get("head_1202_2_raw_len", np.nan),
        "lwrist_1202_1_raw_len": meta.get("lwrist_1202_1_raw_len", np.nan),
        "lwrist_1202_2_raw_len": meta.get("lwrist_1202_2_raw_len", np.nan),
        "rwrist_1202_1_raw_len": meta.get("rwrist_1202_1_raw_len", np.nan),
        "rwrist_1202_2_raw_len": meta.get("rwrist_1202_2_raw_len", np.nan),
        "synced_seq_len_60hz": meta.get("synced_seq_len", np.nan),
        "sync_overlap_sec": meta.get("overlap_sec", np.nan),
        "raw_3pos_equal_sum2": bool(meta.get("raw_3pos_equal_sum2", False)),
        "synced_6streams_equal": bool(meta.get("synced_6streams_equal", True)),
        "saved_npz": meta.get("saved_npz", str(out_npz)),
        "saved_meta": str(out_meta),
    }


def _process_one_sample(task: dict[str, Any]) -> dict[str, Any]:
    sample_dir = Path(task["sample_dir"])
    out_npz = sample_dir / "synced_6imu_60hz.npz"
    out_meta = sample_dir / "synced_6imu_60hz.meta.json"
    try:
        if (not task["overwrite"]) and out_npz.exists() and out_meta.exists():
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
        else:
            meta = sync_sample_to_60hz_npz(
                sample_dir=sample_dir,
                target_hz=int(task["target_hz"]),
                save_csv_copy=bool(task["save_csv_copy"]),
                nymeria_pkg_root=task["nymeria_pkg_root"],
                show_inner_progress=bool(task.get("show_inner_progress", False)),
            )
        row = _build_summary_row(sample_dir, meta, out_npz, out_meta)
        return {"ok": True, "row": row, "error": None}
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
    p = argparse.ArgumentParser(
        description="Sync Nymeria head/lwrist/rwrist 6 IMU streams to TIME_CODE+CLOSEST and resample to 60Hz."
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--nymeria-pkg-root", type=Path, default=DEFAULT_NYMERIA_PKG_ROOT)
    p.add_argument("--target-hz", type=int, default=60)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--save-csv-copy", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes. Use 1 for easier debugging.",
    )
    p.add_argument(
        "--show-inner-progress",
        action="store_true",
        help="Show per-sample grid tqdm (recommended only with --workers 1).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir
    nymeria_pkg_root = args.nymeria_pkg_root

    sample_meta_paths = sorted(base_dir.rglob("metadata.json"))
    if args.limit_samples is not None:
        sample_meta_paths = sample_meta_paths[: int(args.limit_samples)]

    print(
        f"Samples to process: {len(sample_meta_paths)} | target_hz={args.target_hz} | "
        f"overwrite={args.overwrite} | workers={args.workers}"
    )
    if not sample_meta_paths:
        print("No metadata.json files found.")
        return 0

    tasks = [
        {
            "sample_dir": str(meta_path.parent),
            "target_hz": int(args.target_hz),
            "overwrite": bool(args.overwrite),
            "save_csv_copy": bool(args.save_csv_copy),
            "nymeria_pkg_root": str(nymeria_pkg_root),
            "show_inner_progress": bool(args.show_inner_progress and args.workers == 1),
        }
        for meta_path in sample_meta_paths
    ]

    summary_rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    error_details: list[dict[str, Any]] = []

    if args.workers == 1:
        iter_tasks = tqdm(tasks, desc="Nymeria 6IMU sync@60Hz", unit="sample")
        for task in iter_tasks:
            result = _process_one_sample(task)
            if result["ok"]:
                summary_rows.append(result["row"])
            else:
                err = result["error"]
                errors.append((err["sample_dir"], err["message"]))
                error_details.append(err)
            iter_tasks.set_postfix(success=len(summary_rows), fail=len(errors))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(_process_one_sample, task) for task in tasks]
            pbar = tqdm(total=len(futures), desc="Nymeria 6IMU sync@60Hz", unit="sample")
            for fut in as_completed(futures):
                result = fut.result()
                if result["ok"]:
                    summary_rows.append(result["row"])
                else:
                    err = result["error"]
                    errors.append((err["sample_dir"], err["message"]))
                    error_details.append(err)
                pbar.update(1)
                pbar.set_postfix(success=len(summary_rows), fail=len(errors))
            pbar.close()

    nymeria_sync60_summary_df = pd.DataFrame(summary_rows)
    if not nymeria_sync60_summary_df.empty:
        nymeria_sync60_summary_df = nymeria_sync60_summary_df.sort_values(
            ["sample_dir", "sample_path"]
        ).reset_index(drop=True)

    print("\nProcessing complete")
    print("Successful samples:", len(nymeria_sync60_summary_df))
    print("Failed samples:", len(errors))
    if errors:
        print("First 10 failed samples:")
        for x in errors[:10]:
            print("  ", x)

    avg_df = pd.DataFrame(
        [
            {
                "metric": "avg raw head len (1202-1+1202-2)",
                "value": nymeria_sync60_summary_df["head_raw_len_sum2"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
            {
                "metric": "avg raw lwrist len (1202-1+1202-2)",
                "value": nymeria_sync60_summary_df["lwrist_raw_len_sum2"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
            {
                "metric": "avg raw rwrist len (1202-1+1202-2)",
                "value": nymeria_sync60_summary_df["rwrist_raw_len_sum2"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
            {
                "metric": "avg synced len @ 60Hz (common grid)",
                "value": nymeria_sync60_summary_df["synced_seq_len_60hz"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
            {
                "metric": "avg sync overlap sec",
                "value": nymeria_sync60_summary_df["sync_overlap_sec"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
            {
                "metric": "ratio raw 3pos equal (sum2)",
                "value": nymeria_sync60_summary_df["raw_3pos_equal_sum2"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
            {
                "metric": "ratio synced 6 streams equal",
                "value": nymeria_sync60_summary_df["synced_6streams_equal"].mean()
                if not nymeria_sync60_summary_df.empty
                else np.nan,
            },
        ]
    )

    summary_csv = base_dir / "nymeria_synced_6imu_60hz_per_sample_summary.csv"
    errors_json = base_dir / "nymeria_synced_6imu_60hz_batch_errors.json"
    errors_detail_json = base_dir / "nymeria_synced_6imu_60hz_batch_errors_detailed.json"
    nymeria_sync60_summary_df.to_csv(summary_csv, index=False)
    errors_json.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_detail_json.write_text(
        json.dumps(error_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSaved summary:")
    print("  ", summary_csv)
    print("  ", errors_json)
    print("  ", errors_detail_json)
    print("\nAverage statistics:")
    print(avg_df.to_string(index=False))
    if not nymeria_sync60_summary_df.empty:
        print("\nsummary head:")
        print(nymeria_sync60_summary_df.head().to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
