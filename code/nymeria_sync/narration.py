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

NARRATION_SPECS = {
    "activity_summarization": {
        "input_name": "activity_summarization.csv",
        "output_name": "activity_summarization_60hz.csv",
        "required_text_cols": ["Describe my activity"],
    },
    "motion_narration": {
        "input_name": "motion_narration.csv",
        "output_name": "motion_narration_60hz.csv",
        "required_text_cols": [
            "Describe my body posture",
            "Describe my hands/arms motion",
            "Describe my legs/feet motion",
            "Describe my focus attention",
        ],
    },
}


def _ensure_imports(nymeria_pkg_root: str) -> None:
    pkg_root = str(Path(nymeria_pkg_root))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


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


def _sync_narration_csv_to_grid(
    sample_dir: Path,
    grid_ns: np.ndarray,
    nymeria_pkg_root: str,
    *,
    input_name: str,
    required_text_cols: list[str],
    meta_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = sample_dir / "narration" / input_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")

    df = pd.read_csv(csv_path)
    if "start_time" not in df.columns or "end_time" not in df.columns:
        raise RuntimeError(f"{input_name} missing start_time/end_time")
    for col in required_text_cols:
        if col not in df.columns:
            raise RuntimeError(f"{input_name} missing required column {col!r}")

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
        start_t_ns_global = _device_time_ns_to_global_time_ns(
            rec.vrs_dp, int(round(start_time * 1e9))
        )
        end_t_ns_global = _device_time_ns_to_global_time_ns(
            rec.vrs_dp, int(round(end_time * 1e9))
        )

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

        out_row = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_t_ns_global": int(start_t_ns_global),
            "end_t_ns_global": int(end_t_ns_global),
            "start_time": start_time,
            "end_time": end_time,
        }
        for col in df.columns:
            if col in out_row:
                continue
            out_row[col] = row[col]
        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows)
    meta = {
        f"{meta_prefix}_num_rows_raw": int(len(df)),
        f"{meta_prefix}_num_rows_kept": int(len(out_df)),
        f"{meta_prefix}_num_rows_out_of_range": int(out_of_range),
        f"{meta_prefix}_num_rows_empty_after_mapping": int(empty_after_mapping),
    }
    return out_df, meta


def sync_additional_narrations_for_sample(
    sample_dir: str | Path,
    *,
    nymeria_pkg_root: str | Path = DEFAULT_NYMERIA_PKG_ROOT,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    out_dir = sample_dir / "multimodal_sync_60hz"
    imu_npz_path = out_dir / "sync_imu_60hz.npz"
    meta_out = out_dir / "sync_meta.json"

    if not imu_npz_path.is_file():
        raise FileNotFoundError(f"missing {imu_npz_path}")

    with np.load(imu_npz_path, allow_pickle=True) as imu_npz:
        grid_ns = np.asarray(imu_npz["t_ns_global_timecode"], dtype=np.int64)
    if grid_ns.ndim != 1 or grid_ns.size == 0:
        raise RuntimeError(f"invalid sync_imu_60hz grid shape: {grid_ns.shape}")

    existing_meta: dict[str, Any] = {}
    if meta_out.is_file():
        existing_meta = json.loads(meta_out.read_text(encoding="utf-8"))

    sample_meta: dict[str, Any] = dict(existing_meta)
    sample_meta.setdefault("sample_dir", sample_dir.name)
    sample_meta.setdefault("target_hz", 60)
    sample_meta.setdefault("status", "ok")

    output_paths: dict[str, str] = {}
    processed_any = False
    for name, spec in NARRATION_SPECS.items():
        input_path = sample_dir / "narration" / spec["input_name"]
        has_file = input_path.is_file()
        sample_meta[f"has_{name}"] = bool(has_file)
        if not has_file:
            continue

        out_df, narr_meta = _sync_narration_csv_to_grid(
            sample_dir=sample_dir,
            grid_ns=grid_ns,
            nymeria_pkg_root=str(nymeria_pkg_root),
            input_name=spec["input_name"],
            required_text_cols=list(spec["required_text_cols"]),
            meta_prefix=name,
        )
        out_path = out_dir / spec["output_name"]
        out_df.to_csv(out_path, index=False)
        output_paths[f"saved_{name}_csv"] = str(out_path)
        sample_meta.update(narr_meta)
        processed_any = True

    if not processed_any:
        raise RuntimeError("no additional narration files found for this sample")

    sample_meta.update(output_paths)
    meta_out.write_text(json.dumps(sample_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample_meta


def _build_summary_row(sample_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_dir": sample_dir.name,
        "sample_path": str(sample_dir),
        "has_activity_summarization": bool(meta.get("has_activity_summarization", False)),
        "has_motion_narration": bool(meta.get("has_motion_narration", False)),
        "activity_summarization_num_rows_kept": meta.get(
            "activity_summarization_num_rows_kept", np.nan
        ),
        "motion_narration_num_rows_kept": meta.get(
            "motion_narration_num_rows_kept", np.nan
        ),
        "status": meta.get("status", "unknown"),
        "saved_activity_summarization_csv": meta.get("saved_activity_summarization_csv", ""),
        "saved_motion_narration_csv": meta.get("saved_motion_narration_csv", ""),
    }


def _process_one_sample(task: dict[str, Any]) -> dict[str, Any]:
    sample_dir = Path(task["sample_dir"])
    out_dir = sample_dir / "multimodal_sync_60hz"
    meta_out = out_dir / "sync_meta.json"
    expected_outputs = []
    if task["has_activity_summarization"]:
        expected_outputs.append(out_dir / NARRATION_SPECS["activity_summarization"]["output_name"])
    if task["has_motion_narration"]:
        expected_outputs.append(out_dir / NARRATION_SPECS["motion_narration"]["output_name"])
    try:
        if (not task["overwrite"]) and meta_out.is_file() and all(p.is_file() for p in expected_outputs):
            meta = json.loads(meta_out.read_text(encoding="utf-8"))
        else:
            meta = sync_additional_narrations_for_sample(
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
    p = argparse.ArgumentParser(
        description="Sync Nymeria activity_summarization and motion_narration to the existing multimodal 60Hz grid"
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--nymeria-pkg-root", type=Path, default=DEFAULT_NYMERIA_PKG_ROOT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir

    sample_infos: list[dict[str, Any]] = []
    for imu_npz in sorted(base_dir.rglob("multimodal_sync_60hz/sync_imu_60hz.npz")):
        sample_dir = imu_npz.parent.parent
        act = (sample_dir / "narration" / NARRATION_SPECS["activity_summarization"]["input_name"]).is_file()
        mot = (sample_dir / "narration" / NARRATION_SPECS["motion_narration"]["input_name"]).is_file()
        if act or mot:
            sample_infos.append(
                {
                    "sample_dir": sample_dir,
                    "has_activity_summarization": act,
                    "has_motion_narration": mot,
                }
            )

    if args.limit_samples is not None:
        sample_infos = sample_infos[: int(args.limit_samples)]

    print(
        f"Samples to process: {len(sample_infos)} | overwrite={args.overwrite} | workers={args.workers}"
    )
    if not sample_infos:
        print("No eligible samples found.")
        return 0

    tasks = [
        {
            "sample_dir": str(info["sample_dir"]),
            "has_activity_summarization": bool(info["has_activity_summarization"]),
            "has_motion_narration": bool(info["has_motion_narration"]),
            "overwrite": bool(args.overwrite),
            "nymeria_pkg_root": str(args.nymeria_pkg_root),
        }
        for info in sample_infos
    ]

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if args.workers == 1:
        pbar = tqdm(tasks, desc="Nymeria extra narrations sync@60Hz", unit="sample")
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
            pbar = tqdm(total=len(futures), desc="Nymeria extra narrations sync@60Hz", unit="sample")
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

    summary_parent = base_dir
    if (base_dir / "multimodal_sync_60hz").is_dir():
        summary_parent = base_dir / "multimodal_sync_60hz"

    summary_csv = summary_parent / "nymeria_additional_narrations_60hz_summary.csv"
    errors_json = summary_parent / "nymeria_additional_narrations_60hz_errors.json"
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
