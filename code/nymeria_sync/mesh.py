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

BODY_SUBDIR = "body"
SYNC_SUBDIR = "multimodal_sync_60hz"
XDATA_FILENAME = "xdata.npz"
GLB_FILENAME = "xdata_blueman.glb"
XSENS_60HZ_FILENAME = "xsens_60hz.npz"
OUTPUT_FILENAME = "momentum_mesh_60hz.npz"


def _ensure_imports(nymeria_pkg_root: str) -> None:
    pkg_root = str(Path(nymeria_pkg_root))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _required_paths(sample_dir: Path) -> dict[str, Path]:
    body_dir = sample_dir / BODY_SUBDIR
    sync_dir = sample_dir / SYNC_SUBDIR
    return {
        "xdata_npz": body_dir / XDATA_FILENAME,
        "xdata_glb": body_dir / GLB_FILENAME,
        "xsens_60hz": sync_dir / XSENS_60HZ_FILENAME,
        "output": sync_dir / OUTPUT_FILENAME,
    }


def _has_required_inputs(sample_dir: Path) -> bool:
    paths = _required_paths(sample_dir)
    return (
        paths["xdata_npz"].is_file()
        and paths["xdata_glb"].is_file()
        and paths["xsens_60hz"].is_file()
    )


def discover_sample_dirs(base_dir: Path, start: int = 0, end: int | None = None) -> list[Path]:
    sample_dirs: list[Path] = []
    for xsens_path in sorted(base_dir.rglob(f"{SYNC_SUBDIR}/{XSENS_60HZ_FILENAME}")):
        sample_dir = xsens_path.parent.parent
        if _has_required_inputs(sample_dir):
            sample_dirs.append(sample_dir)
    if start < 0:
        raise ValueError("start must be >= 0")
    if end is not None and end < start:
        raise ValueError("end must be >= start")
    return sample_dirs[start:end]


def _load_synced_xsens_indices(xsens_60hz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(xsens_60hz_path, allow_pickle=False) as data:
        grid_ns = np.asarray(data["t_ns_global_timecode"], dtype=np.int64)
        src_idx = np.asarray(data["xsens_src_idx_for_60hz"], dtype=np.int64)
        if "timestamps_us_src" in data:
            timestamps_us_src = np.asarray(data["timestamps_us_src"], dtype=np.int64)
        else:
            timestamps_us_src = np.empty_like(src_idx, dtype=np.int64)
    if grid_ns.ndim != 1 or src_idx.ndim != 1:
        raise RuntimeError("xsens_60hz arrays must be 1D")
    if grid_ns.shape[0] != src_idx.shape[0]:
        raise RuntimeError("60Hz time grid and source indices length mismatch")
    return grid_ns, src_idx, timestamps_us_src


def _load_body_provider(sample_dir: Path, nymeria_pkg_root: str):
    _ensure_imports(nymeria_pkg_root)
    from nymeria.body_motion_provider import BodyDataProvider

    paths = _required_paths(sample_dir)
    body_dp = BodyDataProvider(npzfile=str(paths["xdata_npz"]), glbfile=str(paths["xdata_glb"]))
    if body_dp.character is None or body_dp.motion is None or body_dp.momentum_template_mesh is None:
        raise RuntimeError("BodyDataProvider did not load Momentum mesh motion")
    return body_dp


def _load_body_motion_module(nymeria_pkg_root: str):
    _ensure_imports(nymeria_pkg_root)
    import nymeria.body_motion_provider as body_motion_module

    return body_motion_module


def _to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _extract_mesh_attr(mesh: Any, names: list[str], expected_ndim: int | None = None) -> np.ndarray | None:
    for name in names:
        if not hasattr(mesh, name):
            continue
        value = getattr(mesh, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        arr = _to_numpy(value)
        if expected_ndim is None or arr.ndim == expected_ndim:
            return arr
    return None


def _apply_xsens_transform(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    if points.ndim == 2:
        return (transform_matrix @ points.T).T.astype(np.float32, copy=False)
    if points.ndim == 3:
        return np.einsum("ij,fvj->fvi", transform_matrix, points).astype(np.float32, copy=False)
    raise ValueError(f"unsupported points ndim: {points.ndim}")


def _get_template_vertices_and_faces(
    body_dp: Any,
    body_motion_module: Any,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = body_dp.momentum_template_mesh
    transform = _to_numpy(body_dp._A_Wx_Wm, dtype=np.float32).reshape(3, 3)

    template_vertices = _extract_mesh_attr(
        mesh,
        ["vertices", "vertex_positions", "points", "positions"],
        expected_ndim=2,
    )
    template_faces = _extract_mesh_attr(
        mesh,
        ["faces", "triangle_indices", "triangles", "indices"],
        expected_ndim=2,
    )
    if template_faces is None:
        raise RuntimeError("unable to extract template faces from Momentum mesh")
    template_faces = np.asarray(template_faces, dtype=np.int32, order="C")

    if template_vertices is not None:
        template_vertices = _apply_xsens_transform(
            np.asarray(template_vertices, dtype=np.float32, order="C"),
            transform,
        )
    else:
        first_idx = 0
        template_vertices = _generate_posed_vertices(
            body_dp,
            np.array([first_idx], dtype=np.int64),
            body_motion_module,
        )[0]

    return template_vertices.astype(np.float32, copy=False), template_faces


def _generate_posed_vertices(
    body_dp: Any,
    src_idx: np.ndarray,
    body_motion_module: Any,
) -> np.ndarray:
    if src_idx.ndim != 1:
        raise ValueError("src_idx must be 1D")

    transform = _to_numpy(body_dp._A_Wx_Wm, dtype=np.float32).reshape(3, 3)
    posed_vertices: list[np.ndarray] = []

    for idx in src_idx.tolist():
        motion = body_motion_module.torch.tensor(body_dp.motion[int(idx)], dtype=body_motion_module.torch.float32)
        skel_state = body_motion_module.pym.geometry.model_parameters_to_skeleton_state(
            body_dp.character,
            motion,
        )
        skin = body_dp.character.skin_points(skel_state)
        skin_np = _to_numpy(skin, dtype=np.float32)
        posed_vertices.append(_apply_xsens_transform(skin_np, transform))

    return np.stack(posed_vertices, axis=0).astype(np.float32, copy=False)


def build_mesh_payload(
    *,
    grid_ns: np.ndarray,
    src_idx: np.ndarray,
    timestamps_us_src: np.ndarray,
    template_vertices: np.ndarray,
    template_faces: np.ndarray,
    posed_vertices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "t_ns_global_timecode": np.asarray(grid_ns, dtype=np.int64),
        "xsens_src_idx_for_60hz": np.asarray(src_idx, dtype=np.int64),
        "timestamps_us_src": np.asarray(timestamps_us_src, dtype=np.int64),
        "template_vertices": np.asarray(template_vertices, dtype=np.float32),
        "template_faces": np.asarray(template_faces, dtype=np.int32),
        "posed_vertices_60hz": np.asarray(posed_vertices, dtype=np.float32),
    }


def _write_payload(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def process_one_sample(task: dict[str, Any]) -> dict[str, Any]:
    sample_dir = Path(task["sample_dir"])
    force = bool(task["force"])
    nymeria_pkg_root = str(task["nymeria_pkg_root"])
    paths = _required_paths(sample_dir)
    output_path = paths["output"]

    try:
        if output_path.exists() and not force:
            return {
                "ok": True,
                "row": {
                    "sample_dir": str(sample_dir),
                    "status": "skipped_existing",
                    "frames": None,
                    "num_vertices": None,
                    "num_faces": None,
                    "output_path": str(output_path),
                },
            }

        missing = [str(p) for k, p in paths.items() if k != "output" and not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing required inputs: {missing}")

        grid_ns, src_idx, timestamps_us_src = _load_synced_xsens_indices(paths["xsens_60hz"])
        body_dp = _load_body_provider(sample_dir, nymeria_pkg_root)
        body_motion_module = _load_body_motion_module(nymeria_pkg_root)

        if timestamps_us_src.size == 0:
            timestamps_us_all = np.asarray(body_dp.xsens_data["timestamps_us"], dtype=np.int64)
            timestamps_us_src = timestamps_us_all[src_idx]

        template_vertices, template_faces = _get_template_vertices_and_faces(body_dp, body_motion_module)
        posed_vertices = _generate_posed_vertices(body_dp, src_idx, body_motion_module)

        payload = build_mesh_payload(
            grid_ns=grid_ns,
            src_idx=src_idx,
            timestamps_us_src=timestamps_us_src,
            template_vertices=template_vertices,
            template_faces=template_faces,
            posed_vertices=posed_vertices,
        )
        _write_payload(output_path, payload)

        return {
            "ok": True,
            "row": {
                "sample_dir": str(sample_dir),
                "status": "written",
                "frames": int(grid_ns.shape[0]),
                "num_vertices": int(template_vertices.shape[0]),
                "num_faces": int(template_faces.shape[0]),
                "output_path": str(output_path),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "sample_dir": str(sample_dir),
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Nymeria Momentum full-body mesh synchronized to 60 Hz.")
    parser.add_argument("--sample-dir", type=Path, default=None, help="Process a single Nymeria sample directory.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Nymeria dataset root for batch mode.")
    parser.add_argument("--nymeria-pkg-root", type=Path, default=DEFAULT_NYMERIA_PKG_ROOT)
    parser.add_argument("--workers", type=int, default=16, help="Process-level parallelism for batch mode.")
    parser.add_argument("--start", type=int, default=0, help="Start index for batch slicing.")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) for batch slicing.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing momentum_mesh_60hz.npz files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.sample_dir is None and args.base_dir is None:
        parser.error("either --sample-dir or --base-dir must be provided")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    tasks: list[dict[str, Any]]
    if args.sample_dir is not None:
        sample_dirs = [args.sample_dir]
    else:
        sample_dirs = discover_sample_dirs(args.base_dir, start=args.start, end=args.end)

    if not sample_dirs:
        print("no eligible sample directories found")
        return 0

    tasks = [
        {
            "sample_dir": str(sample_dir),
            "force": bool(args.force),
            "nymeria_pkg_root": str(args.nymeria_pkg_root),
        }
        for sample_dir in sample_dirs
    ]

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if args.workers == 1 or len(tasks) == 1:
        pbar = tqdm(tasks, desc="Nymeria mesh sync@60Hz", unit="sample")
        for task in pbar:
            result = process_one_sample(task)
            if result["ok"]:
                summary_rows.append(result["row"])
            else:
                errors.append(result["error"])
            pbar.set_postfix(success=len(summary_rows), fail=len(errors))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(process_one_sample, task) for task in tasks]
            pbar = tqdm(total=len(futures), desc="Nymeria mesh sync@60Hz", unit="sample")
            for fut in as_completed(futures):
                result = fut.result()
                if result["ok"]:
                    summary_rows.append(result["row"])
                else:
                    errors.append(result["error"])
                pbar.update(1)
                pbar.set_postfix(success=len(summary_rows), fail=len(errors))
            pbar.close()

    if args.sample_dir is None:
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["sample_dir"]).reset_index(drop=True)

        summary_csv = args.base_dir / "nymeria_mesh_60hz_summary.csv"
        errors_json = args.base_dir / "nymeria_mesh_60hz_errors.json"
        summary_df.to_csv(summary_csv, index=False)
        errors_json.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nProcessing complete")
        print("Successful samples:", len(summary_rows))
        print("Failed samples:", len(errors))
        print("\nSaved summary:")
        print("  ", summary_csv)
        print("  ", errors_json)
    else:
        if summary_rows:
            print(json.dumps(summary_rows[0], ensure_ascii=False, indent=2))
        if errors:
            print(json.dumps(errors[0], ensure_ascii=False, indent=2))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
