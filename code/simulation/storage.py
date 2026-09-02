"""Convert generated body-surface IMU arrays to time-chunked Zarr storage."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

import config
import data


def parse_args() -> argparse.Namespace:
    cfg = config.TrainSTGCNConfig()
    parser = argparse.ArgumentParser(
        description="Convert generated body-surface IMU x.npy arrays into time-chunked Zarr arrays.",
    )
    parser.add_argument("--summary-csv", type=Path, default=cfg.summary_csv)
    parser.add_argument("--base-dir", type=Path, default=cfg.base_dir)
    parser.add_argument("--candidate-name", type=str, default=cfg.candidate_name)
    parser.add_argument("--chunk-size", type=int, default=cfg.window_size)
    parser.add_argument("--compressor", choices=["zstd", "lz4", "none"], default="zstd")
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument("--sample-dir", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--verify-slices", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_bundle_dir(synth_npz: str) -> Path:
    synth_path = Path(synth_npz)
    if synth_path.is_dir():
        return synth_path
    bundle_dir = synth_path.with_suffix("")
    if bundle_dir.is_dir():
        return bundle_dir
    raise FileNotFoundError(f"Could not resolve bundle directory from {synth_path}")


def build_compressor(name: str, clevel: int):
    if name == "none":
        return None
    return Blosc(cname=name, clevel=clevel, shuffle=Blosc.BITSHUFFLE)


def convert_x_to_zarr(
    x_npy_path: Path,
    chunk_size: int,
    compressor_name: str,
    clevel: int,
    force: bool,
    verify: bool,
    verify_slices: int,
    dry_run: bool,
) -> Path:
    if not x_npy_path.exists():
        raise FileNotFoundError(f"Missing x.npy: {x_npy_path}")

    x_zarr_path = x_npy_path.with_suffix(".zarr")
    tmp_path = x_npy_path.with_suffix(".zarr.tmp")
    if x_zarr_path.exists():
        if not force:
            return x_zarr_path
        shutil.rmtree(x_zarr_path)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    src = np.load(x_npy_path, mmap_mode="r", allow_pickle=True)
    if src.ndim != 3:
        raise ValueError(f"Expected x.npy to be 3D, got shape {src.shape} at {x_npy_path}")
    chunks = (int(src.shape[0]), int(chunk_size), int(src.shape[2]))
    compressor = build_compressor(compressor_name, clevel)

    if dry_run:
        print(
            f"[dry-run] {x_npy_path} -> {x_zarr_path} "
            f"shape={src.shape} dtype={src.dtype} chunks={chunks} compressor={compressor_name}"
        )
        return x_zarr_path

    z = zarr.open(
        str(tmp_path),
        mode="w",
        shape=src.shape,
        chunks=chunks,
        dtype=src.dtype,
        compressor=compressor,
    )
    z.attrs["source_path"] = str(x_npy_path)
    z.attrs["source_shape"] = tuple(int(x) for x in src.shape)
    z.attrs["chunk_size"] = int(chunk_size)

    for start in range(0, int(src.shape[1]), int(chunk_size)):
        end = min(start + int(chunk_size), int(src.shape[1]))
        z[:, start:end, :] = np.asarray(src[:, start:end, :], dtype=src.dtype)

    if verify:
        validate_zarr_conversion(src, z, verify_slices)

    tmp_path.rename(x_zarr_path)
    return x_zarr_path


def validate_zarr_conversion(src: np.ndarray, z: zarr.Array, verify_slices: int) -> None:
    if tuple(z.shape) != tuple(src.shape):
        raise ValueError(f"Shape mismatch: src={src.shape} zarr={z.shape}")
    if np.dtype(z.dtype) != np.dtype(src.dtype):
        raise ValueError(f"Dtype mismatch: src={src.dtype} zarr={z.dtype}")
    if verify_slices <= 0:
        return

    time_len = int(src.shape[1])
    probe_starts = sorted({0, max(0, time_len // 2 - 1), max(0, time_len - 1)})
    for start in probe_starts[:verify_slices]:
        end = min(start + 1, time_len)
        if not np.array_equal(np.asarray(src[:, start:end, :]), np.asarray(z[:, start:end, :])):
            raise ValueError(f"Verification mismatch at slice [{start}:{end}]")


def iter_target_records(args: argparse.Namespace) -> list[data.SampleRecord]:
    records = data.load_summary_records(args.summary_csv, args.candidate_name, args.base_dir)
    if args.sample_dir:
        wanted = set(args.sample_dir)
        records = [record for record in records if record.sample_dir in wanted]
    if args.limit is not None:
        records = records[: args.limit]
    return records


def main() -> int:
    args = parse_args()
    records = iter_target_records(args)
    if not records:
        print("No matching records found.")
        return 0

    converted = 0
    skipped = 0
    for record in records:
        bundle_dir = resolve_bundle_dir(record.synth_npz)
        x_npy_path = bundle_dir / "x.npy"
        x_zarr_path = x_npy_path.with_suffix(".zarr")
        if x_zarr_path.exists() and not args.force:
            print(f"[skip] {record.sample_dir}: {x_zarr_path} already exists")
            skipped += 1
            continue
        try:
            out_path = convert_x_to_zarr(
                x_npy_path=x_npy_path,
                chunk_size=args.chunk_size,
                compressor_name=args.compressor,
                clevel=args.clevel,
                force=args.force,
                verify=args.verify,
                verify_slices=args.verify_slices,
                dry_run=args.dry_run,
            )
            converted += 1
            print(f"[ok] {record.sample_dir}: {x_npy_path} -> {out_path}")
        except Exception as exc:
            print(f"[error] {record.sample_dir}: {exc}")
            return 1

    print(f"Finished. converted={converted} skipped={skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
