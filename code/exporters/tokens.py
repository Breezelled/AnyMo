from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import data
import imu_token_utils
import model
import train_pqvae
import training_utils as utils


DEFAULT_TOKENIZER_EXPORT = config.TOKENIZER_OUTPUT_DIR / "tokenizer_export.pt"
DEFAULT_OUTPUT_DIR = config.MOTION_LANGUAGE_DATA_DIR
DEFAULT_TEXT_ALIGNED_OUTPUT_DIR = config.MOTION_LANGUAGE_DATA_DIR
DEFAULT_FILTERED_ACTIVITY_LABEL_FILENAME = "atomic_action_60hz_activity_labels_filtered.csv"


@dataclass(frozen=True)
class ExportWindowSpec:
    sample_dir: str
    clip_start: int
    clip_end: int
    script: str
    label_name: str
    order_index: int
    source_csv_path: str | None = None
    source_row_index: int | None = None

    @property
    def window_size(self) -> int:
        return int(self.clip_end) - int(self.clip_start) + 1


def build_argparser() -> argparse.ArgumentParser:
    cfg = config.TrainSTGCNConfig()
    parser = argparse.ArgumentParser(
        description="Export the motion-language pretraining corpus from the frozen encoder and tokenizer."
    )
    parser.add_argument("--summary-csv", type=Path, default=cfg.summary_csv)
    parser.add_argument("--base-dir", type=Path, default=cfg.base_dir)
    parser.add_argument("--candidate-name", type=str, default=cfg.candidate_name)
    parser.add_argument("--real1-device-suffix", type=str, default=cfg.real1_device_suffix)
    parser.add_argument("--real2-device-suffix", type=str, default=cfg.real2_device_suffix)
    parser.add_argument("--window-size", type=int, default=cfg.window_size)
    parser.add_argument("--encoder-ckpt", type=Path, default=config.DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--tokenizer-export", type=Path, default=DEFAULT_TOKENIZER_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-cache", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--alignment-mode", type=str, choices=("fixed300", "text_labels"), default="text_labels")
    parser.add_argument("--activity-label-filename", type=str, default=DEFAULT_FILTERED_ACTIVITY_LABEL_FILENAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-visible-nodes", type=int, default=5)
    parser.add_argument("--preload-all-views", dest="preload_all_views", action="store_true")
    parser.add_argument("--no-preload-all-views", dest="preload_all_views", action="store_false")
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--surface-rotation-augment", dest="surface_rotation_augment", action="store_true")
    parser.add_argument("--no-surface-rotation-augment", dest="surface_rotation_augment", action="store_false")
    parser.add_argument("--surface-rotation-inplane-max-deg", type=float, default=cfg.surface_rotation_inplane_max_deg)
    parser.add_argument("--surface-rotation-tilt-max-deg", type=float, default=cfg.surface_rotation_tilt_max_deg)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(preload_all_views=True, surface_rotation_augment=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def load_frozen_pqvae(tokenizer_export: Path, device: torch.device) -> model.MotionPQVAE:
    obj = torch.load(tokenizer_export, map_location="cpu")
    cfg = obj["config"]
    pqvae = model.MotionPQVAE(
        input_dim=int(cfg["input_dim"]),
        bottleneck_dim=int(cfg["bottleneck_dim"]),
        num_codebooks=int(cfg["num_codebooks"]),
        codebook_size=int(cfg["codebook_size"]),
        codebook_dim=int(cfg["codebook_dim"]),
        commitment_weight=float(cfg["commitment_weight"]),
        dead_code_threshold_ratio=float(cfg["dead_code_threshold_ratio"]),
    )
    pqvae.load_state_dict(utils.normalize_compiled_state_dict_keys(obj["pqvae_state_dict"]))
    pqvae.to(device)
    pqvae.eval()
    for parameter in pqvae.parameters():
        parameter.requires_grad_(False)
    return pqvae


def write_codebook_lookup_artifact(tokenizer_export: Path, artifact_path: Path) -> None:
    obj = torch.load(tokenizer_export, map_location="cpu")
    payload = imu_token_utils.build_codebook_lookup_payload(obj)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, artifact_path)


def build_pretraining_record(
    sample_dir: str,
    clip_start: int,
    script: str,
    imu_token_ids: list[int],
    clip_end: int | None = None,
    source_csv_path: str | None = None,
    source_row_index: int | None = None,
) -> dict[str, Any]:
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": imu_token_utils.local_ids_to_token_text(imu_token_ids, include_bos_eos=True),
            }
        ],
        "imu_token_ids": [int(token_id) for token_id in imu_token_ids],
        "sample_dir": str(sample_dir),
        "clip_start": int(clip_start),
        "script": str(script),
    }
    if clip_end is not None:
        row["clip_end"] = int(clip_end)
    if source_csv_path is not None:
        row["source_csv_path"] = str(source_csv_path)
    if source_row_index is not None:
        row["source_row_index"] = int(source_row_index)
    return row


def write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    if args.alignment_mode == "text_labels":
        return DEFAULT_TEXT_ALIGNED_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def resolve_split_cache(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.split_cache is not None:
        return Path(args.split_cache)
    if args.alignment_mode == "text_labels":
        return None
    return output_dir / "sample_train_val_split.json"


def _load_npz_time_axis(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as npz:
        if "t_ns_global_timecode" in npz:
            return np.asarray(npz["t_ns_global_timecode"], dtype=np.int64)
        if "t_ns" in npz:
            return np.asarray(npz["t_ns"], dtype=np.int64)
        raise KeyError(f"Expected 't_ns_global_timecode' or 't_ns' in {npz_path}")


def _load_synth_time_axis(synth_npz_path: Path) -> np.ndarray:
    bundle_dir = synth_npz_path.with_suffix("")
    if bundle_dir.is_dir():
        time_axis_path = bundle_dir / "t_ns_global_timecode.npy"
        if time_axis_path.exists():
            return np.asarray(np.load(time_axis_path, mmap_mode="r"), dtype=np.int64)
    return _load_npz_time_axis(synth_npz_path)


def remap_text_indices_to_synth_time_axis(
    clip_start: int,
    clip_end: int,
    raw_time_axis: np.ndarray,
    synth_time_axis: np.ndarray,
) -> tuple[int, int]:
    if clip_start < 0 or clip_end < clip_start:
        raise ValueError(f"Invalid text-aligned range: start={clip_start}, end={clip_end}")
    if clip_end >= int(raw_time_axis.shape[0]):
        raise IndexError(
            f"Text-aligned end index {clip_end} exceeds raw time axis length {int(raw_time_axis.shape[0])}"
        )
    start_t_ns = int(raw_time_axis[clip_start])
    end_t_ns = int(raw_time_axis[clip_end])
    synth_start = int(np.searchsorted(synth_time_axis, start_t_ns, side="left"))
    synth_end_exclusive = int(np.searchsorted(synth_time_axis, end_t_ns, side="right"))
    if synth_start >= synth_end_exclusive:
        raise ValueError(
            "No valid synthetic frames overlap the requested text span: "
            f"start_idx={clip_start}, end_idx={clip_end}, start_t_ns={start_t_ns}, end_t_ns={end_t_ns}"
        )
    return synth_start, synth_end_exclusive - 1


def build_text_label_window_specs(
    records: list[data.SampleRecord],
    activity_label_filename: str = DEFAULT_FILTERED_ACTIVITY_LABEL_FILENAME,
) -> list[ExportWindowSpec]:
    specs: list[ExportWindowSpec] = []
    order_index = 0
    for record in records:
        csv_path = Path(record.sample_path) / "multimodal_sync_60hz" / activity_label_filename
        raw_time_axis = _load_npz_time_axis(Path(record.real_npz))
        synth_time_axis = _load_synth_time_axis(Path(record.synth_npz))
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader):
                clip_start = int(row["start_idx"])
                clip_end = int(row["end_idx"])
                synth_clip_start, synth_clip_end = remap_text_indices_to_synth_time_axis(
                    clip_start=clip_start,
                    clip_end=clip_end,
                    raw_time_axis=raw_time_axis,
                    synth_time_axis=synth_time_axis,
                )
                script = str(row.get("Describe my atomic actions") or record.label_name)
                specs.append(
                    ExportWindowSpec(
                        sample_dir=record.sample_dir,
                        clip_start=synth_clip_start,
                        clip_end=synth_clip_end,
                        script=script,
                        label_name=record.label_name,
                        order_index=order_index,
                        source_csv_path=str(csv_path),
                        source_row_index=row_index,
                    )
                )
                order_index += 1
    return specs


def build_fixed_window_specs(records: list[data.SampleRecord], args: argparse.Namespace, order_start: int = 0) -> list[ExportWindowSpec]:
    specs: list[ExportWindowSpec] = []
    clip_keys = data.build_all_clip_index(
        records=records,
        window_size=int(args.window_size),
        real1_device_suffix=args.real1_device_suffix,
        real2_device_suffix=args.real2_device_suffix,
    )
    for offset, clip_key in enumerate(clip_keys):
        specs.append(
            ExportWindowSpec(
                sample_dir=clip_key.sample_dir,
                clip_start=clip_key.clip_start,
                clip_end=clip_key.clip_start + int(args.window_size) - 1,
                script=clip_key.label_name,
                label_name=clip_key.label_name,
                order_index=order_start + offset,
            )
        )
    return specs


def plan_export_splits(
    records: list[data.SampleRecord],
    args: argparse.Namespace,
) -> tuple[list[ExportWindowSpec], list[ExportWindowSpec]]:
    if args.alignment_mode == "text_labels":
        return build_text_label_window_specs(records, args.activity_label_filename), []

    output_dir = resolve_output_dir(args)
    split_cache = resolve_split_cache(args, output_dir)
    assert split_cache is not None
    train_records, val_records = imu_token_utils.get_or_build_sample_level_train_val_split(
        records=records,
        cache_path=split_cache,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    train_specs = build_fixed_window_specs(train_records, args, order_start=0)
    val_specs = build_fixed_window_specs(val_records, args, order_start=len(train_specs))
    return train_specs, val_specs


def _build_export_loader_for_specs(
    args: argparse.Namespace,
    record_by_sample: dict[str, data.SampleRecord],
    synth_store: data.DominantTop2Store,
    specs: list[ExportWindowSpec],
    window_size: int,
) -> tuple[data.MaskedMotionTokenizerDataset, DataLoader]:
    clip_keys = [data.ClipKey(sample_dir=spec.sample_dir, clip_start=spec.clip_start, label_name=spec.label_name) for spec in specs]
    dataset = data.MaskedMotionTokenizerDataset(
        clip_keys=clip_keys,
        record_by_sample=record_by_sample,
        synth_store=synth_store,
        window_size=window_size,
        seed=args.seed,
        surface_rotation_augment=args.surface_rotation_augment,
        inplane_max_deg=args.surface_rotation_inplane_max_deg,
        tilt_max_deg=args.surface_rotation_tilt_max_deg,
        preload_all_views=args.preload_all_views,
        preload_workers=args.preload_workers,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(args.num_workers), pin_memory=True)
    return dataset, loader


def export_window_specs(
    split_name: str,
    specs: list[ExportWindowSpec],
    args: argparse.Namespace,
    record_by_sample: dict[str, data.SampleRecord],
    encoder: model.STGCNEncoder,
    pqvae: model.MotionPQVAE,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    synth_store = data.DominantTop2Store()
    grouped_specs: dict[int, list[ExportWindowSpec]] = defaultdict(list)
    for spec in specs:
        grouped_specs[spec.window_size].append(spec)

    rows_with_order: list[tuple[int, dict[str, Any]]] = []
    for window_size, window_specs in tqdm(sorted(grouped_specs.items()), desc=f"export {split_name}", leave=False):
        dataset, loader = _build_export_loader_for_specs(args, record_by_sample, synth_store, window_specs, window_size)
        clip_cursor = 0
        for batch in loader:
            xb = batch["view_full"].to(device)
            batch_specs = window_specs[clip_cursor : clip_cursor + xb.size(0)]
            clip_cursor += xb.size(0)
            batch_clip_keys = dataset.clip_keys[clip_cursor - xb.size(0) : clip_cursor]
            visible_mask = torch.stack(
                [
                    imu_token_utils.build_deterministic_visible_mask(
                        clip_key=clip_key,
                        seed=args.seed,
                        num_nodes=23,
                        max_visible_nodes=args.max_visible_nodes,
                    )
                    for clip_key in batch_clip_keys
                ],
                dim=0,
            ).to(device)
            with torch.no_grad():
                latents = encoder(xb, visible_node_mask=visible_mask)["global_seq_latent"]
                codes = pqvae.encode_codes(latents)
                imu_token_ids = imu_token_utils.codes_to_local_imu_ids(codes, codebook_size=pqvae.codebook_size).cpu()
            for spec, token_row in zip(batch_specs, imu_token_ids.tolist()):
                rows_with_order.append(
                    (
                        spec.order_index,
                        build_pretraining_record(
                            sample_dir=spec.sample_dir,
                            clip_start=spec.clip_start,
                            script=spec.script,
                            imu_token_ids=token_row,
                            clip_end=spec.clip_end,
                            source_csv_path=spec.source_csv_path,
                            source_row_index=spec.source_row_index,
                        ),
                    )
                )
    rows_with_order.sort(key=lambda pair: pair[0])
    return [row for _, row in rows_with_order]


def export_split_rows(
    split_name: str,
    specs: list[ExportWindowSpec],
    args: argparse.Namespace,
    record_by_sample: dict[str, data.SampleRecord],
    encoder: model.STGCNEncoder,
    pqvae: model.MotionPQVAE,
    device: torch.device,
) -> list[dict[str, Any]]:
    return export_window_specs(
        split_name=split_name,
        specs=specs,
        args=args,
        record_by_sample=record_by_sample,
        encoder=encoder,
        pqvae=pqvae,
        device=device,
    )


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    utils.set_seed(args.seed)
    records = data.exclude_held_out_subjects(
        data.load_summary_records(args.summary_csv, args.candidate_name, args.base_dir)
    )
    encoder = train_pqvae.load_frozen_encoder(args.encoder_ckpt, device=device, dropout=args.dropout)
    pqvae = load_frozen_pqvae(args.tokenizer_export, device=device)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_codebook_lookup_artifact(args.tokenizer_export, output_dir / "imu_codebook_lookup.pt")
    train_specs, val_specs = plan_export_splits(records, args)
    record_by_sample = {record.sample_dir: record for record in records}

    train_rows = export_split_rows("train", train_specs, args, record_by_sample, encoder, pqvae, device)
    val_rows = export_split_rows("val", val_specs, args, record_by_sample, encoder, pqvae, device)
    write_jsonl_records(output_dir / "train.jsonl", train_rows)
    val_path = output_dir / "val.jsonl"
    if val_rows:
        write_jsonl_records(val_path, val_rows)
    elif val_path.exists():
        val_path.unlink()

    summary = {
        "alignment_mode": str(args.alignment_mode),
        "train_samples": len({spec.sample_dir for spec in train_specs}),
        "val_samples": len({spec.sample_dir for spec in val_specs}),
        "train_windows": len(train_rows),
        "val_windows": len(val_rows),
        "tokenizer_export": str(args.tokenizer_export),
        "encoder_ckpt": str(args.encoder_ckpt),
        "train_ratio": float(args.train_ratio),
        "activity_label_filename": str(args.activity_label_filename),
    }
    utils.save_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    export_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
