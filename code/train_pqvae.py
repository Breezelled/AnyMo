from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import data
import model
import training_utils as utils


DEFAULT_OUTPUT_DIR = config.TOKENIZER_OUTPUT_DIR


def build_argparser() -> argparse.ArgumentParser:
    cfg = config.TrainSTGCNConfig()
    parser = argparse.ArgumentParser(
        description="Train the product-quantized tokenizer on frozen full-body encoder latents."
    )
    parser.add_argument("--summary-csv", type=Path, default=cfg.summary_csv)
    parser.add_argument("--base-dir", type=Path, default=cfg.base_dir)
    parser.add_argument("--candidate-name", type=str, default=cfg.candidate_name)
    parser.add_argument("--real1-device-suffix", type=str, default=cfg.real1_device_suffix)
    parser.add_argument("--real2-device-suffix", type=str, default=cfg.real2_device_suffix)
    parser.add_argument("--window-size", type=int, default=cfg.window_size)
    parser.add_argument("--encoder-ckpt", type=Path, default=config.DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=16)
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
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--num-codebooks", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--codebook-dim", type=int, default=64)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-epsilon", type=float, default=1e-5)
    parser.add_argument("--dead-code-threshold-ratio", type=float, default=0.2)
    parser.set_defaults(surface_rotation_augment=True, preload_all_views=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def load_frozen_encoder(ckpt_path: Path, device: torch.device, dropout: float) -> model.STGCNEncoder:
    encoder = model.STGCNEncoder(in_channels=6, latent_dim=256, dropout=dropout)
    obj = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(utils.normalize_compiled_state_dict_keys(obj["encoder_state_dict"]))
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def resolve_loader_num_workers(requested_num_workers: int, preload_all_views: bool) -> int:
    return 0 if preload_all_views else max(0, int(requested_num_workers))


def build_train_loader(args: argparse.Namespace) -> DataLoader:
    records = data.exclude_held_out_subjects(
        data.load_summary_records(args.summary_csv, args.candidate_name, args.base_dir)
    )
    rng = np.random.default_rng(args.seed)
    record_order = rng.permutation(len(records)).tolist()
    records = [records[i] for i in record_order]
    clip_keys = data.build_all_clip_index(
        records=records,
        window_size=args.window_size,
        real1_device_suffix=args.real1_device_suffix,
        real2_device_suffix=args.real2_device_suffix,
    )
    record_by_sample = {record.sample_dir: record for record in records}
    synth_store = data.DominantTop2Store()
    train_ds = data.MaskedMotionTokenizerDataset(
        clip_keys=clip_keys,
        record_by_sample=record_by_sample,
        synth_store=synth_store,
        window_size=args.window_size,
        seed=args.seed,
        surface_rotation_augment=args.surface_rotation_augment,
        inplane_max_deg=args.surface_rotation_inplane_max_deg,
        tilt_max_deg=args.surface_rotation_tilt_max_deg,
        preload_all_views=args.preload_all_views,
        preload_workers=args.preload_workers,
    )
    effective_num_workers = resolve_loader_num_workers(args.num_workers, args.preload_all_views)
    loader_kwargs: dict[str, Any] = {"pin_memory": True}
    if effective_num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 8})
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        **loader_kwargs,
    )


def compute_codebook_metrics(codes: torch.Tensor, codebook_size: int) -> dict[str, Any]:
    codes_cpu = codes.detach().cpu().long()
    b, l, n = codes_cpu.shape
    codebook_metrics: list[dict[str, Any]] = []
    collapse_risk = False
    for codebook_idx in range(n):
        flat = codes_cpu[:, :, codebook_idx].reshape(-1)
        counts = torch.bincount(flat, minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        used_mask = counts > 0
        used_codes = int(used_mask.sum().item())
        usage_rate = used_codes / float(codebook_size)
        dead_code_ratio = 1.0 - usage_rate
        active_probs = probs[used_mask]
        entropy = float((-(active_probs * active_probs.log()).sum()).item()) if active_probs.numel() > 0 else 0.0
        perplexity = float(active_probs.log().mul(active_probs).sum().neg().exp().item()) if active_probs.numel() > 0 else 0.0
        topk = torch.topk(probs, k=min(10, codebook_size)).values
        top1_mass = float(topk[0].item()) if topk.numel() > 0 else 0.0
        top5_mass = float(topk[: min(5, topk.numel())].sum().item()) if topk.numel() > 0 else 0.0
        top10_mass = float(topk.sum().item()) if topk.numel() > 0 else 0.0
        if usage_rate < 0.1 or dead_code_ratio > 0.9 or perplexity < 10.0 or top1_mass > 0.5:
            collapse_risk = True
        codebook_metrics.append(
            {
                "codebook_index": codebook_idx,
                "used_codes": used_codes,
                "usage_rate": usage_rate,
                "dead_code_ratio": dead_code_ratio,
                "perplexity": perplexity,
                "usage_entropy": entropy,
                "max_cluster_ratio": top1_mass,
                "top1_mass": top1_mass,
                "top5_mass": top5_mass,
                "top10_mass": top10_mass,
            }
        )

    tokens = model.interleave_codes(codes_cpu)
    token_rows = [tuple(row.tolist()) for row in tokens]
    row_counts = Counter(token_rows)
    unique_sequences = len(row_counts)
    num_sequences = len(token_rows)
    exact_sequence_uniqueness = unique_sequences / float(num_sequences) if num_sequences > 0 else 0.0
    exact_sequence_collision_rate = 1.0 - exact_sequence_uniqueness

    pair_rows = [tuple(row.tolist()) for row in codes_cpu.reshape(b * l, n)]
    pair_counts = Counter(pair_rows)
    unique_pairs = len(pair_counts)
    num_pairs = len(pair_rows)
    pair_uniqueness = unique_pairs / float(num_pairs) if num_pairs > 0 else 0.0

    duplicate_histogram: dict[str, int] = {}
    for freq, count in Counter(row_counts.values()).items():
        duplicate_histogram[str(int(freq))] = int(count)

    return {
        "collapse_risk": collapse_risk,
        "codebooks": codebook_metrics,
        "sequence": {
            "num_sequences": num_sequences,
            "unique_sequences": unique_sequences,
            "exact_sequence_uniqueness": exact_sequence_uniqueness,
            "exact_sequence_collision_rate": exact_sequence_collision_rate,
            "per_step_pair_uniqueness": pair_uniqueness,
            "duplicate_histogram": duplicate_histogram,
        },
    }


def save_checkpoint(
    path: Path,
    pqvae: model.MotionPQVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "pqvae_state_dict": utils.export_module_state_dict(pqvae),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def export_tokenizer(path: Path, pqvae: model.MotionPQVAE, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pqvae_state_dict": utils.export_module_state_dict(pqvae),
            "config": {
                "input_dim": args.input_dim,
                "bottleneck_dim": args.bottleneck_dim,
                "num_codebooks": args.num_codebooks,
                "codebook_size": args.codebook_size,
                "codebook_dim": args.codebook_dim,
                "commitment_weight": args.commitment_weight,
                "dead_code_threshold_ratio": args.dead_code_threshold_ratio,
            },
        },
        path,
    )


def format_epoch_metrics(epoch: int, metrics: dict[str, Any]) -> str:
    parts = [
        f"epoch={epoch}",
        f"train_loss={metrics['loss']:.6f}",
        f"recon_loss={metrics['recon_loss']:.6f}",
        f"commitment_loss={metrics['commitment_loss']:.6f}",
        f"dead_code_replacements={metrics['dead_code_replacements']}",
        f"collapse_risk={metrics['collapse_risk']}",
    ]
    sequence_metrics = metrics.get("sequence", {})
    if "exact_sequence_collision_rate" in sequence_metrics:
        parts.append(f"exact_sequence_collision_rate={sequence_metrics['exact_sequence_collision_rate']:.6f}")
    for codebook_metrics in metrics.get("codebooks", []):
        codebook_idx = int(codebook_metrics["codebook_index"])
        if "usage_rate" in codebook_metrics:
            parts.append(f"codebook{codebook_idx}_usage_rate={codebook_metrics['usage_rate']:.6f}")
        if "dead_code_ratio" in codebook_metrics:
            parts.append(f"codebook{codebook_idx}_dead_code_ratio={codebook_metrics['dead_code_ratio']:.6f}")
    return " ".join(parts)


def run_epoch(
    encoder: model.STGCNEncoder,
    pqvae: model.MotionPQVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_visible_nodes: int,
    desc: str,
) -> dict[str, Any]:
    pqvae.train()
    total_loss = 0.0
    total_recon = 0.0
    total_commit = 0.0
    count = 0
    code_batches: list[torch.Tensor] = []
    dead_code_replacements = torch.zeros(pqvae.num_codebooks, dtype=torch.long)

    for batch in tqdm(loader, desc=desc, leave=False):
        xb = batch["view_full"].to(device, non_blocking=True)
        visible = model.sample_visible_node_mask(
            batch_size=xb.size(0),
            num_nodes=xb.size(3),
            device=device,
            max_visible_nodes=max_visible_nodes,
        )
        with torch.no_grad():
            latents = encoder(xb, visible_node_mask=visible)["global_seq_latent"]
        out = pqvae(latents)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        optimizer.step()

        bs = int(xb.size(0))
        total_loss += float(out["loss"].item()) * bs
        total_recon += float(out["recon_loss"].item()) * bs
        total_commit += float(out["commitment_loss"].item()) * bs
        count += bs
        code_batches.append(out["codes"].detach().cpu())
        dead_code_replacements += out["dead_code_replacements"].detach().cpu()

    codes = torch.cat(code_batches, dim=0) if code_batches else torch.zeros(0, 0, 0, dtype=torch.long)
    metrics = compute_codebook_metrics(codes, pqvae.codebook_size) if codes.numel() > 0 else {
        "collapse_risk": False,
        "codebooks": [],
        "sequence": {
            "num_sequences": 0,
            "unique_sequences": 0,
            "exact_sequence_uniqueness": 0.0,
            "exact_sequence_collision_rate": 0.0,
            "per_step_pair_uniqueness": 0.0,
            "duplicate_histogram": {},
        },
    }
    metrics["loss"] = total_loss / max(count, 1)
    metrics["recon_loss"] = total_recon / max(count, 1)
    metrics["commitment_loss"] = total_commit / max(count, 1)
    metrics["dead_code_replacements"] = dead_code_replacements.tolist()
    for codebook_idx, replacement_count in enumerate(metrics["dead_code_replacements"]):
        if codebook_idx < len(metrics["codebooks"]):
            metrics["codebooks"][codebook_idx]["dead_code_replacements"] = int(replacement_count)
    return metrics


def main() -> int:
    args = parse_args()
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_frozen_encoder(args.encoder_ckpt, device=device, dropout=args.dropout)
    pqvae = model.MotionPQVAE(
        input_dim=args.input_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        commitment_weight=args.commitment_weight,
        ema_decay=args.ema_decay,
        ema_epsilon=args.ema_epsilon,
        dead_code_threshold_ratio=args.dead_code_threshold_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(pqvae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_train_loader(args)

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        metrics = run_epoch(
            encoder=encoder,
            pqvae=pqvae,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_visible_nodes=args.max_visible_nodes,
            desc=f"pqvae train {epoch + 1}/{args.epochs}",
        )
        metrics["epoch"] = int(epoch)
        history.append(metrics)
        save_checkpoint(args.output_dir / "pqvae_last.pt", pqvae, optimizer, epoch, best_loss, metrics)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(args.output_dir / "pqvae_best.pt", pqvae, optimizer, epoch, best_loss, metrics)
            export_tokenizer(args.output_dir / "tokenizer_export.pt", pqvae, args)
        utils.save_json(args.output_dir / "pqvae_metrics.json", {"best_loss": best_loss, "history": history})
        print(format_epoch_metrics(epoch=epoch, metrics=metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
