from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import data
import loss
import model
import training_utils as utils


def build_argparser() -> argparse.ArgumentParser:
    cfg = config.TrainSTGCNConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the ST-GCN encoder with masked cross-view "
            "predictive-contrastive learning."
        )
    )
    parser.add_argument("--summary-csv", type=Path, default=cfg.summary_csv)
    parser.add_argument("--base-dir", type=Path, default=cfg.base_dir)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--split-cache", type=Path, default=cfg.split_cache)
    parser.add_argument("--candidate-name", type=str, default=cfg.candidate_name)
    parser.add_argument("--real1-device-suffix", type=str, default=cfg.real1_device_suffix)
    parser.add_argument("--real2-device-suffix", type=str, default=cfg.real2_device_suffix)
    parser.add_argument("--window-size", type=int, default=cfg.window_size)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--pretrain-loss", type=str, choices=config.PRETRAIN_LOSS_CHOICES, default=cfg.pretrain_loss)
    parser.add_argument("--sigreg", dest="use_sigreg", action="store_true")
    parser.add_argument("--no-sigreg", dest="use_sigreg", action="store_false")
    parser.add_argument("--lambda-sigreg", type=float, default=cfg.lambda_sigreg)
    parser.add_argument("--infonce-temperature", type=float, default=cfg.infonce_temperature)
    parser.add_argument("--graphview-proj-dim", type=int, default=cfg.graphview_proj_dim)
    parser.add_argument("--max-visible-nodes", type=int, default=cfg.max_visible_nodes)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--surface-rotation-augment", dest="surface_rotation_augment", action="store_true")
    parser.add_argument("--no-surface-rotation-augment", dest="surface_rotation_augment", action="store_false")
    parser.add_argument("--surface-rotation-inplane-max-deg", type=float, default=cfg.surface_rotation_inplane_max_deg)
    parser.add_argument("--surface-rotation-tilt-max-deg", type=float, default=cfg.surface_rotation_tilt_max_deg)
    parser.set_defaults(surface_rotation_augment=cfg.surface_rotation_augment, use_sigreg=cfg.use_sigreg)
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def build_models(args: argparse.Namespace) -> tuple[model.STGCNEncoder, model.TemporalPredictor, model.SequenceProjectorMLP]:
    encoder = model.STGCNEncoder(in_channels=6, latent_dim=256, dropout=args.dropout)
    predictor = model.TemporalPredictor(dim=256)
    projector = model.SequenceProjectorMLP(in_dim=256, proj_dim=args.graphview_proj_dim)
    return encoder, predictor, projector


def maybe_compile_models(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    projector: torch.nn.Module,
    *,
    enabled: bool,
    mode: str,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    if not enabled:
        return encoder, predictor, projector
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("torch.compile was requested, but the active torch build does not provide it.")
    return (
        compile_fn(encoder, mode=mode),
        compile_fn(predictor, mode=mode),
        compile_fn(projector, mode=mode),
    )


def build_optimizer(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    projector: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    params = list(encoder.parameters())
    if args.pretrain_loss != "graphview_infonce":
        params += list(predictor.parameters())
    else:
        params += list(projector.parameters())
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def save_checkpoint(
    ckpt_path: Path,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    projector: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "encoder_state_dict": utils.export_module_state_dict(encoder),
            "predictor_state_dict": utils.export_module_state_dict(predictor),
            "projector_state_dict": utils.export_module_state_dict(projector),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )


def export_encoder_checkpoint(path: Path, encoder: torch.nn.Module, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": int(epoch), "encoder_state_dict": utils.export_module_state_dict(encoder)}, path)


def load_checkpoint(
    ckpt_path: Path,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    projector: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    obj = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(utils.normalize_compiled_state_dict_keys(obj["encoder_state_dict"]))
    predictor.load_state_dict(utils.normalize_compiled_state_dict_keys(obj["predictor_state_dict"]))
    if projector is not None and "projector_state_dict" in obj:
        projector.load_state_dict(utils.normalize_compiled_state_dict_keys(obj["projector_state_dict"]))
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj


def run_pretrain_step(
    encoder: model.STGCNEncoder,
    predictor: model.TemporalPredictor,
    projector: model.SequenceProjectorMLP,
    batch: dict[str, torch.Tensor],
    use_sigreg: bool,
    pretrain_loss: str,
    lambda_sigreg: float,
    infonce_temperature: float,
    max_visible_nodes: int,
) -> dict[str, torch.Tensor]:
    view_a = batch["view_a_full"]
    view_b = batch["view_b_full"]
    full_a = encoder(view_a)
    full_b = encoder(view_b)
    if pretrain_loss == "graphview_infonce":
        proj_a = projector(full_a["global_seq_latent"])
        proj_b = projector(full_b["global_seq_latent"])
        masked_a = {"global_seq_latent": proj_a}
        masked_b = {"global_seq_latent": proj_b}
        full_a = {"global_seq_latent": proj_a}
        full_b = {"global_seq_latent": proj_b}
        predictor_for_loss = None
    else:
        visible_a = model.sample_visible_node_mask(
            view_a.size(0),
            view_a.size(3),
            device=view_a.device,
            max_visible_nodes=max_visible_nodes,
        )
        visible_b = model.sample_visible_node_mask(
            view_b.size(0),
            view_b.size(3),
            device=view_b.device,
            max_visible_nodes=max_visible_nodes,
        )
        masked_a = encoder(view_a, visible_node_mask=visible_a)
        masked_b = encoder(view_b, visible_node_mask=visible_b)
        predictor_for_loss = predictor
    return loss.compute_pretrain_loss(
        predictor=predictor_for_loss,
        full_view_a=full_a["global_seq_latent"],
        full_view_b=full_b["global_seq_latent"],
        masked_view_a=masked_a["global_seq_latent"],
        masked_view_b=masked_b["global_seq_latent"],
        use_sigreg=use_sigreg,
        pretrain_loss=pretrain_loss,
        lambda_sigreg=lambda_sigreg,
        infonce_temperature=infonce_temperature,
    )


def run_epoch(
    encoder: model.STGCNEncoder,
    predictor: model.TemporalPredictor,
    projector: model.SequenceProjectorMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_sigreg: bool,
    pretrain_loss: str,
    lambda_sigreg: float,
    infonce_temperature: float,
    max_visible_nodes: int,
    desc: str,
) -> dict[str, float]:
    train_mode = optimizer is not None
    encoder.train(train_mode)
    predictor.train(train_mode)
    projector.train(train_mode)
    total_loss = 0.0
    total_mask = 0.0
    total_sig = 0.0
    count = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = utils.move_batch_to_device(batch, device)
        out = run_pretrain_step(
            encoder,
            predictor,
            projector,
            batch,
            use_sigreg,
            pretrain_loss,
            lambda_sigreg,
            infonce_temperature,
            max_visible_nodes,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()
        bs = int(batch["view_a_full"].size(0))
        total_loss += float(out["loss"].item()) * bs
        total_mask += float(out["mask_loss"].item()) * bs
        total_sig += float(out["sigreg_loss"].item()) * bs
        count += bs
    return {
        "loss": total_loss / max(count, 1),
        "mask_loss": total_mask / max(count, 1),
        "sigreg_loss": total_sig / max(count, 1),
    }


def build_pretrain_loaders(args: argparse.Namespace) -> DataLoader:
    records = data.exclude_held_out_subjects(
        data.load_summary_records(args.summary_csv, args.candidate_name, args.base_dir)
    )
    rng = np.random.default_rng(args.seed)
    record_order = rng.permutation(len(records)).tolist()
    records = [records[i] for i in record_order]
    train_keys = data.build_all_clip_index(
        records=records,
        window_size=args.window_size,
        real1_device_suffix=args.real1_device_suffix,
        real2_device_suffix=args.real2_device_suffix,
    )
    record_by_sample = {record.sample_dir: record for record in records}
    synth_store = data.DominantTop2Store()
    train_ds = data.PretrainDataset(
        train_keys,
        record_by_sample,
        synth_store,
        window_size=args.window_size,
        seed=args.seed,
        surface_rotation_augment=args.surface_rotation_augment,
        inplane_max_deg=args.surface_rotation_inplane_max_deg,
        tilt_max_deg=args.surface_rotation_tilt_max_deg,
    )
    loader_kwargs = {"pin_memory": True}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **loader_kwargs,
    )
    return train_loader


def main() -> int:
    args = parse_args()
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, projector = build_models(args)
    encoder.to(device)
    predictor.to(device)
    projector.to(device)
    encoder, predictor, projector = maybe_compile_models(
        encoder,
        predictor,
        projector,
        enabled=args.torch_compile,
        mode=args.torch_compile_mode,
    )
    optimizer = build_optimizer(encoder, predictor, projector, args)
    train_loader = build_pretrain_loaders(args)

    best_train_loss = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            encoder,
            predictor,
            projector,
            train_loader,
            optimizer,
            device,
            args.use_sigreg,
            args.pretrain_loss,
            args.lambda_sigreg,
            args.infonce_temperature,
            args.max_visible_nodes,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
        )
        save_checkpoint(args.output_dir / "last.pt", encoder, predictor, projector, optimizer, epoch, best_train_loss)
        if train_metrics["loss"] < best_train_loss:
            best_train_loss = train_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(args.output_dir / "best.pt", encoder, predictor, projector, optimizer, epoch, best_train_loss)
            export_encoder_checkpoint(args.output_dir / "encoder_best.pt", encoder, epoch)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"best_train_loss={best_train_loss:.6f} best_epoch={best_epoch}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
