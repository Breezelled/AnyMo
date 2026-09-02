from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ a).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


def masked_prediction_loss(
    predictor: nn.Module,
    masked_view: torch.Tensor,
    target_view: torch.Tensor,
) -> torch.Tensor:
    pred = predictor(masked_view)
    return F.mse_loss(pred, target_view.detach())


def predictive_infonce_loss(
    predictor: nn.Module,
    masked_view: torch.Tensor,
    target_view: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    pred = predictor(masked_view)
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target_view.detach(), dim=-1)
    logits = torch.einsum("btd,ctd->bc", pred, target) / float(pred.size(1))
    logits = logits / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def predictive_infonce_tokenwise_loss(
    predictor: nn.Module,
    masked_view: torch.Tensor,
    target_view: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    pred = predictor(masked_view)
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target_view.detach(), dim=-1)
    logits = torch.einsum("btd,ctd->tbc", pred, target) / float(temperature)
    labels = torch.arange(logits.size(1), device=logits.device)
    labels = labels.unsqueeze(0).expand(logits.size(0), -1).reshape(-1)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels)


def sequence_infonce_loss(
    source_view: torch.Tensor,
    target_view: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    source = F.normalize(source_view, dim=-1)
    target = F.normalize(target_view.detach(), dim=-1)
    logits = torch.einsum("btd,ctd->bc", source, target) / float(source.size(1))
    logits = logits / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def compute_pretrain_loss(
    predictor: nn.Module | None,
    full_view_a: torch.Tensor,
    full_view_b: torch.Tensor,
    masked_view_a: torch.Tensor,
    masked_view_b: torch.Tensor,
    use_sigreg: bool = True,
    pretrain_loss: str = "masked_mse",
    lambda_sigreg: float = 0.09,
    infonce_temperature: float = 0.1,
    sigreg: SIGReg | None = None,
) -> dict[str, torch.Tensor]:
    if pretrain_loss == "masked_mse":
        if predictor is None:
            raise ValueError("masked_mse requires a predictor")
        mask_loss_a = masked_prediction_loss(predictor, masked_view_a, full_view_b)
        mask_loss_b = masked_prediction_loss(predictor, masked_view_b, full_view_a)
    elif pretrain_loss == "predictive_infonce":
        if predictor is None:
            raise ValueError("predictive_infonce requires a predictor")
        mask_loss_a = predictive_infonce_loss(predictor, masked_view_a, full_view_b, infonce_temperature)
        mask_loss_b = predictive_infonce_loss(predictor, masked_view_b, full_view_a, infonce_temperature)
    elif pretrain_loss == "predictive_infonce_tokenwise":
        if predictor is None:
            raise ValueError("predictive_infonce_tokenwise requires a predictor")
        mask_loss_a = predictive_infonce_tokenwise_loss(predictor, masked_view_a, full_view_b, infonce_temperature)
        mask_loss_b = predictive_infonce_tokenwise_loss(predictor, masked_view_b, full_view_a, infonce_temperature)
    elif pretrain_loss == "graphview_infonce":
        mask_loss_a = sequence_infonce_loss(full_view_a, full_view_b, infonce_temperature)
        mask_loss_b = sequence_infonce_loss(full_view_b, full_view_a, infonce_temperature)
    else:
        raise ValueError(f"Unsupported pretrain loss: {pretrain_loss}")
    l_mask = mask_loss_a + mask_loss_b
    if use_sigreg:
        reg = sigreg if sigreg is not None else SIGReg()
        l_sig = reg(full_view_a.transpose(0, 1)) + reg(full_view_b.transpose(0, 1))
        total = l_mask + lambda_sigreg * l_sig
    else:
        l_sig = l_mask.new_zeros(())
        total = l_mask
    return {
        "loss": total,
        "mask_loss": l_mask,
        "sigreg_loss": l_sig,
    }
