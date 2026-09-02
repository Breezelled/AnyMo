from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hop_distance(num_node: int, edge: list[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    adjacency = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in edge:
        adjacency[j, i] = 1.0
        adjacency[i, j] = 1.0

    hop_dis = np.zeros((num_node, num_node), dtype=np.float32) + np.inf
    transfer_mat = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(a: np.ndarray) -> np.ndarray:
    degree = np.sum(a, axis=0)
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        if degree[i] > 0:
            out[i, i] = degree[i] ** (-1)
    return np.dot(a, out)


def zero(x):
    return 0


def iden(x):
    return x


class Graph:
    def __init__(self, strategy: str = "spatial", max_hop: int = 1, dilation: int = 1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = 23
        self.center = 0
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 4),
            (8, 7),
            (9, 8),
            (10, 9),
            (11, 4),
            (12, 11),
            (13, 12),
            (14, 13),
            (15, 0),
            (16, 15),
            (17, 16),
            (18, 17),
            (19, 0),
            (20, 19),
            (21, 20),
            (22, 21),
        ]
        self.edge = self_link + neighbor_link
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.A = self._build_adjacency(strategy)

    def _build_adjacency(self, strategy: str) -> np.ndarray:
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1.0
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            a = np.zeros((1, self.num_node, self.num_node), dtype=np.float32)
            a[0] = normalize_adjacency
            return a

        if strategy == "distance":
            a = np.zeros((len(valid_hop), self.num_node, self.num_node), dtype=np.float32)
            for i, hop in enumerate(valid_hop):
                a[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            return a

        a_parts: list[np.ndarray] = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node, self.num_node), dtype=np.float32)
            a_close = np.zeros((self.num_node, self.num_node), dtype=np.float32)
            a_further = np.zeros((self.num_node, self.num_node), dtype=np.float32)
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] != hop:
                        continue
                    if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                        a_close[j, i] = normalize_adjacency[j, i]
                    else:
                        a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                a_parts.append(a_root)
            else:
                a_parts.append(a_root + a_close)
                a_parts.append(a_further)
        return np.stack(a_parts)


class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        t_kernel_size: int = 1,
        t_stride: int = 1,
        t_padding: int = 0,
        t_dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, a)
        return x.contiguous(), a


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = zero
        elif in_channels == out_channels and stride == 1:
            self.residual = iden
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res = self.residual(x)
        x, a = self.gcn(x, a)
        x = self.tcn(x) + res
        return self.relu(x), a


def sample_visible_node_mask(
    batch_size: int,
    num_nodes: int,
    device: torch.device | None = None,
    max_visible_nodes: int = 5,
) -> torch.Tensor:
    max_visible = max(1, min(int(max_visible_nodes), int(num_nodes)))
    k = torch.randint(1, max_visible + 1, (batch_size,), device=device)
    mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
    for i in range(batch_size):
        order = torch.randperm(num_nodes, device=device)
        mask[i, order[: int(k[i].item())]] = True
    return mask


class STGCNEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, latent_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.graph = Graph()
        a = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", a)
        spatial_kernel_size = a.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        self.mask_token = nn.Parameter(torch.zeros(in_channels))
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(in_channels, 64, kernel_size, 1, residual=False),
                STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
                STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
                STGCNBlock(64, 128, kernel_size, 2, dropout=dropout),
                STGCNBlock(128, 128, kernel_size, 1, dropout=dropout),
                STGCNBlock(128, 128, kernel_size, 1, dropout=dropout),
                STGCNBlock(128, 256, kernel_size, 2, dropout=dropout),
                STGCNBlock(256, 256, kernel_size, 1, dropout=dropout),
                STGCNBlock(256, latent_dim, kernel_size, 1, dropout=dropout),
            )
        )
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.mean(dim=1)
        return x

    def apply_node_mask(self, x: torch.Tensor, visible_node_mask: torch.Tensor | None) -> torch.Tensor:
        if visible_node_mask is None:
            return x
        mask = visible_node_mask[:, None, None, :]
        token = self.mask_token.view(1, self.in_channels, 1, 1)
        return torch.where(mask, x, token)

    def forward(
        self,
        x: torch.Tensor,
        visible_node_mask: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        normalized = self.normalize_input(x)
        masked = self.apply_node_mask(normalized, visible_node_mask)
        out = masked
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            out, _ = gcn(out, self.A * importance)

        node_seq_latent = out.permute(0, 2, 3, 1).contiguous()
        global_seq_latent = node_seq_latent.mean(dim=2)
        result = {
            "node_seq_latent": node_seq_latent,
            "global_seq_latent": global_seq_latent,
        }
        if return_debug:
            result["normalized_input"] = normalized
            result["masked_input"] = masked
        return result


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalPredictor(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_tokens: int = 75,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class SequenceProjectorMLP(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        out = self.net(x.view(b * t, c))
        return out.view(b, t, -1)


class SequenceClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + residual


class TemporalConvDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(hidden_dim, kernel_size=kernel_size, dilation=dilation) for dilation in dilations]
        )
        self.out_norm = nn.GroupNorm(1, hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(self.act(self.out_norm(x)))
        return x.transpose(1, 2).contiguous()


def interleave_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 3:
        raise ValueError(f"Expected codes to have shape [B, L, N], got {tuple(codes.shape)}")
    b, l, n = codes.shape
    return codes.reshape(b, l * n)


class EMAProductQuantizer(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold_ratio: float = 0.2,
    ):
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.dead_code_threshold_ratio = float(dead_code_threshold_ratio)

        embed = torch.randn(self.num_codebooks, self.codebook_size, self.codebook_dim)
        embed = embed / math.sqrt(self.codebook_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_codebooks, self.codebook_size))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("last_dead_code_replacements", torch.zeros(self.num_codebooks, dtype=torch.long))
        self.register_buffer("total_dead_code_replacements", torch.zeros(self.num_codebooks, dtype=torch.long))

    def _flatten_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, L, C], got {tuple(x.shape)}")
        expected_dim = self.num_codebooks * self.codebook_dim
        if x.size(-1) != expected_dim:
            raise ValueError(f"Expected last dim={expected_dim}, got {x.size(-1)}")
        b, l, _ = x.shape
        return x.reshape(b * l, self.num_codebooks, self.codebook_dim).permute(1, 0, 2).contiguous()

    def _update_ema(self, flat_x: torch.Tensor, indices: torch.Tensor) -> None:
        replacement_counts: list[int] = []
        for codebook_idx in range(self.num_codebooks):
            code_ids = indices[codebook_idx]
            batch_cluster_size = torch.bincount(code_ids, minlength=self.codebook_size).to(flat_x.dtype)
            batch_embed_sum = torch.zeros_like(self.embed_avg[codebook_idx])
            batch_embed_sum.index_add_(0, code_ids, flat_x[codebook_idx])

            self.cluster_size[codebook_idx].mul_(self.decay).add_(batch_cluster_size, alpha=1.0 - self.decay)
            self.embed_avg[codebook_idx].mul_(self.decay).add_(batch_embed_sum, alpha=1.0 - self.decay)

            n = self.cluster_size[codebook_idx].sum()
            cluster_size = (
                (self.cluster_size[codebook_idx] + self.epsilon)
                / (n + self.codebook_size * self.epsilon)
                * n
            )
            normalized_embed = self.embed_avg[codebook_idx] / cluster_size.unsqueeze(1).clamp_min(self.epsilon)
            self.embedding[codebook_idx].copy_(normalized_embed)

            num_dead = 0
            if self.dead_code_threshold_ratio > 0.0 and flat_x[codebook_idx].size(0) > 0:
                avg_usage = self.cluster_size[codebook_idx].mean()
                dead_threshold = avg_usage * self.dead_code_threshold_ratio
                dead_indices = torch.where(self.cluster_size[codebook_idx] < dead_threshold)[0]
                num_dead = int(dead_indices.numel())
                if num_dead > 0:
                    num_vectors = flat_x[codebook_idx].size(0)
                    if num_vectors >= num_dead:
                        sample_indices = torch.randperm(num_vectors, device=flat_x.device)[:num_dead]
                    else:
                        sample_indices = torch.randint(0, num_vectors, (num_dead,), device=flat_x.device)
                    replacement = flat_x[codebook_idx].index_select(0, sample_indices)
                    self.embedding[codebook_idx, dead_indices] = replacement
                    self.embed_avg[codebook_idx, dead_indices] = replacement
                    self.cluster_size[codebook_idx, dead_indices] = avg_usage.clamp_min(1.0)
            replacement_counts.append(num_dead)

        replacement_tensor = torch.tensor(replacement_counts, device=self.last_dead_code_replacements.device, dtype=torch.long)
        self.last_dead_code_replacements.copy_(replacement_tensor)
        self.total_dead_code_replacements.add_(replacement_tensor)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_x = self._flatten_inputs(x)
        quantized_chunks: list[torch.Tensor] = []
        code_chunks: list[torch.Tensor] = []
        commitment_terms: list[torch.Tensor] = []

        for codebook_idx in range(self.num_codebooks):
            embed = self.embedding[codebook_idx]
            chunk = flat_x[codebook_idx]
            distances = (
                chunk.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * chunk @ embed.t()
                + embed.pow(2).sum(dim=1).unsqueeze(0)
            )
            indices = distances.argmin(dim=1)
            quantized = embed.index_select(0, indices)
            quantized_chunks.append(quantized)
            code_chunks.append(indices)
            commitment_terms.append(F.mse_loss(chunk, quantized.detach()))

        code_tensor = torch.stack(code_chunks, dim=0)
        quantized_flat = torch.stack(quantized_chunks, dim=0)

        if self.training:
            with torch.no_grad():
                self._update_ema(flat_x.detach(), code_tensor.detach())

        quantized_st = flat_x + (quantized_flat - flat_x).detach()
        b, l, _ = x.shape
        quantized = quantized_st.permute(1, 0, 2).reshape(b, l, self.num_codebooks * self.codebook_dim)
        codes = code_tensor.permute(1, 0).reshape(b, l, self.num_codebooks)
        commitment_loss = torch.stack(commitment_terms).sum()
        return quantized, codes, commitment_loss

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(f"Expected codes to have shape [B, L, N], got {tuple(codes.shape)}")
        if codes.size(-1) != self.num_codebooks:
            raise ValueError(f"Expected last dim={self.num_codebooks}, got {codes.size(-1)}")
        b, l, _ = codes.shape
        chunks: list[torch.Tensor] = []
        for codebook_idx in range(self.num_codebooks):
            embed = self.embedding[codebook_idx]
            ids = codes[:, :, codebook_idx].reshape(-1)
            chunks.append(embed.index_select(0, ids))
        return torch.stack(chunks, dim=1).reshape(b, l, self.num_codebooks * self.codebook_dim)


class MotionPQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        bottleneck_dim: int = 128,
        num_codebooks: int = 2,
        codebook_size: int = 2048,
        codebook_dim: int = 64,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_threshold_ratio: float = 0.2,
    ):
        super().__init__()
        expected_bottleneck_dim = int(num_codebooks) * int(codebook_dim)
        if bottleneck_dim != expected_bottleneck_dim:
            raise ValueError(
                f"bottleneck_dim must equal num_codebooks * codebook_dim "
                f"({expected_bottleneck_dim}), got {bottleneck_dim}"
            )
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.commitment_weight = float(commitment_weight)

        self.pre_quant = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.bottleneck_dim),
        )
        self.quantizer = EMAProductQuantizer(
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            decay=ema_decay,
            epsilon=ema_epsilon,
            dead_code_threshold_ratio=dead_code_threshold_ratio,
        )
        self.decoder = TemporalConvDecoder(
            input_dim=self.bottleneck_dim,
            output_dim=self.input_dim,
            hidden_dim=max(self.bottleneck_dim, self.input_dim),
        )

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        pre_quant = self.pre_quant(x)
        _, codes, _ = self.quantizer(pre_quant)
        return codes

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return interleave_codes(self.encode_codes(x))

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode_codes(codes)
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_quant = self.pre_quant(x)
        quantized, codes, commitment_loss = self.quantizer(pre_quant)
        reconstruction = self.decoder(quantized)
        recon_loss = F.smooth_l1_loss(reconstruction, x)
        loss = recon_loss + self.commitment_weight * commitment_loss
        return {
            "pre_quant": pre_quant,
            "quantized": quantized,
            "codes": codes,
            "tokens": interleave_codes(codes),
            "reconstruction": reconstruction,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "dead_code_replacements": self.quantizer.last_dead_code_replacements.clone(),
            "loss": loss,
        }
