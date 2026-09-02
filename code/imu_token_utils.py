from __future__ import annotations

import json
import zlib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import data
import model


IMU_BOS_TOKEN = "<imu_bos>"
IMU_EOS_TOKEN = "<imu_eos>"


def build_imu_code_tokens(num_codebooks: int = 2, codebook_size: int = 2048) -> list[str]:
    total = int(num_codebooks) * int(codebook_size)
    return [f"<imu_{idx:04d}>" for idx in range(total)]


def build_imu_special_tokens(num_codebooks: int = 2, codebook_size: int = 2048) -> list[str]:
    return [IMU_BOS_TOKEN, IMU_EOS_TOKEN] + build_imu_code_tokens(num_codebooks=num_codebooks, codebook_size=codebook_size)


def codes_to_local_imu_ids(codes: torch.Tensor, codebook_size: int = 2048) -> torch.Tensor:
    if codes.ndim != 3:
        raise ValueError(f"Expected codes to have shape [B, L, N], got {tuple(codes.shape)}")
    offsets = torch.arange(codes.size(-1), device=codes.device, dtype=codes.dtype).view(1, 1, -1) * int(codebook_size)
    return model.interleave_codes(codes + offsets)


def local_id_to_token(local_id: int) -> str:
    return f"<imu_{int(local_id):04d}>"


def local_ids_to_token_strings(local_ids: Iterable[int], include_bos_eos: bool = False) -> list[str]:
    tokens = [local_id_to_token(local_id) for local_id in local_ids]
    if include_bos_eos:
        return [IMU_BOS_TOKEN, *tokens, IMU_EOS_TOKEN]
    return tokens


def local_ids_to_token_text(local_ids: Iterable[int], include_bos_eos: bool = False) -> str:
    return "".join(local_ids_to_token_strings(local_ids, include_bos_eos=include_bos_eos))


def flatten_codebook_embedding(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 3:
        raise ValueError(f"Expected codebook embedding to have shape [N, K, D], got {tuple(embedding.shape)}")
    return embedding.reshape(embedding.size(0) * embedding.size(1), embedding.size(2)).contiguous()


def build_codebook_lookup_payload(tokenizer_export_obj: dict[str, Any]) -> dict[str, Any]:
    config = dict(tokenizer_export_obj["config"])
    state_dict = tokenizer_export_obj["pqvae_state_dict"]
    embedding = state_dict["quantizer.embedding"].detach().cpu().float()
    num_codebooks = int(config["num_codebooks"])
    codebook_size = int(config["codebook_size"])
    flat_codebook = flatten_codebook_embedding(embedding)
    return {
        "config": config,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "codebook_dim": int(config["codebook_dim"]),
        "flat_codebook": flat_codebook,
        "special_tokens": {
            "bos": IMU_BOS_TOKEN,
            "eos": IMU_EOS_TOKEN,
        },
        "imu_code_tokens": build_imu_code_tokens(num_codebooks=num_codebooks, codebook_size=codebook_size),
    }


def clip_seed(clip_key: data.ClipKey, seed: int) -> int:
    return int(seed) + int(clip_key.clip_start) + int(zlib.crc32(clip_key.sample_dir.encode("utf-8")))


def build_deterministic_visible_mask(
    clip_key: data.ClipKey,
    seed: int,
    num_nodes: int,
    max_visible_nodes: int,
) -> torch.Tensor:
    max_visible = max(1, min(int(max_visible_nodes), int(num_nodes)))
    rng = np.random.default_rng(clip_seed(clip_key, seed))
    num_visible = int(rng.integers(1, max_visible + 1))
    order = rng.permutation(int(num_nodes))
    mask = np.zeros(int(num_nodes), dtype=bool)
    mask[order[:num_visible]] = True
    return torch.from_numpy(mask)


def build_sample_level_train_val_split(
    records: list[data.SampleRecord],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[data.SampleRecord], list[data.SampleRecord]]:
    train_ratio = float(train_ratio)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    rng = np.random.default_rng(int(seed))
    by_script: dict[str, list[data.SampleRecord]] = defaultdict(list)
    for record in records:
        by_script[record.label_name].append(record)

    train_records: list[data.SampleRecord] = []
    val_records: list[data.SampleRecord] = []
    val_ratio = 1.0 - train_ratio
    for script_name in sorted(by_script):
        group = sorted(by_script[script_name], key=lambda record: record.sample_dir)
        order = rng.permutation(len(group))
        shuffled = [group[idx] for idx in order.tolist()]
        if len(shuffled) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(shuffled) * val_ratio)))
            n_val = min(len(shuffled) - 1, n_val)
        val_records.extend(shuffled[:n_val])
        train_records.extend(shuffled[n_val:])

    train_records = [train_records[idx] for idx in rng.permutation(len(train_records)).tolist()] if train_records else []
    val_records = [val_records[idx] for idx in rng.permutation(len(val_records)).tolist()] if val_records else []
    return train_records, val_records


def get_or_build_sample_level_train_val_split(
    records: list[data.SampleRecord],
    cache_path: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[data.SampleRecord], list[data.SampleRecord]]:
    cache_path = Path(cache_path)
    sample_dirs = [record.sample_dir for record in sorted(records, key=lambda item: item.sample_dir)]
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            obj.get("sample_dirs") == sample_dirs
            and abs(float(obj.get("train_ratio", -1.0)) - float(train_ratio)) < 1e-9
            and int(obj.get("seed", -1)) == int(seed)
        ):
            train_records = [data.SampleRecord(**row) for row in obj["train_records"]]
            val_records = [data.SampleRecord(**row) for row in obj["val_records"]]
            return train_records, val_records

    train_records, val_records = build_sample_level_train_val_split(records, train_ratio=train_ratio, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "sample_dirs": sample_dirs,
                "train_ratio": float(train_ratio),
                "seed": int(seed),
                "train_records": [asdict(record) for record in train_records],
                "val_records": [asdict(record) for record in val_records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_records, val_records
