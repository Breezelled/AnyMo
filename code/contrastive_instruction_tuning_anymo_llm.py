from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import Trainer, TrainingArguments
except ModuleNotFoundError:  # Keep lightweight unit tests runnable outside the AnyMo env.
    class Trainer:  # type: ignore[no-redef]
        pass

    TrainingArguments = None  # type: ignore[assignment]

import config
from anymo_swift_model import AnyMoRuntimeReplacementWrapper, _resolve_llm_hidden_size


DEFAULT_LM_TRAIN_JSONL = config.INSTRUCTION_DATA_DIR / "train.jsonl"
DEFAULT_CONTRASTIVE_TRAIN_JSONL = (
    config.INSTRUCTION_DATA_DIR / "train_contrastive.jsonl"
)
DEFAULT_CODEBOOK_ARTIFACT = config.INSTRUCTION_DATA_DIR / "imu_codebook_lookup.pt"
DEFAULT_CACHE_ROOT = config.CACHE_ROOT
DEFAULT_HF_HOME = DEFAULT_CACHE_ROOT / "huggingface"
DEFAULT_HF_HUB_CACHE = DEFAULT_HF_HOME / "hub"
DEFAULT_MODELSCOPE_CACHE = DEFAULT_CACHE_ROOT / "modelscope"

IMU_CONTRASTIVE_PROMPT_TEMPLATE = (
    "Represent the human motion from the wearable IMU motion tokens.\n\n"
    "The IMU tokens are from IMU sensors attached to the user's {sensor_context}.\n\n"
    "Input IMU token:\n{imu_token}\n\n"
    "Return a compact embedding of the motion."
)
TEXT_CONTRASTIVE_PREFIX = "Represent the human motion described by the text.\n\nMotion description:\n"
TEXT_CONTRASTIVE_SUFFIX = "\n\nReturn a compact embedding of the motion."

BACKBONE_SPECS = {
    "qwen2_5": {
        "model": "Qwen/Qwen2.5-0.5B",
        "pretrained_checkpoint": config.MOTION_LANGUAGE_MODEL_DIR,
        "output_dir": config.ANYMO_MODEL_DIR,
    },
}


def format_imu_contrastive_prompt(imu_token_text: str, sensor_context: str) -> str:
    return IMU_CONTRASTIVE_PROMPT_TEMPLATE.format(
        imu_token=str(imu_token_text),
        sensor_context=str(sensor_context or "visible body segments"),
    )


def format_text_contrastive_prompt(positive_text: str) -> str:
    return f"{TEXT_CONTRASTIVE_PREFIX}{str(positive_text)}{TEXT_CONTRASTIVE_SUFFIX}"


def normalize_activity_label(label: Any) -> str:
    return " ".join(str(label or "").strip().lower().split())


def format_label_contrastive_prompt(activity_label: Any) -> str:
    return format_text_contrastive_prompt(normalize_activity_label(activity_label))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AnyMo with causal LM loss plus IMU-text contrastive alignment.",
    )
    parser.add_argument("--backbone", choices=sorted(BACKBONE_SPECS), default="qwen2_5")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint", dest="pretrained_checkpoint", type=Path, default=None)
    parser.add_argument("--lm-train-jsonl", type=Path, default=DEFAULT_LM_TRAIN_JSONL)
    parser.add_argument("--contrastive-train-jsonl", type=Path, default=DEFAULT_CONTRASTIVE_TRAIN_JSONL)
    parser.add_argument("--codebook-artifact", type=Path, default=DEFAULT_CODEBOOK_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--contrastive-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-imu-length", type=int, default=2048)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--lm-loss-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--label-contrastive-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the supervised label-level contrastive loss derived from activity_label. "
            "Set >0 to add an in-batch multi-positive IMU-label contrastive branch."
        ),
    )
    parser.add_argument(
        "--grad-cache-steps",
        type=int,
        default=1,
        help=(
            "Number of contrastive micro-batches to cache per optimizer step. "
            "Effective local contrastive batch = contrastive_batch_size * grad_cache_steps."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--pooler-latents", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument(
        "--text-soft-prompt-length",
        type=int,
        default=0,
        help=(
            "Number of shared learnable soft prompt tokens prepended to text/label contrastive branches. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--imu-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for the post-projector motion-token residual added to the final IMU contrastive embedding. "
            "Set to 0 to disable the residual path."
        ),
    )
    parser.add_argument("--projection-hidden-size", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument(
        "--deepspeed",
        type=Path,
        default=None,
        help="Optional DeepSpeed config path. Prefer ZeRO-2 without CPU offload for this script.",
    )
    parser.add_argument("--pretokenize", dest="pretokenize", action="store_true", default=True)
    parser.add_argument("--no-pretokenize", dest="pretokenize", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-impl", "--attn_impl", dest="attn_impl", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", "--gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-hf", type=str, default="true")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--hf-hub-cache", type=Path, default=DEFAULT_HF_HUB_CACHE)
    parser.add_argument("--modelscope-cache", type=Path, default=DEFAULT_MODELSCOPE_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_backbone_config(args: argparse.Namespace) -> dict[str, Any]:
    spec = BACKBONE_SPECS[args.backbone]
    return {
        "model": args.model or spec["model"],
        "pretrained_checkpoint": Path(args.pretrained_checkpoint) if args.pretrained_checkpoint else spec["pretrained_checkpoint"],
        "output_dir": Path(args.output_dir) if args.output_dir else spec["output_dir"],
    }


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def _is_import_available(*module_names: str) -> bool:
    return any(importlib.util.find_spec(module_name) is not None for module_name in module_names)


def build_attention_debug_summary(config_obj: Any) -> dict[str, Any]:
    if hasattr(config_obj, "to_dict"):
        config_dict = dict(config_obj.to_dict())
    else:
        config_dict = dict(vars(config_obj))
    interesting_tokens = ("attn", "attention", "linear", "conv", "kernel")
    attention_config = {
        key: value
        for key, value in sorted(config_dict.items())
        if any(token in str(key).lower() for token in interesting_tokens)
    }
    return {
        "config": attention_config,
        "package_available": {
            "flash_attn": _is_import_available("flash_attn"),
            # flash-linear-attention commonly imports as ``fla``; keep the
            # package-style name in the log because that is what pip displays.
            "flash_linear_attention": _is_import_available("flash_linear_attention", "fla"),
            "causal_conv1d": _is_import_available("causal_conv1d"),
        },
    }


def set_local_cuda_device_from_env() -> int | None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return None
    device_index = int(local_rank)
    torch.cuda.set_device(device_index)
    return device_index


def log_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    print(
        json.dumps(
            {
                "event": label,
                "rank": int(os.environ.get("RANK", "0")),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "cuda_device": int(device),
                "allocated_gib": round(torch.cuda.memory_allocated(device) / 1024**3, 4),
                "reserved_gib": round(torch.cuda.memory_reserved(device) / 1024**3, 4),
                "free_gib": round(free / 1024**3, 4),
                "total_gib": round(total / 1024**3, 4),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def log_model_placement(model: nn.Module, label: str) -> None:
    summary: dict[str, dict[str, float]] = {}
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    for param in model.parameters():
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        total_bytes += numel * int(param.element_size())
        key = f"{param.device}:{param.dtype}"
        bucket = summary.setdefault(key, {"params_m": 0.0, "bytes_gib": 0.0})
        bucket["params_m"] += numel / 1e6
        bucket["bytes_gib"] += numel * int(param.element_size()) / 1024**3
    for bucket in summary.values():
        bucket["params_m"] = round(bucket["params_m"], 4)
        bucket["bytes_gib"] = round(bucket["bytes_gib"], 4)
    print(
        json.dumps(
            {
                "event": label,
                "rank": int(os.environ.get("RANK", "0")),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "total_params_m": round(total_params / 1e6, 4),
                "trainable_params_m": round(trainable_params / 1e6, 4),
                "parameter_bytes_gib": round(total_bytes / 1024**3, 4),
                "by_device_dtype": summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def prewarm_nccl_process_group(label: str = "prewarm_nccl_process_group") -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.cuda.is_available() or not dist.is_available():
        return
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialized_here = True
    device = torch.device("cuda", torch.cuda.current_device())
    probe = torch.ones((), device=device)
    dist.all_reduce(probe)
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    print(
        json.dumps(
            {
                "event": label,
                "rank": int(os.environ.get("RANK", "0")),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "backend": dist.get_backend(),
                "initialized_here": initialized_here,
                "value": float(probe.item()),
                "allocated_gib": round(torch.cuda.memory_allocated(device) / 1024**3, 4),
                "reserved_gib": round(torch.cuda.memory_reserved(device) / 1024**3, 4),
                "free_gib": round(free / 1024**3, 4),
                "total_gib": round(total / 1024**3, 4),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def skip_transformers_cuda_allocator_warmup():
    import transformers.modeling_utils as modeling_utils

    original_warmup = modeling_utils.caching_allocator_warmup

    def _noop_caching_allocator_warmup(*args, **kwargs):
        print(
            json.dumps(
                {
                    "event": "skip_transformers_cuda_allocator_warmup",
                    "rank": int(os.environ.get("RANK", "0")),
                    "local_rank": os.environ.get("LOCAL_RANK"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return None

    modeling_utils.caching_allocator_warmup = _noop_caching_allocator_warmup
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = original_warmup


_DDP_INIT_SETTINGS_PATCHED = False


def patch_ddp_init_settings(*, init_sync: bool, gradient_as_bucket_view: bool) -> None:
    global _DDP_INIT_SETTINGS_PATCHED
    if _DDP_INIT_SETTINGS_PATCHED:
        return
    ddp_cls = torch.nn.parallel.DistributedDataParallel
    original_init = ddp_cls.__init__

    def _init_settings(self, module, *args, **kwargs):
        kwargs.setdefault("init_sync", bool(init_sync))
        kwargs.setdefault("gradient_as_bucket_view", bool(gradient_as_bucket_view))
        print(
            json.dumps(
                {
                    "event": "patch_ddp_init_settings",
                    "rank": int(os.environ.get("RANK", "0")),
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "init_sync": kwargs.get("init_sync"),
                    "gradient_as_bucket_view": kwargs.get("gradient_as_bucket_view"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return original_init(self, module, *args, **kwargs)

    ddp_cls.__init__ = _init_settings
    _DDP_INIT_SETTINGS_PATCHED = True


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= int(max_samples):
                break
    return rows


class JsonlDataset(Dataset):
    def __init__(self, path: Path, max_samples: int | None = None):
        self.path = Path(path)
        self.rows = read_jsonl(self.path, max_samples=max_samples)
        if not self.rows:
            raise ValueError(f"No rows found in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class MappedDataset(Dataset):
    def __init__(self, source: Dataset, map_fn, *, name: str, log_every: int = 100_000):
        self.rows: list[dict[str, Any]] = []
        total = len(source)
        print(f"Pre-tokenizing {name} dataset: {total} rows")
        for index in range(total):
            if index and index % int(log_every) == 0:
                print(f"Pre-tokenized {name}: {index}/{total}")
            self.rows.append(map_fn(source[index]))
        print(f"Finished pre-tokenizing {name}: {len(self.rows)} rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long)
    for row_index, seq in enumerate(sequences):
        if seq:
            out[row_index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def _pad_masks(masks: list[list[bool]]) -> torch.Tensor:
    if not masks:
        return torch.empty((0, 0), dtype=torch.bool)
    max_len = max(len(mask) for mask in masks)
    out = torch.zeros((len(masks), max_len), dtype=torch.bool)
    for row_index, mask in enumerate(masks):
        if mask:
            out[row_index, : len(mask)] = torch.tensor(mask, dtype=torch.bool)
    return out


def _truncate_pair(ids: list[int], mask: list[bool], max_length: int) -> tuple[list[int], list[bool]]:
    if max_length <= 0:
        return ids, mask
    return ids[:max_length], mask[:max_length]


def format_qwen_plain_lm(messages: list[dict[str, str]]) -> tuple[str, str]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("LM examples must end with an assistant message.")
    prompt_parts: list[str] = []
    for message in messages[:-1]:
        prompt_parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")
    answer = f"{messages[-1]['content']}<|im_end|>\n"
    return "".join(prompt_parts), answer


def tokenize_lm_example(example: dict[str, Any], *, tokenizer: Any, max_length: int) -> dict[str, Any]:
    prompt, answer = format_qwen_plain_lm(example["messages"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_ids = (prompt_ids + answer_ids)[: int(max_length)]
    labels = ([-100] * len(prompt_ids) + answer_ids)[: int(max_length)]
    return {"input_ids": input_ids, "labels": labels}


class LmDataCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_id_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for example in examples:
            if "input_ids" in example and "labels" in example:
                input_ids = list(example["input_ids"])[: self.max_length]
                labels = list(example["labels"])[: self.max_length]
            else:
                tokenized = tokenize_lm_example(example, tokenizer=self.tokenizer, max_length=self.max_length)
                input_ids = tokenized["input_ids"]
                labels = tokenized["labels"]
            input_id_rows.append(input_ids)
            label_rows.append(labels)

        pad_token_id = int(self.tokenizer.pad_token_id)
        input_ids_tensor = _pad_sequences(input_id_rows, pad_token_id)
        labels_tensor = _pad_sequences(label_rows, -100)
        attention_mask = input_ids_tensor.ne(pad_token_id).long()
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }


def _encode_text_prompt_with_content_mask(tokenizer: Any, text: str, max_length: int) -> tuple[list[int], list[bool]]:
    prefix_ids = tokenizer.encode(TEXT_CONTRASTIVE_PREFIX, add_special_tokens=False)
    content_ids = tokenizer.encode(str(text), add_special_tokens=False)
    suffix_ids = tokenizer.encode(TEXT_CONTRASTIVE_SUFFIX, add_special_tokens=False)
    ids = prefix_ids + content_ids + suffix_ids
    mask = [False] * len(prefix_ids) + [True] * len(content_ids) + [False] * len(suffix_ids)
    return _truncate_pair(ids, mask, max_length)


def _encode_label_prompt_with_content_mask(tokenizer: Any, activity_label: Any, max_length: int) -> tuple[list[int], list[bool]]:
    return _encode_text_prompt_with_content_mask(
        tokenizer,
        normalize_activity_label(activity_label),
        max_length=max_length,
    )


def tokenize_contrastive_example(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    imu_code_token_ids: list[int] | set[int],
    max_imu_length: int,
    max_text_length: int,
) -> dict[str, Any]:
    imu_prompt = format_imu_contrastive_prompt(
        imu_token_text=str(example["imu_token_text"]),
        sensor_context=str(example.get("sensor_context", "visible body segments")),
    )
    imu_ids = tokenizer.encode(imu_prompt, add_special_tokens=False)[: int(max_imu_length)]
    imu_code_token_ids = {int(token_id) for token_id in imu_code_token_ids}
    imu_mask = [int(token_id) in imu_code_token_ids for token_id in imu_ids]

    positive_text_options: list[dict[str, Any]] = []
    for item in example.get("positive_texts", []):
        text = str(item.get("text", ""))
        if not text.strip():
            continue
        text_ids, text_mask = _encode_text_prompt_with_content_mask(
            tokenizer,
            text,
            max_length=int(max_text_length),
        )
        positive_text_options.append(
            {
                "type": str(item.get("type", "")),
                "text_input_ids": text_ids,
                "text_token_mask": text_mask,
            }
        )
    if not positive_text_options:
        raise ValueError("Contrastive example has no non-empty positive_texts.")
    activity_label = normalize_activity_label(example.get("activity_label", ""))
    label_ids, label_mask = _encode_label_prompt_with_content_mask(
        tokenizer,
        activity_label,
        max_length=int(max_text_length),
    )

    return {
        "imu_input_ids": imu_ids,
        "imu_token_mask": imu_mask,
        "positive_text_options": positive_text_options,
        "activity_label": activity_label,
        "label_input_ids": label_ids,
        "label_token_mask": label_mask,
    }


class ContrastiveDataCollator:
    def __init__(
        self,
        tokenizer: Any,
        imu_code_token_ids: list[int] | set[int],
        max_imu_length: int,
        max_text_length: int,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.imu_code_token_ids = {int(token_id) for token_id in imu_code_token_ids}
        self.max_imu_length = int(max_imu_length)
        self.max_text_length = int(max_text_length)
        self.rng = random.Random(int(seed))

    def _sample_positive(self, example: dict[str, Any]) -> dict[str, str]:
        if "positive_text_options" in example:
            positives = list(example.get("positive_text_options", []))
        else:
            positives = [item for item in example.get("positive_texts", []) if str(item.get("text", "")).strip()]
        if not positives:
            raise ValueError("Contrastive example has no non-empty positive_texts.")
        return self.rng.choice(positives)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        imu_rows: list[list[int]] = []
        imu_masks: list[list[bool]] = []
        text_rows: list[list[int]] = []
        text_masks: list[list[bool]] = []
        label_rows: list[list[int]] = []
        label_masks: list[list[bool]] = []
        activity_labels: list[str] = []
        positive_types: list[str] = []

        for example in examples:
            if (
                "imu_input_ids" in example
                and "imu_token_mask" in example
                and "positive_text_options" in example
                and "label_input_ids" in example
                and "label_token_mask" in example
            ):
                positive = self._sample_positive(example)
                imu_ids = list(example["imu_input_ids"])[: self.max_imu_length]
                imu_mask = list(example["imu_token_mask"])[: len(imu_ids)]
                text_ids = list(positive["text_input_ids"])[: self.max_text_length]
                text_mask = list(positive["text_token_mask"])[: len(text_ids)]
                activity_label = normalize_activity_label(example.get("activity_label", ""))
                label_ids = list(example["label_input_ids"])[: self.max_text_length]
                label_mask = list(example["label_token_mask"])[: len(label_ids)]
            else:
                tokenized = tokenize_contrastive_example(
                    example,
                    tokenizer=self.tokenizer,
                    imu_code_token_ids=self.imu_code_token_ids,
                    max_imu_length=self.max_imu_length,
                    max_text_length=self.max_text_length,
                )
                positive = self._sample_positive(tokenized)
                imu_ids = tokenized["imu_input_ids"]
                imu_mask = tokenized["imu_token_mask"]
                text_ids = positive["text_input_ids"]
                text_mask = positive["text_token_mask"]
                activity_label = str(tokenized["activity_label"])
                label_ids = tokenized["label_input_ids"]
                label_mask = tokenized["label_token_mask"]
            positive_types.append(str(positive.get("type", "")))
            imu_rows.append(imu_ids)
            imu_masks.append(imu_mask)
            text_rows.append(text_ids)
            text_masks.append(text_mask)
            label_rows.append(label_ids)
            label_masks.append(label_mask)
            activity_labels.append(activity_label)

        pad_token_id = int(self.tokenizer.pad_token_id)
        imu_input_ids = _pad_sequences(imu_rows, pad_token_id)
        text_input_ids = _pad_sequences(text_rows, pad_token_id)
        label_input_ids = _pad_sequences(label_rows, pad_token_id)
        return {
            "imu_input_ids": imu_input_ids,
            "imu_attention_mask": imu_input_ids.ne(pad_token_id).long(),
            "imu_token_mask": _pad_masks(imu_masks),
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_input_ids.ne(pad_token_id).long(),
            "text_token_mask": _pad_masks(text_masks),
            "label_input_ids": label_input_ids,
            "label_attention_mask": label_input_ids.ne(pad_token_id).long(),
            "label_token_mask": _pad_masks(label_masks),
            "activity_labels": activity_labels,
            "positive_types": positive_types,
        }


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, mult: int = 2):
        super().__init__()
        inner_size = int(hidden_size) * int(mult)
        self.norm = nn.RMSNorm(int(hidden_size))
        self.gate_proj = nn.Linear(int(hidden_size), inner_size)
        self.up_proj = nn.Linear(int(hidden_size), inner_size)
        self.down_proj = nn.Linear(inner_size, int(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        return self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))


class LatentAttentionPooler(nn.Module):
    def __init__(self, hidden_size: int, num_latents: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_latents = int(num_latents)
        self.num_heads = 8
        self.ff_mult = 2
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size={self.hidden_size} must be divisible by {self.num_heads} attention heads.")
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.hidden_size) * 0.02)
        self.token_norm = nn.RMSNorm(self.hidden_size)
        self.latent_norm = nn.RMSNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.ffn = SwiGLUFFN(self.hidden_size, mult=self.ff_mult)
        self.norm = nn.RMSNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        module_dtype = self.attn.in_proj_weight.dtype
        if hidden_states.dtype != module_dtype:
            hidden_states = hidden_states.to(dtype=module_dtype)
        token_mask = token_mask.to(device=hidden_states.device, dtype=torch.bool)
        valid = token_mask.any(dim=1)
        safe_mask = token_mask.clone()
        if not bool(valid.all()):
            safe_mask[~valid] = True
        latents = self.latents.to(device=hidden_states.device, dtype=hidden_states.dtype)
        latents = latents.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        attn_out, _ = self.attn(
            query=self.token_norm(hidden_states),
            key=self.latent_norm(latents),
            value=latents,
            need_weights=False,
        )
        enhanced = hidden_states + attn_out
        enhanced = enhanced + self.ffn(enhanced)
        weights = safe_mask.unsqueeze(-1).to(dtype=enhanced.dtype)
        pooled = (enhanced * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


def attach_contrastive_modules(
    model: AnyMoRuntimeReplacementWrapper,
    *,
    pooler_latents: int,
    embedding_dim: int | None,
    imu_residual_scale: float = 1.0,
    text_soft_prompt_length: int = 0,
) -> AnyMoRuntimeReplacementWrapper:
    hidden_size = _resolve_llm_hidden_size(model.llm)
    out_dim = int(embedding_dim or hidden_size)
    model.imu_pooler = LatentAttentionPooler(hidden_size, num_latents=pooler_latents)
    model.text_pooler = LatentAttentionPooler(hidden_size, num_latents=pooler_latents)
    model.imu_embedding_head = nn.Linear(hidden_size, out_dim, bias=False)
    model.text_embedding_head = nn.Linear(hidden_size, out_dim, bias=False)
    if int(text_soft_prompt_length) > 0:
        model.text_soft_prompt = nn.Parameter(torch.randn(int(text_soft_prompt_length), hidden_size) * 0.02)
    if float(imu_residual_scale) != 0.0:
        model.imu_residual_norm = nn.RMSNorm(hidden_size)
        model.imu_residual_head = nn.Linear(hidden_size, out_dim, bias=False)
        model.register_buffer("imu_residual_scale", torch.tensor(float(imu_residual_scale), dtype=torch.float32))
    return model


def freeze_unused_vision_modules(llm: nn.Module) -> list[str]:
    """Freeze vision towers present in multimodal Qwen variants.

    The AnyMo training loop feeds only text/IMU-token prompts. Multimodal
    backbones such as Qwen3.5 may still carry a vision tower, which DDP sees as
    trainable-but-unused unless it is frozen.
    """
    frozen_modules: list[str] = []
    candidates: list[tuple[str, nn.Module]] = []
    backbone = getattr(llm, "model", None)
    if backbone is not None and isinstance(getattr(backbone, "visual", None), nn.Module):
        candidates.append(("model.visual", backbone.visual))
    if isinstance(getattr(llm, "visual", None), nn.Module):
        candidates.append(("visual", llm.visual))

    seen: set[int] = set()
    for name, module in candidates:
        if id(module) in seen:
            continue
        seen.add(id(module))
        for param in module.parameters():
            param.requires_grad_(False)
        frozen_modules.append(name)
    return frozen_modules


def unwrap_distributed_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _pool_projected_imu_token_embeddings(
    base_model: AnyMoRuntimeReplacementWrapper,
    *,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    residual_head = getattr(base_model, "imu_residual_head", None)
    hidden_size = int(residual_head.in_features) if residual_head is not None else _resolve_llm_hidden_size(base_model.llm)
    residual_norm = getattr(base_model, "imu_residual_norm", None)
    residual_param = None
    if residual_norm is not None:
        residual_param = next(residual_norm.parameters(), None)
    if residual_param is None and residual_head is not None:
        residual_param = next(residual_head.parameters(), None)
    dtype = residual_param.dtype if residual_param is not None else getattr(base_model, "runtime_dtype", torch.float32)
    device = residual_param.device if residual_param is not None else input_ids.device
    token_mask = token_mask.to(device=input_ids.device, dtype=torch.bool)
    code_map = base_model.token_id_to_code_index.to(device=input_ids.device)
    code_indices = torch.full_like(input_ids, -1)
    in_range = input_ids < code_map.numel()
    if bool(in_range.any()):
        code_indices[in_range] = code_map[input_ids[in_range]]
    valid_mask = token_mask & code_indices.ge(0)
    pooled = torch.zeros((input_ids.shape[0], hidden_size), device=device, dtype=dtype)
    if not bool(valid_mask.any()):
        return pooled

    projected = base_model.project_codebook_vectors(code_indices[valid_mask]).to(device=device, dtype=dtype)
    row_indices = valid_mask.nonzero(as_tuple=False)[:, 0].to(device=device)
    pooled.index_add_(0, row_indices, projected)
    counts = valid_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(device=device, dtype=dtype)
    return pooled / counts


def _scalar_value(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _prepend_text_soft_prompt_inputs(
    base_model: AnyMoRuntimeReplacementWrapper,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_soft_prompt = getattr(base_model, "text_soft_prompt", None)
    if text_soft_prompt is None or int(text_soft_prompt.shape[0]) <= 0:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return base_model.get_input_embeddings()(input_ids), attention_mask, token_mask

    input_embeds = base_model.get_input_embeddings()(input_ids)
    prompt = text_soft_prompt.to(device=input_embeds.device, dtype=input_embeds.dtype)
    prompt = prompt.unsqueeze(0).expand(input_embeds.shape[0], -1, -1)
    inputs_embeds = torch.cat([prompt, input_embeds], dim=1)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
    prompt_attention = torch.ones(
        (input_ids.shape[0], int(text_soft_prompt.shape[0])),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

    prompt_mask = torch.zeros(
        (input_ids.shape[0], int(text_soft_prompt.shape[0])),
        device=token_mask.device,
        dtype=torch.bool,
    )
    token_mask = torch.cat([prompt_mask, token_mask.to(dtype=torch.bool)], dim=1)
    return inputs_embeds, attention_mask, token_mask


def encode_contrastive_branch(
    model: AnyMoRuntimeReplacementWrapper,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_mask: torch.Tensor,
    branch: str,
    use_text_soft_prompt: bool = True,
) -> torch.Tensor:
    base_model = unwrap_distributed_model(model)
    if branch == "text" and use_text_soft_prompt and getattr(base_model, "text_soft_prompt", None) is not None:
        inputs_embeds, attention_mask, token_mask = _prepend_text_soft_prompt_inputs(
            base_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_mask=token_mask,
        )
        outputs = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_backbone_last_hidden_state=True,
            use_cache=False,
        )
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_backbone_last_hidden_state=True,
            use_cache=False,
        )
    hidden_states = outputs
    if branch == "imu":
        pooled = base_model.imu_pooler(hidden_states, token_mask)
        projected = base_model.imu_embedding_head(pooled)
        if getattr(base_model, "imu_residual_head", None) is not None:
            residual_pooled = _pool_projected_imu_token_embeddings(
                base_model,
                input_ids=input_ids,
                token_mask=token_mask,
            )
            residual = base_model.imu_residual_head(base_model.imu_residual_norm(residual_pooled))
            residual_scale = _scalar_value(getattr(base_model, "imu_residual_scale", None), 1.0)
            projected = projected + residual_scale * residual.to(device=projected.device, dtype=projected.dtype)
    elif branch == "text":
        pooled = base_model.text_pooler(hidden_states, token_mask)
        projected = base_model.text_embedding_head(pooled)
    else:
        raise ValueError(f"Unsupported contrastive branch: {branch}")
    return F.normalize(projected.float(), p=2, dim=-1)


def _gather_with_grad(features: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return features
    world_size = dist.get_world_size()
    if world_size == 1:
        return features
    gathered = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features)
    gathered[dist.get_rank()] = features
    return torch.cat(gathered, dim=0)


def _distributed_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def _model_uses_zero_partitioned_gradients(model: nn.Module) -> bool:
    zero_partitions_gradients = getattr(model, "zero_optimization_partition_gradients", None)
    if callable(zero_partitions_gradients):
        try:
            return bool(zero_partitions_gradients())
        except Exception:
            return True
    return False


def _model_allows_no_sync(model: nn.Module) -> bool:
    if not callable(getattr(model, "no_sync", None)):
        return False
    if _model_uses_zero_partitioned_gradients(model):
        return False
    return True


def _zero_touch_trainable_parameters(model: nn.Module) -> torch.Tensor:
    """Create a zero-valued graph edge to every trainable parameter.

    GradCache splits one logical training step into multiple DDP
    forward/backward pairs. The LM pair does not use contrastive pooler params,
    while the contrastive replay pairs do not use LM-only params. With
    ``find_unused_parameters=False`` DDP still expects every trainable
    parameter to be marked ready for each pair, so this zero term completes the
    reduction without changing gradients.
    """
    zero: torch.Tensor | None = None
    first_param: nn.Parameter | None = None
    base_model = unwrap_distributed_model(model)
    for param in base_model.parameters():
        if first_param is None:
            first_param = param
    if _model_uses_zero_partitioned_gradients(model):
        device = first_param.device if first_param is not None else torch.device("cpu")
        return torch.zeros((), device=device)
    for param in base_model.parameters():
        if not param.requires_grad:
            continue
        if param.numel() == 0:
            continue
        term = param.reshape(-1)[0] * 0.0
        zero = term if zero is None else zero + term
    if zero is None:
        device = first_param.device if first_param is not None else torch.device("cpu")
        zero = torch.zeros((), device=device)
    return zero


def symmetric_contrastive_loss(z_imu: torch.Tensor, z_text: torch.Tensor, temperature: float) -> torch.Tensor:
    z_imu = _gather_with_grad(F.normalize(z_imu.float(), p=2, dim=-1))
    z_text = _gather_with_grad(F.normalize(z_text.float(), p=2, dim=-1))
    logits = torch.matmul(z_imu, z_text.T) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _gather_activity_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    local_labels = [normalize_activity_label(label) for label in labels]
    if not dist.is_available() or not dist.is_initialized():
        return local_labels
    world_size = dist.get_world_size()
    if world_size == 1:
        return local_labels
    gathered: list[list[str] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_labels)
    all_labels: list[str] = []
    for rank_labels in gathered:
        if rank_labels:
            all_labels.extend(rank_labels)
    return all_labels


def _multi_positive_cross_entropy(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    valid_rows = positive_mask.any(dim=1)
    if not bool(valid_rows.any()):
        return logits.sum() * 0.0
    row_logits = logits[valid_rows]
    row_mask = positive_mask[valid_rows]
    log_denominator = torch.logsumexp(row_logits, dim=1)
    masked_logits = row_logits.masked_fill(~row_mask, torch.finfo(row_logits.dtype).min)
    log_numerator = torch.logsumexp(masked_logits, dim=1)
    return (log_denominator - log_numerator).mean()


def supervised_label_contrastive_loss(
    z_imu: torch.Tensor,
    z_label: torch.Tensor,
    activity_labels: list[str] | tuple[str, ...],
    temperature: float,
) -> torch.Tensor:
    z_imu = _gather_with_grad(F.normalize(z_imu.float(), p=2, dim=-1))
    z_label = _gather_with_grad(F.normalize(z_label.float(), p=2, dim=-1))
    gathered_labels = _gather_activity_labels(activity_labels)
    if len(gathered_labels) != z_imu.shape[0] or len(gathered_labels) != z_label.shape[0]:
        raise ValueError(
            "Number of gathered activity labels must match gathered embeddings: "
            f"labels={len(gathered_labels)}, imu={z_imu.shape[0]}, text={z_label.shape[0]}"
        )

    label_count = len(gathered_labels)
    device = z_imu.device
    positives = torch.zeros((label_count, label_count), dtype=torch.bool, device=device)
    for row_label in sorted(set(gathered_labels)):
        if not row_label:
            continue
        indices = [index for index, label in enumerate(gathered_labels) if label == row_label]
        if indices:
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            positives[index_tensor[:, None], index_tensor[None, :]] = True

    logits = torch.matmul(z_imu, z_label.T) / float(temperature)
    imu_to_label = _multi_positive_cross_entropy(logits, positives)
    label_to_imu = _multi_positive_cross_entropy(logits.T, positives.T)
    return 0.5 * (imu_to_label + label_to_imu)


def _contrastive_batch_size(batch: dict[str, Any]) -> int:
    return int(batch["imu_input_ids"].shape[0])


def _input_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for item in value.values():
            tensors.extend(_input_tensors(item))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_input_tensors(item))
        return tensors
    return []


class RandContext:
    """Record and restore RNG state for GradCache's graph-less/replay forwards."""

    def __init__(self, *tensors: torch.Tensor):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)
        self._fork = None

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def next_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return torch.zeros((), device=logits.device)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    preds = shift_logits.argmax(dim=-1)
    return preds.eq(shift_labels).masked_select(mask).float().mean()


class AnyMoContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args,
        contrastive_dataset: Dataset,
        contrastive_data_collator: ContrastiveDataCollator,
        contrastive_batch_size: int,
        lm_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 10.0,
        label_contrastive_loss_weight: float = 0.0,
        temperature: float = 0.05,
        grad_cache_steps: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.contrastive_dataset = contrastive_dataset
        self.contrastive_data_collator = contrastive_data_collator
        self.contrastive_batch_size = int(contrastive_batch_size)
        self.lm_loss_weight = float(lm_loss_weight)
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.label_contrastive_loss_weight = float(label_contrastive_loss_weight)
        self.temperature = float(temperature)
        self.grad_cache_steps = max(1, int(grad_cache_steps))
        self._contrastive_loader = None
        self._contrastive_iter = None
        self._last_aux_log_step = -1

    def get_contrastive_dataloader(self):
        if self._contrastive_loader is None:
            dataloader = DataLoader(
                self.contrastive_dataset,
                batch_size=self.contrastive_batch_size,
                shuffle=True,
                collate_fn=self.contrastive_data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            self._contrastive_loader = self.accelerator.prepare(dataloader)
        return self._contrastive_loader

    def _next_contrastive_batch(self) -> dict[str, Any]:
        if self._contrastive_iter is None:
            self._contrastive_iter = iter(self.get_contrastive_dataloader())
        try:
            return next(self._contrastive_iter)
        except StopIteration:
            self._contrastive_iter = iter(self.get_contrastive_dataloader())
            return next(self._contrastive_iter)

    def _next_prepared_contrastive_batch(self) -> dict[str, Any]:
        batch = self._prepare_inputs(self._next_contrastive_batch())
        batch.pop("positive_types", None)
        return batch

    def _encode_contrastive_pair(self, model, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        z_imu = encode_contrastive_branch(
            model,
            input_ids=batch["imu_input_ids"],
            attention_mask=batch["imu_attention_mask"],
            token_mask=batch["imu_token_mask"],
            branch="imu",
        )
        z_text = encode_contrastive_branch(
            model,
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            token_mask=batch["text_token_mask"],
            branch="text",
        )
        return z_imu, z_text

    def _encode_label_branch(self, model, batch: dict[str, Any]) -> torch.Tensor:
        return encode_contrastive_branch(
            model,
            input_ids=batch["label_input_ids"],
            attention_mask=batch["label_attention_mask"],
            token_mask=batch["label_token_mask"],
            branch="text",
        )

    def _compute_label_contrastive_loss(
        self,
        z_imu: torch.Tensor,
        z_label: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if getattr(self, "label_contrastive_loss_weight", 0.0) <= 0.0:
            return z_imu.sum() * 0.0
        return supervised_label_contrastive_loss(
            z_imu,
            z_label,
            list(batch["activity_labels"]),
            temperature=self.temperature,
        )

    def _scale_loss_for_backward(self, loss: torch.Tensor) -> torch.Tensor:
        if getattr(self.args, "gradient_accumulation_steps", 1) > 1:
            loss = loss / int(self.args.gradient_accumulation_steps)
        return loss

    def _should_run_grad_cache_contrastive(self) -> bool:
        if self.grad_cache_steps <= 1:
            return False
        return bool(getattr(self.accelerator, "sync_gradients", True))

    def _grad_cache_contrastive_backward(self, model) -> tuple[torch.Tensor, torch.Tensor]:
        cache_batches = [self._next_prepared_contrastive_batch() for _ in range(self.grad_cache_steps)]
        random_states: list[RandContext] = []
        cached_imu: list[torch.Tensor] = []
        cached_text: list[torch.Tensor] = []
        cached_label: list[torch.Tensor] = []
        cached_activity_labels: list[str] = []
        use_label_loss = getattr(self, "label_contrastive_loss_weight", 0.0) > 0.0
        with torch.no_grad():
            for batch in cache_batches:
                random_states.append(RandContext(*_input_tensors(batch)))
                with self.compute_loss_context_manager():
                    z_imu, z_text = self._encode_contrastive_pair(model, batch)
                    if use_label_loss:
                        z_label = self._encode_label_branch(model, batch)
                cached_imu.append(z_imu.detach())
                cached_text.append(z_text.detach())
                if use_label_loss:
                    cached_label.append(z_label.detach())
                    cached_activity_labels.extend(list(batch["activity_labels"]))

        z_imu_cache = torch.cat(cached_imu, dim=0).detach().requires_grad_(True)
        z_text_cache = torch.cat(cached_text, dim=0).detach().requires_grad_(True)
        grad_targets: list[torch.Tensor] = [z_imu_cache, z_text_cache]
        with torch.enable_grad():
            con_loss = symmetric_contrastive_loss(z_imu_cache, z_text_cache, temperature=self.temperature)
            weighted_contrastive_loss = self.contrastive_loss_weight * con_loss
            if use_label_loss:
                z_label_cache = torch.cat(cached_label, dim=0).detach().requires_grad_(True)
                grad_targets.append(z_label_cache)
                label_loss = supervised_label_contrastive_loss(
                    z_imu_cache,
                    z_label_cache,
                    cached_activity_labels,
                    temperature=self.temperature,
                )
                weighted_contrastive_loss = (
                    weighted_contrastive_loss + self.label_contrastive_loss_weight * label_loss
                )
            else:
                z_label_cache = None
                label_loss = z_imu_cache.sum() * 0.0
        grads = torch.autograd.grad(weighted_contrastive_loss, tuple(grad_targets))
        grad_imu = grads[0]
        grad_text = grads[1]
        grad_label = grads[2] if use_label_loss else None
        grad_imu = grad_imu.detach()
        grad_text = grad_text.detach()
        if grad_label is not None:
            grad_label = grad_label.detach()

        offset = 0
        sync_contexts: list[Any]
        if self.grad_cache_steps > 1 and _model_allows_no_sync(model):
            sync_contexts = [model.no_sync for _ in range(len(cache_batches) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in cache_batches]
        for batch, random_state, sync_context in zip(cache_batches, random_states, sync_contexts):
            batch_size = _contrastive_batch_size(batch)
            with sync_context():
                with random_state:
                    with self.compute_loss_context_manager():
                        z_imu, z_text = self._encode_contrastive_pair(model, batch)
                        if use_label_loss:
                            z_label = self._encode_label_branch(model, batch)
                imu_grad = grad_imu[offset : offset + batch_size].to(device=z_imu.device, dtype=z_imu.dtype)
                text_grad = grad_text[offset : offset + batch_size].to(device=z_text.device, dtype=z_text.dtype)
                surrogate = (z_imu * imu_grad).sum() + (z_text * text_grad).sum()
                if use_label_loss and grad_label is not None:
                    label_grad = grad_label[offset : offset + batch_size].to(device=z_label.device, dtype=z_label.dtype)
                    surrogate = surrogate + (z_label * label_grad).sum()
                surrogate = surrogate + _zero_touch_trainable_parameters(model)
                # GradCache contrastive is run only once on the sync micro-step.
                # The cached embedding gradients already include the narration
                # and label contrastive weights, so replay only backprops the
                # surrogate dot product.
                self.accelerator.backward(surrogate)
            offset += batch_size
        return con_loss.detach(), label_loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
        lm_loss = outputs.loss

        con_batch = self._next_prepared_contrastive_batch()
        z_imu, z_text = self._encode_contrastive_pair(model, con_batch)
        con_loss = symmetric_contrastive_loss(z_imu, z_text, temperature=self.temperature)
        if getattr(self, "label_contrastive_loss_weight", 0.0) > 0.0:
            z_label = self._encode_label_branch(model, con_batch)
            label_loss = self._compute_label_contrastive_loss(z_imu, z_label, con_batch)
        else:
            label_loss = con_loss.detach() * 0.0
        total_loss = (
            self.lm_loss_weight * lm_loss
            + self.contrastive_loss_weight * con_loss
            + self.label_contrastive_loss_weight * label_loss
        )

        if self.state.global_step != self._last_aux_log_step and self.state.global_step % max(1, self.args.logging_steps) == 0:
            self._last_aux_log_step = self.state.global_step
            with torch.no_grad():
                token_acc = next_token_accuracy(outputs.logits.detach(), inputs["labels"])
                weighted_aux = (
                    self.contrastive_loss_weight * con_loss.detach()
                    + self.label_contrastive_loss_weight * label_loss.detach()
                )
                ratio = weighted_aux / lm_loss.detach().clamp_min(1e-8)
            self.log(
                {
                    "lm_loss": float(lm_loss.detach().cpu()),
                    "contrastive_loss": float(con_loss.detach().cpu()),
                    "weighted_contrastive_loss": float((self.contrastive_loss_weight * con_loss.detach()).cpu()),
                    "label_contrastive_loss": float(label_loss.detach().cpu()),
                    "weighted_label_contrastive_loss": float(
                        (self.label_contrastive_loss_weight * label_loss.detach()).cpu()
                    ),
                    "contrastive_to_lm_loss_ratio": float(ratio.cpu()),
                    "token_acc": float(token_acc.cpu()),
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.grad_cache_steps <= 1:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        del num_items_in_batch
        model.train()
        inputs = self._prepare_inputs(inputs)
        run_contrastive = self._should_run_grad_cache_contrastive()

        lm_sync_context = model.no_sync() if run_contrastive and _model_allows_no_sync(model) else nullcontext()
        with lm_sync_context:
            with self.compute_loss_context_manager():
                outputs = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
                lm_loss = outputs.loss
            lm_backward_loss = self.lm_loss_weight * lm_loss + _zero_touch_trainable_parameters(model)
            self.accelerator.backward(self._scale_loss_for_backward(lm_backward_loss))

        if not run_contrastive:
            return self.lm_loss_weight * lm_loss.detach()

        con_loss, label_loss = self._grad_cache_contrastive_backward(model)
        total_loss = (
            self.lm_loss_weight * lm_loss.detach()
            + self.contrastive_loss_weight * con_loss.detach()
            + self.label_contrastive_loss_weight * label_loss.detach()
        )
        if self.state.global_step != self._last_aux_log_step and self.state.global_step % max(1, self.args.logging_steps) == 0:
            self._last_aux_log_step = self.state.global_step
            with torch.no_grad():
                token_acc = next_token_accuracy(outputs.logits.detach(), inputs["labels"])
                weighted_aux = (
                    self.contrastive_loss_weight * con_loss.detach()
                    + self.label_contrastive_loss_weight * label_loss.detach()
                )
                ratio = weighted_aux / lm_loss.detach().clamp_min(1e-8)
                local_effective_batch = self.contrastive_batch_size * self.grad_cache_steps
                global_effective_batch = local_effective_batch * _distributed_world_size()
            self.log(
                {
                    "lm_loss": float(lm_loss.detach().cpu()),
                    "contrastive_loss": float(con_loss.detach().cpu()),
                    "weighted_contrastive_loss": float((self.contrastive_loss_weight * con_loss.detach()).cpu()),
                    "label_contrastive_loss": float(label_loss.detach().cpu()),
                    "weighted_label_contrastive_loss": float(
                        (self.label_contrastive_loss_weight * label_loss.detach()).cpu()
                    ),
                    "contrastive_to_lm_loss_ratio": float(ratio.cpu()),
                    "token_acc": float(token_acc.cpu()),
                    "grad_cache_steps": int(self.grad_cache_steps),
                    "contrastive_local_effective_batch": int(local_effective_batch),
                    "contrastive_global_effective_batch": int(global_effective_batch),
                }
            )

        return total_loss

    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.accelerator.unwrap_model(self.model)
        torch.save(state_dict or model_to_save.state_dict(), Path(output_dir) / "pytorch_model.bin")
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def _checkpoint_state_path(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint
    for name in ("pytorch_model.bin", "model.safetensors"):
        candidate = checkpoint / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find pytorch_model.bin or model.safetensors in {checkpoint}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def _named_parameters_with_duplicates(model: nn.Module) -> dict[str, nn.Parameter]:
    try:
        return dict(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return dict(model.named_parameters())


def _is_zero3_partitioned_param(param: nn.Parameter) -> bool:
    return hasattr(param, "ds_id") or hasattr(param, "ds_status")


def _copy_checkpoint_tensor_(target: torch.Tensor, source: torch.Tensor, *, name: str) -> None:
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(
            f"Size mismatch for {name}: checkpoint has shape {tuple(source.shape)}, "
            f"target has shape {tuple(target.shape)}"
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _load_anymo_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    params = _named_parameters_with_duplicates(model)
    buffers = dict(model.named_buffers())
    target_keys = set(params) | set(buffers)
    unexpected: list[str] = []

    with torch.no_grad():
        for name, tensor in state_dict.items():
            if name in params:
                param = params[name]
                if _is_zero3_partitioned_param(param):
                    import deepspeed

                    with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                            _copy_checkpoint_tensor_(param.data, tensor, name=name)
                else:
                    _copy_checkpoint_tensor_(param.data, tensor, name=name)
            elif name in buffers:
                _copy_checkpoint_tensor_(buffers[name].data, tensor, name=name)
            else:
                unexpected.append(name)

    missing = sorted(target_keys - set(state_dict))
    return missing, unexpected


def _resolve_runtime_vocab_size(state_dict: dict[str, Any], tokenizer_vocab_size: int) -> int:
    vocab_size = int(tokenizer_vocab_size)
    for key in ("llm.model.embed_tokens.weight", "llm.model.language_model.embed_tokens.weight", "llm.lm_head.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and getattr(tensor, "shape", None):
            vocab_size = max(vocab_size, int(tensor.shape[0]))
    return vocab_size


def _resolve_mapping_vocab_size(state_dict: dict[str, Any], runtime_vocab_size: int) -> int:
    tensor = state_dict.get("token_id_to_code_index")
    if tensor is not None and getattr(tensor, "shape", None):
        return int(tensor.shape[0])
    return int(runtime_vocab_size)


def load_anymo_training_model(
    model_name_or_path: str,
    checkpoint: Path,
    codebook_artifact: Path,
    *,
    torch_dtype: torch.dtype | str,
    device_map: str | None,
    projection_hidden_size: int,
    attn_impl: str | None,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    codebook_payload = torch.load(codebook_artifact, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    special_tokens = [
        codebook_payload["special_tokens"]["bos"],
        codebook_payload["special_tokens"]["eos"],
        *codebook_payload["imu_code_tokens"],
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    state_dict = _load_state_dict(_checkpoint_state_path(checkpoint))
    runtime_vocab_size = _resolve_runtime_vocab_size(state_dict, tokenizer_vocab_size=len(tokenizer))
    mapping_vocab_size = _resolve_mapping_vocab_size(state_dict, runtime_vocab_size=runtime_vocab_size)

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": torch_dtype}
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if attn_impl:
        model_kwargs["attn_implementation"] = str(attn_impl)
    config_obj = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    with skip_transformers_cuda_allocator_warmup():
        llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    print("LLM attention debug summary:")
    print(json.dumps(build_attention_debug_summary(getattr(llm, "config", config_obj)), indent=2, default=str))
    frozen_modules = freeze_unused_vision_modules(llm)
    llm.resize_token_embeddings(runtime_vocab_size)

    imu_bos_token_id = tokenizer.convert_tokens_to_ids(codebook_payload["special_tokens"]["bos"])
    imu_eos_token_id = tokenizer.convert_tokens_to_ids(codebook_payload["special_tokens"]["eos"])
    imu_token_ids = tokenizer.convert_tokens_to_ids(codebook_payload["imu_code_tokens"])
    trainable_ids = [imu_bos_token_id, imu_eos_token_id, *imu_token_ids]

    model = AnyMoRuntimeReplacementWrapper(
        llm=llm,
        flat_codebook=codebook_payload["flat_codebook"],
        imu_token_ids=imu_token_ids,
        imu_bos_token_id=imu_bos_token_id,
        imu_eos_token_id=imu_eos_token_id,
        base_vocab_size=min(trainable_ids),
        tokenizer_vocab_size=mapping_vocab_size,
        projection_hidden_size=int(projection_hidden_size),
        init_anymo_lm_head_from_projector=False,
        freeze_non_anymo_token_rows=False,
    )
    missing, unexpected = _load_anymo_state_dict(model, state_dict)
    unexpected = [key for key in unexpected if not key.startswith("llm.model.visual.")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    model._align_runtime_state_to_llm()
    return tokenizer, model, {"missing_keys": missing, "unexpected_keys": unexpected, "frozen_modules": frozen_modules}


def build_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        **os.environ,
        "USE_HF": "1" if str(args.use_hf).lower() == "true" else "0",
        "HF_HOME": str(args.hf_home),
        "HUGGINGFACE_HUB_CACHE": str(args.hf_hub_cache),
        "MODELSCOPE_CACHE": str(args.modelscope_cache),
    }


def get_deepspeed_zero_stage(path: Path | None) -> int | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    zero_config = config.get("zero_optimization") or {}
    stage = zero_config.get("stage")
    return int(stage) if stage is not None else None


def build_deepspeed_zero_init_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    micro_batch = int(args.batch_size)
    grad_accum = int(args.gradient_accumulation_steps)
    # This copy is used only by HfDeepSpeedConfig before from_pretrained; the
    # zero.Init path constructs a DeepSpeedConfig before the full distributed
    # training engine exists and validates with world_size=1. The later
    # Trainer/Accelerate initialization still receives the original JSON path
    # with "auto" fields and resolves the real distributed batch itself.
    train_batch = micro_batch * grad_accum
    replacements = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": train_batch,
        "gradient_clipping": 0.0,
    }
    for key, value in replacements.items():
        if config.get(key) == "auto":
            config[key] = value
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    backbone = resolve_backbone_config(args)
    output_dir = Path(backbone["output_dir"])
    prewarm_before_load = env_flag("ANYMO_PREWARM_NCCL_BEFORE_LOAD", default=False)
    ddp_init_sync = env_flag("ANYMO_DDP_INIT_SYNC", default=False)
    ddp_gradient_as_bucket_view = env_flag("ANYMO_DDP_GRADIENT_AS_BUCKET_VIEW", default=True)
    deepspeed_path = Path(args.deepspeed) if args.deepspeed is not None else None
    deepspeed_zero_stage = get_deepspeed_zero_stage(deepspeed_path)
    use_zero3_load_init = deepspeed_zero_stage == 3

    print("AnyMo contrastive instruction tuning config:")
    print(
        json.dumps(
            {
                "model": str(backbone["model"]),
                "pretrained_checkpoint": str(backbone["pretrained_checkpoint"]),
                "lm_train_jsonl": str(args.lm_train_jsonl),
                "contrastive_train_jsonl": str(args.contrastive_train_jsonl),
                "codebook_artifact": str(args.codebook_artifact),
                "output_dir": str(output_dir),
                "deepspeed": str(args.deepspeed) if args.deepspeed is not None else None,
                "deepspeed_zero_stage": deepspeed_zero_stage,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "load_device_map": None
                if use_zero3_load_init
                else f"cuda:{os.environ.get('LOCAL_RANK')}"
                if os.environ.get("LOCAL_RANK") is not None
                else None,
                "accelerate_bypass_device_map": os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP"),
                "prewarm_nccl_before_load": prewarm_before_load,
                "ddp_init_sync": ddp_init_sync,
                "ddp_gradient_as_bucket_view": ddp_gradient_as_bucket_view,
                "skip_transformers_cuda_allocator_warmup": True,
                "pretokenize": bool(args.pretokenize),
                "lm_loss_weight": float(args.lm_loss_weight),
                "contrastive_loss_weight": float(args.contrastive_loss_weight),
                "label_contrastive_loss_weight": float(args.label_contrastive_loss_weight),
                "imu_residual_scale": float(args.imu_residual_scale),
                "text_soft_prompt_length": int(args.text_soft_prompt_length),
                "grad_cache_steps": int(args.grad_cache_steps),
                "contrastive_local_effective_batch": int(args.contrastive_batch_size or args.batch_size)
                * int(args.grad_cache_steps),
                "contrastive_global_effective_batch": int(args.contrastive_batch_size or args.batch_size)
                * int(args.grad_cache_steps)
                * int(os.environ.get("WORLD_SIZE", "1")),
                "pooler": {
                    "latents": int(args.pooler_latents),
                    "heads": 8,
                    "ff_mult": 2,
                    "pooling": "masked_mean",
                },
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if TrainingArguments is None:
        raise ModuleNotFoundError("transformers is required for training. Run this script inside the AnyMo environment.")

    os.environ.update(build_runtime_env(args))
    local_cuda_device = set_local_cuda_device_from_env()
    if local_cuda_device is not None:
        print(f"Set CUDA device from LOCAL_RANK: cuda:{local_cuda_device}")
    load_device_map = None if use_zero3_load_init else f"cuda:{local_cuda_device}" if local_cuda_device is not None else None
    log_cuda_memory("after_set_local_cuda_device")
    if prewarm_before_load or use_zero3_load_init:
        prewarm_nccl_process_group("zero3_before_model_load_process_group" if use_zero3_load_init else "prewarm_nccl_process_group")
    hf_deepspeed_config = None
    if use_zero3_load_init:
        from transformers.integrations import HfDeepSpeedConfig

        hf_deepspeed_config = HfDeepSpeedConfig(build_deepspeed_zero_init_config(deepspeed_path, args))
        print(
            json.dumps(
                {
                    "event": "hf_deepspeed_zero3_configured_before_model_load",
                    "rank": int(os.environ.get("RANK", "0")),
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "deepspeed": str(deepspeed_path),
                    "load_device_map": load_device_map,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    tokenizer, model, load_info = load_anymo_training_model(
        str(backbone["model"]),
        Path(backbone["pretrained_checkpoint"]),
        Path(args.codebook_artifact),
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device_map=load_device_map,
        projection_hidden_size=int(args.projection_hidden_size),
        attn_impl=args.attn_impl,
    )
    print(f"Loaded AnyMo checkpoint. Missing keys: {load_info['missing_keys'][:8]}")
    log_model_placement(model, "after_load_anymo_training_model")
    log_cuda_memory("after_load_anymo_training_model")
    if load_info.get("frozen_modules"):
        print(f"Frozen unused modules: {load_info['frozen_modules']}")
    attach_contrastive_modules(
        model,
        pooler_latents=int(args.pooler_latents),
        embedding_dim=args.embedding_dim,
        imu_residual_scale=float(args.imu_residual_scale),
        text_soft_prompt_length=int(args.text_soft_prompt_length),
    )
    log_model_placement(model, "after_attach_contrastive_modules")
    log_cuda_memory("after_attach_contrastive_modules")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lm_raw_dataset = JsonlDataset(Path(args.lm_train_jsonl))
    contrastive_raw_dataset = JsonlDataset(Path(args.contrastive_train_jsonl))
    if args.pretokenize:
        lm_dataset = MappedDataset(
            lm_raw_dataset,
            lambda example: tokenize_lm_example(example, tokenizer=tokenizer, max_length=int(args.max_length)),
            name="LM",
        )
        contrastive_dataset = MappedDataset(
            contrastive_raw_dataset,
            lambda example: tokenize_contrastive_example(
                example,
                tokenizer=tokenizer,
                imu_code_token_ids=model.imu_token_ids,
                max_imu_length=int(args.max_imu_length),
                max_text_length=int(args.max_text_length),
            ),
            name="contrastive",
        )
    else:
        lm_dataset = lm_raw_dataset
        contrastive_dataset = contrastive_raw_dataset
    lm_collator = LmDataCollator(tokenizer=tokenizer, max_length=int(args.max_length))
    contrastive_collator = ContrastiveDataCollator(
        tokenizer=tokenizer,
        imu_code_token_ids=model.imu_token_ids,
        max_imu_length=int(args.max_imu_length),
        max_text_length=int(args.max_text_length),
        seed=int(args.seed),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        logging_steps=int(args.logging_steps),
        save_strategy=str(args.save_strategy),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        remove_unused_columns=False,
        # The contrastive poolers/heads participate in the loss after the DDP
        # forward call. With unused-parameter detection on, DDP can mark those
        # params ready once as "unused" and then again during backward.
        ddp_find_unused_parameters=False,
        dataloader_num_workers=int(args.dataloader_num_workers),
        average_tokens_across_devices=False,
        deepspeed=str(args.deepspeed) if args.deepspeed is not None else None,
        report_to=[],
    )
    trainer = AnyMoContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=lm_collator,
        processing_class=tokenizer,
        contrastive_dataset=contrastive_dataset,
        contrastive_data_collator=contrastive_collator,
        contrastive_batch_size=int(args.contrastive_batch_size or args.batch_size),
        lm_loss_weight=float(args.lm_loss_weight),
        contrastive_loss_weight=float(args.contrastive_loss_weight),
        label_contrastive_loss_weight=float(args.label_contrastive_loss_weight),
        temperature=float(args.temperature),
        grad_cache_steps=int(args.grad_cache_steps),
    )
    log_model_placement(model, "before_trainer_train")
    log_cuda_memory("before_trainer_train")
    if args.deepspeed is None and (not ddp_init_sync or ddp_gradient_as_bucket_view):
        patch_ddp_init_settings(
            init_sync=ddp_init_sync,
            gradient_as_bucket_view=ddp_gradient_as_bucket_view,
        )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
