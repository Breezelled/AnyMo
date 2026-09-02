from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import config
import training_utils as utils
from anymo_swift_model import AnyMoRuntimeReplacementWrapper
from contrastive_instruction_tuning_anymo_llm import (
    _encode_text_prompt_with_content_mask,
    _pad_masks,
    _pad_sequences,
    _pool_projected_imu_token_embeddings,
    _scalar_value,
    attach_contrastive_modules,
    encode_contrastive_branch,
    format_imu_contrastive_prompt,
    freeze_unused_vision_modules,
    unwrap_distributed_model,
)
from evaluation_utils import (
    _attach_prediction,
    _checkpoint_state_path,
    _load_state_dict,
    compute_classification_metrics,
    dataset_relative_name,
    discover_eval_jsonls,
    load_jsonl,
    normalize_confusion_matrix,
    resolve_codebook_artifact_for_args,
    resolve_mapping_vocab_size,
    resolve_runtime_vocab_size,
    resolve_torch_dtype,
    summarize_metric_row,
    write_analysis_outputs,
    write_confusion_matrix_csv,
    write_jsonl,
    write_metrics_summary,
    write_predictions_csv,
)


DEFAULT_EVAL_ROOT = config.HAR_EVALUATION_DIR
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_CHECKPOINT = config.ANYMO_MODEL_DIR
DEFAULT_OUTPUT_DIR = config.OUTPUT_ROOT / "results" / "har"
LABEL_PROMPT_MODES = ("bare", "person", "imu", "activity", "ensemble", "score_ensemble")
LEARNED_LABEL_PROMPT_MODE = "learned"
ALL_LABEL_PROMPT_MODES = (*LABEL_PROMPT_MODES, LEARNED_LABEL_PROMPT_MODE)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AnyMo contrastive checkpoints by IMU-text embedding similarity on HAR MCQ JSONL files.",
    )
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codebook-artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-imu-length", type=int, default=2048)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument(
        "--label-prompt-mode",
        choices=(*LABEL_PROMPT_MODES, LEARNED_LABEL_PROMPT_MODE, "all"),
        default=LEARNED_LABEL_PROMPT_MODE,
        help="Use one label prompt template, ensemble templates, or all modes with separate output dirs.",
    )
    parser.add_argument("--projection-hidden-size", type=int, default=512)
    parser.add_argument(
        "--residual-score-fusion-weight",
        type=float,
        default=None,
        help=(
            "If set, score labels with score_llm_text + weight * score_residual_label. "
            "This keeps the LLM semantic IMU embedding separate from the motion-token residual path."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def infer_contrastive_module_config(state_dict: dict[str, Any]) -> dict[str, int | float]:
    queries = state_dict.get("imu_pooler.latents")
    if queries is None:
        queries = state_dict.get("imu_pooler.queries")
    head = state_dict.get("imu_embedding_head.weight")
    if queries is None or head is None:
        raise KeyError("Checkpoint is missing imu_pooler.latents/queries or imu_embedding_head.weight.")
    residual_scale = 0.0
    if "imu_residual_head.weight" in state_dict:
        scale_tensor = state_dict.get("imu_residual_scale")
        residual_scale = float(scale_tensor.detach().cpu().item()) if scale_tensor is not None else 1.0
    text_soft_prompt = state_dict.get("text_soft_prompt")
    text_soft_prompt_length = int(text_soft_prompt.shape[0]) if text_soft_prompt is not None else 0
    return {
        "pooler_latents": int(queries.shape[0]),
        "embedding_dim": int(head.shape[0]),
        "imu_residual_scale": float(residual_scale),
        "text_soft_prompt_length": int(text_soft_prompt_length),
    }


def align_contrastive_modules_to_runtime(model: AnyMoRuntimeReplacementWrapper) -> None:
    for module_name in (
        "imu_pooler",
        "text_pooler",
        "imu_embedding_head",
        "text_embedding_head",
        "imu_residual_norm",
        "imu_residual_head",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            module.to(device=model.device, dtype=model.runtime_dtype)
    text_soft_prompt = getattr(model, "text_soft_prompt", None)
    if text_soft_prompt is not None:
        with torch.no_grad():
            text_soft_prompt.data = text_soft_prompt.data.to(device=model.device, dtype=model.runtime_dtype)


def _resolve_llm_class(model_name_or_path: str, model_kwargs: dict[str, Any]):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


class LegacyV4LatentAttentionPooler(torch.nn.Module):
    """Pooler layout used by v4 checkpoints before RMSNorm/SwiGLU migration."""

    def __init__(self, hidden_size: int, num_latents: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_latents = int(num_latents)
        self.num_heads = 8
        self.ff_mult = 2
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size={self.hidden_size} must be divisible by {self.num_heads} attention heads.")
        self.latents = torch.nn.Parameter(torch.randn(self.num_latents, self.hidden_size) * 0.02)
        self.token_norm = torch.nn.LayerNorm(self.hidden_size)
        self.latent_norm = torch.nn.LayerNorm(self.hidden_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.ff_mult),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * self.ff_mult, self.hidden_size),
        )
        self.norm = torch.nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
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


def uses_legacy_v4_pooler_state_dict(state_dict: dict[str, Any]) -> bool:
    return any(
        key in state_dict
        for key in (
            "imu_pooler.ffn.0.weight",
            "text_pooler.ffn.0.weight",
            "imu_pooler.token_norm.bias",
            "text_pooler.token_norm.bias",
        )
    )


def _pooler_hidden_size_from_state_dict(state_dict: dict[str, Any]) -> int:
    latents = state_dict.get("imu_pooler.latents")
    if latents is None:
        latents = state_dict.get("imu_pooler.queries")
    if latents is None:
        raise KeyError("Checkpoint is missing imu_pooler.latents/queries.")
    return int(latents.shape[1])


def attach_checkpoint_compatible_contrastive_modules(
    model: AnyMoRuntimeReplacementWrapper,
    state_dict: dict[str, Any],
    *,
    pooler_latents: int,
    embedding_dim: int | None,
    imu_residual_scale: float = 1.0,
    text_soft_prompt_length: int = 0,
) -> str:
    if uses_legacy_v4_pooler_state_dict(state_dict):
        hidden_size = _pooler_hidden_size_from_state_dict(state_dict)
        out_dim = int(embedding_dim or hidden_size)
        model.imu_pooler = LegacyV4LatentAttentionPooler(hidden_size, num_latents=int(pooler_latents))
        model.text_pooler = LegacyV4LatentAttentionPooler(hidden_size, num_latents=int(pooler_latents))
        model.imu_embedding_head = torch.nn.Linear(hidden_size, out_dim, bias=False)
        model.text_embedding_head = torch.nn.Linear(hidden_size, out_dim, bias=False)
        if int(text_soft_prompt_length) > 0:
            model.text_soft_prompt = torch.nn.Parameter(torch.randn(int(text_soft_prompt_length), hidden_size) * 0.02)
        if float(imu_residual_scale) != 0.0:
            if "imu_residual_norm.bias" in state_dict or not hasattr(torch.nn, "RMSNorm"):
                model.imu_residual_norm = torch.nn.LayerNorm(hidden_size)
            else:
                model.imu_residual_norm = torch.nn.RMSNorm(hidden_size)
            model.imu_residual_head = torch.nn.Linear(hidden_size, out_dim, bias=False)
            model.register_buffer("imu_residual_scale", torch.tensor(float(imu_residual_scale), dtype=torch.float32))
        return "legacy_v4_layernorm_gelu"

    attach_contrastive_modules(
        model,
        pooler_latents=int(pooler_latents),
        embedding_dim=embedding_dim,
        imu_residual_scale=float(imu_residual_scale),
        text_soft_prompt_length=int(text_soft_prompt_length),
    )
    return "rmsnorm_swiglu"


def load_anymo_embedding_model(
    model_name_or_path: str,
    checkpoint: Path,
    codebook_artifact: Path,
    *,
    device: torch.device,
    torch_dtype: torch.dtype | str,
    projection_hidden_size: int,
):
    from transformers import AutoTokenizer

    codebook_payload = torch.load(codebook_artifact, map_location="cpu")
    checkpoint = Path(checkpoint)
    tokenizer_source = checkpoint if checkpoint.is_dir() and (checkpoint / "tokenizer_config.json").exists() else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    special_tokens = [
        codebook_payload["special_tokens"]["bos"],
        codebook_payload["special_tokens"]["eos"],
        *codebook_payload["imu_code_tokens"],
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    state_dict = _load_state_dict(_checkpoint_state_path(checkpoint))
    runtime_vocab_size = resolve_runtime_vocab_size(state_dict, tokenizer_vocab_size=len(tokenizer))
    mapping_vocab_size = resolve_mapping_vocab_size(state_dict, runtime_vocab_size=runtime_vocab_size)
    contrastive_config = infer_contrastive_module_config(state_dict)

    llm_cls = _resolve_llm_class(model_name_or_path, {"torch_dtype": torch_dtype})
    llm = llm_cls.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
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
    pooler_arch = attach_checkpoint_compatible_contrastive_modules(
        model,
        state_dict,
        pooler_latents=int(contrastive_config["pooler_latents"]),
        embedding_dim=int(contrastive_config["embedding_dim"]),
        imu_residual_scale=float(contrastive_config["imu_residual_scale"]),
        text_soft_prompt_length=int(contrastive_config["text_soft_prompt_length"]),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    model._align_runtime_state_to_llm()
    model.to(device)
    align_contrastive_modules_to_runtime(model)
    model.eval()
    return tokenizer, model, {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "frozen_modules": frozen_modules,
        "pooler_arch": pooler_arch,
        **contrastive_config,
    }


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    checkpoint_name = Path(args.checkpoint).name
    if args.eval_jsonl is None:
        return Path(args.eval_root) / f"embedding_eval_{checkpoint_name}"
    return Path(args.eval_jsonl).parent / f"embedding_eval_{checkpoint_name}"


def clone_args_for_label_prompt_mode(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args))
    cloned.label_prompt_mode = mode
    if str(args.label_prompt_mode) == "all":
        base_output_dir = resolve_output_dir(args)
        cloned.output_dir = base_output_dir.parent / f"{base_output_dir.name}_prompt_{mode}"
    return cloned


def extract_imu_token_text(row: dict[str, Any]) -> str:
    if row.get("imu_token_text"):
        return str(row["imu_token_text"])
    if row.get("imu_token_ids"):
        pieces = ["<imu_bos>"]
        pieces.extend(f"<imu_{int(token_id):04d}>" for token_id in row["imu_token_ids"])
        pieces.append("<imu_eos>")
        return "".join(pieces)
    content = "\n".join(str(message.get("content", "")) for message in row.get("messages", []))
    match = re.search(r"Input IMU token:\n(?P<tokens>.*?)(?:\n\nCHOICES:|\Z)", content, flags=re.S)
    if not match:
        raise KeyError("Could not resolve IMU token text from row.")
    return match.group("tokens").strip()


def extract_sensor_context(row: dict[str, Any]) -> str:
    if row.get("sensor_context"):
        return str(row["sensor_context"])
    if row.get("visible_segments"):
        return ", ".join(str(segment).replace("_", " ") for segment in row["visible_segments"])
    content = "\n".join(str(message.get("content", "")) for message in row.get("messages", []))
    match = re.search(r"attached to the user's (?P<context>.*?)\.\n\nInput IMU token:", content, flags=re.S)
    if match:
        return match.group("context").strip()
    return "visible body segments"


def _encode_imu_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    imu_code_token_ids: set[int],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_rows: list[list[int]] = []
    token_masks: list[list[bool]] = []
    for row in rows:
        prompt = format_imu_contrastive_prompt(
            imu_token_text=extract_imu_token_text(row),
            sensor_context=extract_sensor_context(row),
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)[: int(max_length)]
        input_rows.append(ids)
        token_masks.append([int(token_id) in imu_code_token_ids for token_id in ids])
    pad_token_id = int(tokenizer.pad_token_id)
    input_ids = _pad_sequences(input_rows, pad_token_id).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id).long(),
        "token_mask": _pad_masks(token_masks).to(device),
    }


def _encode_choice_text_rows(
    tokenizer: Any,
    choices: list[dict[str, Any]],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_rows: list[list[int]] = []
    token_masks: list[list[bool]] = []
    for choice in choices:
        ids, mask = _encode_text_prompt_with_content_mask(
            tokenizer,
            str(choice["label"]),
            max_length=int(max_length),
        )
        input_rows.append(ids)
        token_masks.append(mask)
    pad_token_id = int(tokenizer.pad_token_id)
    input_ids = _pad_sequences(input_rows, pad_token_id).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id).long(),
        "token_mask": _pad_masks(token_masks).to(device),
    }


def expand_label_prompts(label: str, *, mode: str) -> list[str]:
    label = str(label)
    if mode in {"bare", LEARNED_LABEL_PROMPT_MODE}:
        return [label]
    if mode == "person":
        return [f"a person is {label}"]
    if mode == "imu":
        return [f"wearable IMU motion of {label}"]
    if mode == "activity":
        return [f"the activity is {label}"]
    if mode in {"ensemble", "score_ensemble"}:
        return [
            label,
            f"a person is {label}",
            f"wearable IMU motion of {label}",
            f"the activity is {label}",
        ]
    raise ValueError(f"Unsupported label prompt mode: {mode}")


def uses_learned_text_soft_prompt(label_prompt_mode: str) -> bool:
    return str(label_prompt_mode) == LEARNED_LABEL_PROMPT_MODE


def uses_score_level_prompt_ensemble(label_prompt_mode: str) -> bool:
    return str(label_prompt_mode) == "score_ensemble"


def attach_embedding_prediction(
    row: dict[str, Any],
    scores: list[float],
    *,
    score_type: str = "cosine_similarity",
) -> dict[str, Any]:
    pred = _attach_prediction(row, scores)
    for choice in pred["choice_scores"]:
        choice["score_type"] = str(score_type)
    pred["scoring_method"] = (
        "embedding_cosine_similarity" if str(score_type) == "cosine_similarity" else str(score_type)
    )
    return pred


def collect_unique_choices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_by_id: dict[int, str] = {}
    for row in rows:
        for choice in row["choices"]:
            label_id = int(choice["label_id"])
            label_by_id.setdefault(label_id, str(choice["label"]))
    return [
        {"label_id": label_id, "label": label}
        for label_id, label in sorted(label_by_id.items())
    ]


def encode_text_embedding_cache(
    choices: list[dict[str, Any]],
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    *,
    device: torch.device,
    batch_size: int,
    max_text_length: int,
    label_prompt_mode: str,
    progress_desc: str | None = None,
) -> dict[int, torch.Tensor]:
    use_text_soft_prompt = uses_learned_text_soft_prompt(label_prompt_mode)
    if use_text_soft_prompt and getattr(unwrap_distributed_model(model), "text_soft_prompt", None) is None:
        raise ValueError("--label-prompt-mode learned requires a checkpoint with text_soft_prompt.")

    text_cache: dict[int, torch.Tensor] = {}
    prompt_rows: list[dict[str, Any]] = []
    for choice in choices:
        for prompt_text in expand_label_prompts(str(choice["label"]), mode=label_prompt_mode):
            prompt_rows.append(
                {
                    "label_id": int(choice["label_id"]),
                    "label": prompt_text,
                }
            )
    starts = range(0, len(prompt_rows), max(1, int(batch_size)))
    iterator = tqdm(starts, desc=progress_desc, unit="batch", leave=False) if progress_desc else starts
    grouped: dict[int, list[torch.Tensor]] = {}
    for start in iterator:
        batch_choices = prompt_rows[start : start + max(1, int(batch_size))]
        text_batch = _encode_choice_text_rows(
            tokenizer,
            batch_choices,
            max_length=int(max_text_length),
            device=device,
        )
        with torch.no_grad():
            z_text = encode_contrastive_branch(
                model,
                branch="text",
                use_text_soft_prompt=use_text_soft_prompt,
                **text_batch,
            )
        for choice, embedding in zip(batch_choices, z_text.detach().cpu()):
            grouped.setdefault(int(choice["label_id"]), []).append(embedding)
    for label_id, embeddings in grouped.items():
        stacked = torch.stack(embeddings, dim=0)
        if uses_score_level_prompt_ensemble(label_prompt_mode):
            text_cache[label_id] = torch.nn.functional.normalize(stacked, p=2, dim=-1)
            continue
        mean_embedding = torch.nn.functional.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
        text_cache[label_id] = mean_embedding
    return text_cache


def stack_choice_text_embeddings(
    row: dict[str, Any],
    text_cache: dict[int, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [text_cache[int(choice["label_id"])] for choice in row["choices"]],
        dim=0,
    ).to(device)


def compute_choice_scores(text_slice: torch.Tensor, z_imu: torch.Tensor) -> torch.Tensor:
    if text_slice.ndim == 2:
        return torch.matmul(text_slice, z_imu.unsqueeze(-1)).squeeze(-1)
    if text_slice.ndim == 3:
        return (text_slice * z_imu.view(1, 1, -1)).sum(dim=-1).mean(dim=1)
    raise ValueError(f"Expected text embeddings with 2 or 3 dimensions, got shape {tuple(text_slice.shape)}.")


def compute_residual_score_fusion_scores(
    text_slice: torch.Tensor,
    z_llm: torch.Tensor,
    z_residual: torch.Tensor,
    *,
    residual_weight: float,
) -> torch.Tensor:
    llm_scores = compute_choice_scores(text_slice, z_llm)
    residual_scores = compute_choice_scores(text_slice, z_residual)
    return llm_scores + float(residual_weight) * residual_scores


def encode_imu_embedding_components(
    model: AnyMoRuntimeReplacementWrapper,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_backbone_last_hidden_state=True,
        use_cache=False,
    )
    base_model = unwrap_distributed_model(model)
    pooled = base_model.imu_pooler(outputs, token_mask)
    projected = base_model.imu_embedding_head(pooled)
    z_llm = torch.nn.functional.normalize(projected.float(), p=2, dim=-1)

    residual_head = getattr(base_model, "imu_residual_head", None)
    if residual_head is None:
        return {"combined": z_llm, "llm": z_llm}

    residual_pooled = _pool_projected_imu_token_embeddings(
        base_model,
        input_ids=input_ids,
        token_mask=token_mask,
    )
    residual = residual_head(base_model.imu_residual_norm(residual_pooled))
    residual = residual.to(device=projected.device, dtype=projected.dtype)
    residual_scale = _scalar_value(getattr(base_model, "imu_residual_scale", None), 1.0)
    combined = torch.nn.functional.normalize((projected + residual_scale * residual).float(), p=2, dim=-1)
    z_residual = torch.nn.functional.normalize(residual.float(), p=2, dim=-1)
    return {"combined": combined, "llm": z_llm, "residual": z_residual}


def evaluate_embedding_records(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    *,
    device: torch.device,
    batch_size: int,
    max_imu_length: int,
    max_text_length: int,
    label_prompt_mode: str = "bare",
    residual_score_fusion_weight: float | None = None,
    progress_desc: str | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    imu_code_token_ids = set(int(token_id) for token_id in model.imu_token_ids)
    unique_choices = collect_unique_choices(rows)
    text_cache = encode_text_embedding_cache(
        unique_choices,
        tokenizer,
        model,
        device=device,
        batch_size=int(batch_size),
        max_text_length=int(max_text_length),
        label_prompt_mode=str(label_prompt_mode),
        progress_desc=f"{progress_desc}/text" if progress_desc else None,
    )
    starts = range(0, len(rows), int(batch_size))
    iterator = tqdm(starts, desc=progress_desc, unit="batch", leave=False) if progress_desc else starts
    for start in iterator:
        batch = rows[start : start + int(batch_size)]
        imu_batch = _encode_imu_rows(
            tokenizer,
            batch,
            imu_code_token_ids=imu_code_token_ids,
            max_length=int(max_imu_length),
            device=device,
        )
        with torch.no_grad():
            if residual_score_fusion_weight is None:
                z_imu = encode_contrastive_branch(model, branch="imu", **imu_batch)
                z_components = None
            else:
                z_components = encode_imu_embedding_components(model, **imu_batch)
                if "residual" not in z_components:
                    raise ValueError("--residual-score-fusion-weight requires a checkpoint with imu_residual_head.")
        for row_index, row in enumerate(batch):
            text_slice = stack_choice_text_embeddings(row, text_cache, device=device)
            if residual_score_fusion_weight is None:
                scores = compute_choice_scores(text_slice, z_imu[row_index])
            else:
                assert z_components is not None
                scores = compute_residual_score_fusion_scores(
                    text_slice,
                    z_components["llm"][row_index],
                    z_components["residual"][row_index],
                    residual_weight=float(residual_score_fusion_weight),
                )
            scores = scores.detach().cpu().tolist()
            score_type = "cosine_similarity" if residual_score_fusion_weight is None else "residual_score_fusion"
            predictions.append(
                attach_embedding_prediction(row, [float(score) for score in scores], score_type=score_type)
            )
    return predictions


def write_embedding_option_probabilities_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sample_index",
        "label_id",
        "label_text",
        "answer_key",
        "predicted_key",
        "correct",
        "choice_key",
        "choice_label_id",
        "choice_label",
        "choice_score",
        "choice_probability",
        "choice_rank",
        "score_type",
        "is_answer",
        "is_predicted",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for choice in row.get("choice_scores", []):
                writer.writerow(
                    {
                        "dataset": row.get("dataset"),
                        "sample_index": row.get("sample_index"),
                        "label_id": row.get("label_id"),
                        "label_text": row.get("label_text"),
                        "answer_key": row.get("answer_key"),
                        "predicted_key": row.get("predicted_key"),
                        "correct": row.get("correct"),
                        "choice_key": choice.get("key"),
                        "choice_label_id": choice.get("label_id"),
                        "choice_label": choice.get("label"),
                        "choice_score": choice.get("score"),
                        "choice_probability": choice.get("probability"),
                        "choice_rank": choice.get("rank"),
                        "score_type": choice.get("score_type"),
                        "is_answer": int(str(choice.get("key")) == str(row.get("answer_key"))),
                        "is_predicted": int(str(choice.get("key")) == str(row.get("predicted_key"))),
                    }
                )


def write_embedding_analysis_outputs(output_dir: Path, predictions: list[dict[str, Any]]) -> None:
    write_analysis_outputs(output_dir, predictions)
    write_embedding_option_probabilities_csv(output_dir / "embedding_choice_scores.csv", predictions)


def evaluate_one_jsonl(
    eval_jsonl: Path,
    output_dir: Path,
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    args: argparse.Namespace,
    load_info: dict[str, Any],
) -> dict[str, Any]:
    rows = load_jsonl(eval_jsonl, max_samples=args.max_samples)
    predictions = evaluate_embedding_records(
        rows,
        tokenizer,
        model,
        device=torch.device(args.device),
        batch_size=int(args.batch_size),
        max_imu_length=int(args.max_imu_length),
        max_text_length=int(args.max_text_length),
        label_prompt_mode=str(args.label_prompt_mode),
        residual_score_fusion_weight=args.residual_score_fusion_weight,
        progress_desc=f"{eval_jsonl.parent.parent.name}/{eval_jsonl.parent.name}",
    )
    metrics = compute_classification_metrics(predictions)
    if args.residual_score_fusion_weight is None:
        metrics["scoring_method"] = "embedding_cosine_similarity"
    else:
        metrics["scoring_method"] = "llm_text_plus_residual_label_score_fusion"
        metrics["residual_score_fusion_weight"] = float(args.residual_score_fusion_weight)
    metrics["label_prompt_mode"] = str(args.label_prompt_mode)
    metrics["load_info"] = {
        "missing_keys_count": len(load_info.get("missing_keys", [])),
        "unexpected_keys_count": len(load_info.get("unexpected_keys", [])),
        "pooler_latents": load_info.get("pooler_latents"),
            "embedding_dim": load_info.get("embedding_dim"),
            "text_soft_prompt_length": load_info.get("text_soft_prompt_length", 0),
            "frozen_modules": load_info.get("frozen_modules", []),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_predictions_csv(output_dir / "predictions.csv", predictions)
    write_embedding_analysis_outputs(output_dir, predictions)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_all_datasets(
    args: argparse.Namespace,
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    load_info: dict[str, Any],
) -> list[dict[str, Any]]:
    eval_root = Path(args.eval_root)
    output_root = resolve_output_dir(args)
    rows: list[dict[str, Any]] = []
    for eval_jsonl in tqdm(discover_eval_jsonls(eval_root), desc="datasets", unit="dataset"):
        dataset_name = dataset_relative_name(eval_root, eval_jsonl)
        dataset_output_dir = output_root / dataset_name
        metrics_path = dataset_output_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[skip] {dataset_name}", flush=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"[eval] {dataset_name}", flush=True)
            metrics = evaluate_one_jsonl(eval_jsonl, dataset_output_dir, tokenizer, model, args, load_info)
        rows.append(summarize_metric_row(dataset_name, metrics))
        write_metrics_summary(output_root, rows)
    return rows


def run_evaluation(
    args: argparse.Namespace,
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    load_info: dict[str, Any],
) -> list[dict[str, Any]] | dict[str, Any]:
    if args.eval_jsonl is None:
        return evaluate_all_datasets(args, tokenizer, model, load_info)
    return evaluate_one_jsonl(
        Path(args.eval_jsonl),
        resolve_output_dir(args),
        tokenizer,
        model,
        args,
        load_info,
    )


def main() -> int:
    args = parse_args()
    utils.set_seed(int(args.seed))
    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    if args.model is None or args.checkpoint is None:
        raise ValueError("--model and --checkpoint are required.")
    device = torch.device(args.device)
    tokenizer, model, load_info = load_anymo_embedding_model(
        args.model,
        args.checkpoint,
        resolve_codebook_artifact_for_args(args),
        device=device,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        projection_hidden_size=int(args.projection_hidden_size),
    )
    if str(args.label_prompt_mode) == "all":
        all_results: dict[str, list[dict[str, Any]] | dict[str, Any]] = {}
        for mode in ALL_LABEL_PROMPT_MODES:
            mode_args = clone_args_for_label_prompt_mode(args, mode)
            print(
                f"[mode] label_prompt_mode={mode} output_dir={resolve_output_dir(mode_args)}",
                flush=True,
            )
            all_results[mode] = run_evaluation(mode_args, tokenizer, model, load_info)
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
        return 0
    result = run_evaluation(args, tokenizer, model, load_info)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
