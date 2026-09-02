from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

import config
from anymo_swift_model import AnyMoRuntimeReplacementWrapper
from contrastive_instruction_tuning_anymo_llm import (
    _encode_text_prompt_with_content_mask,
    _pad_masks,
    _pad_sequences,
    format_qwen_plain_lm,
)
from evaluate_anymo_har_embedding import (
    _encode_imu_rows,
    load_anymo_embedding_model,
)
from evaluation_utils import (
    resolve_torch_dtype,
)


DEFAULT_INPUT_DIR = config.NYMERIA_HELDOUT_DIR
DEFAULT_CHECKPOINT = config.ANYMO_MODEL_DIR
DEFAULT_OUTPUT_DIR = config.OUTPUT_ROOT / "results" / "nymeria_heldout"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
PRIMARY_REFERENCE_MODE = "primary_ground_truth_only"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AnyMo held-out Nymeria retrieval and captioning.",
    )
    parser.add_argument("--task", choices=("retrieval", "captioning", "captioning_score", "all"), default="all")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codebook-artifact", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=("auto", "subset", "full"),
        default="auto",
        help="auto/subset uses *_100_seed42.jsonl when present; full uses the complete JSONL.",
    )
    parser.add_argument("--retrieval-jsonl", type=Path, default=None)
    parser.add_argument("--captioning-jsonl", type=Path, default=None)
    parser.add_argument("--caption-predictions", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-imu-length", type=int, default=2048)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--projection-hidden-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--bertscore-model", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--bertscore-max-length", type=int, default=512)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_eval_jsonl(input_dir: Path, task: str, split: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.is_absolute() else Path(input_dir) / path
    task = str(task)
    if task not in {"retrieval", "captioning"}:
        raise ValueError(f"Unsupported task for JSONL resolution: {task}")
    input_dir = Path(input_dir)
    if str(split) == "full":
        return input_dir / f"{task}.jsonl"
    subset = input_dir / f"{task}_100_seed42.jsonl"
    if str(split) in {"auto", "subset"} and subset.exists():
        return subset
    if str(split) == "subset":
        raise FileNotFoundError(f"Requested subset split but {subset} does not exist.")
    return input_dir / f"{task}.jsonl"


def primary_reference_text(row: dict[str, Any]) -> str:
    for key in ("positive_texts", "references"):
        for item in row.get(key, []) or []:
            if str(item.get("type", "")) == "narration" and str(item.get("text", "")).strip():
                return str(item["text"])
        for item in row.get(key, []) or []:
            if str(item.get("text", "")).strip():
                return str(item["text"])
    raise ValueError("Row has no non-empty primary reference text.")


def metadata_without_large_fields(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {"messages", "positive_texts", "references", "imu_prompt", "imu_token_text", "imu_token_ids"}
    return {key: value for key, value in row.items() if key not in excluded}


def query_id_from_row(row: dict[str, Any], fallback_index: int) -> str:
    if str(row.get("query_id", "")).strip():
        return str(row["query_id"])
    if str(row.get("sample_dir", "")).strip() or row.get("source_row_index") is not None:
        return str(row.get("sample_dir", "")) + f":{row.get('source_row_index', fallback_index)}"
    return str(int(fallback_index))


def _rank_from_scores(scores: torch.Tensor, positives: set[int]) -> int:
    if not positives:
        raise ValueError("positives must be non-empty.")
    order = torch.argsort(scores, descending=True).detach().cpu().tolist()
    for rank, index in enumerate(order, start=1):
        if int(index) in positives:
            return rank
    return len(order) + 1


def retrieval_metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"num_queries": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    arr = np.asarray(ranks, dtype=np.float32)
    return {
        "num_queries": int(len(ranks)),
        "recall_at_1": float(np.mean(arr <= 1)),
        "recall_at_5": float(np.mean(arr <= 5)),
        "recall_at_10": float(np.mean(arr <= 10)),
        "mrr": float(np.mean(1.0 / arr)),
    }


def compute_bidirectional_retrieval_metrics(
    similarity: torch.Tensor,
    imu_positive_texts: list[set[int]],
    text_positive_imus: list[set[int]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    del top_k
    sensor_ranks = [
        _rank_from_scores(similarity[imu_index], positives)
        for imu_index, positives in enumerate(imu_positive_texts)
        if positives
    ]
    text_ranks = [
        _rank_from_scores(similarity[:, text_index], positives)
        for text_index, positives in enumerate(text_positive_imus)
        if positives
    ]
    return {
        "sensor_to_text": retrieval_metrics_from_ranks(sensor_ranks),
        "text_to_sensor": retrieval_metrics_from_ranks(text_ranks),
    }


def _encode_text_rows(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_rows: list[list[int]] = []
    token_masks: list[list[bool]] = []
    for text in texts:
        ids, mask = _encode_text_prompt_with_content_mask(tokenizer, text, max_length=int(max_length))
        input_rows.append(ids)
        token_masks.append(mask)
    pad_token_id = int(tokenizer.pad_token_id)
    input_ids = _pad_sequences(input_rows, pad_token_id).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id).long(),
        "token_mask": _pad_masks(token_masks).to(device),
    }


def encode_imu_embeddings(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    *,
    device: torch.device,
    batch_size: int,
    max_imu_length: int,
) -> torch.Tensor:
    from contrastive_instruction_tuning_anymo_llm import encode_contrastive_branch

    embeddings: list[torch.Tensor] = []
    imu_code_token_ids = set(int(token_id) for token_id in model.imu_token_ids)
    for start in tqdm(range(0, len(rows), int(batch_size)), desc="encode IMU", unit="batch", leave=False):
        batch = rows[start : start + int(batch_size)]
        encoded = _encode_imu_rows(
            tokenizer,
            batch,
            imu_code_token_ids=imu_code_token_ids,
            max_length=int(max_imu_length),
            device=device,
        )
        with torch.no_grad():
            z = encode_contrastive_branch(model, branch="imu", **encoded)
        embeddings.append(z.detach().cpu())
    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(embeddings, dim=0).float()


def encode_text_embeddings(
    texts: list[str],
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    *,
    device: torch.device,
    batch_size: int,
    max_text_length: int,
) -> torch.Tensor:
    from contrastive_instruction_tuning_anymo_llm import encode_contrastive_branch, unwrap_distributed_model

    base_model = unwrap_distributed_model(model)
    use_text_soft_prompt = getattr(base_model, "text_soft_prompt", None) is not None
    embeddings: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), int(batch_size)), desc="encode text", unit="batch", leave=False):
        batch_texts = texts[start : start + int(batch_size)]
        encoded = _encode_text_rows(
            tokenizer,
            batch_texts,
            max_length=int(max_text_length),
            device=device,
        )
        with torch.no_grad():
            z = encode_contrastive_branch(
                model,
                branch="text",
                use_text_soft_prompt=use_text_soft_prompt,
                **encoded,
            )
        embeddings.append(z.detach().cpu())
    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(embeddings, dim=0).float()


def build_retrieval_pool(rows: list[dict[str, Any]]) -> tuple[list[str], list[set[int]], list[set[int]], list[dict[str, Any]]]:
    text_keys: dict[str, int] = {}
    texts: list[str] = []
    text_positive_imus: defaultdict[int, set[int]] = defaultdict(set)
    imu_positive_texts: list[set[int]] = []
    text_records: list[dict[str, Any]] = []

    for imu_index, row in enumerate(rows):
        text = primary_reference_text(row)
        text_index = text_keys.get(text)
        if text_index is None:
            text_index = len(texts)
            text_keys[text] = text_index
            texts.append(text)
            text_records.append(
                {
                    "text_id": f"T{text_index:04d}",
                    "text": text,
                    "reference_mode": PRIMARY_REFERENCE_MODE,
                    "source_query_id": str(row.get("query_id", "")),
                }
            )
        imu_positive_texts.append({text_index})
        text_positive_imus[text_index].add(imu_index)

    text_positive_sets = [text_positive_imus[index] for index in range(len(texts))]
    return texts, imu_positive_texts, text_positive_sets, text_records


def top_indices(scores: torch.Tensor, k: int) -> list[int]:
    if scores.numel() == 0:
        return []
    k = min(int(k), int(scores.numel()))
    return [int(index) for index in torch.topk(scores, k=k, largest=True).indices.detach().cpu().tolist()]


def run_retrieval(
    args: argparse.Namespace,
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    load_info: dict[str, Any],
) -> dict[str, Any]:
    retrieval_jsonl = resolve_eval_jsonl(args.input_dir, "retrieval", args.split, args.retrieval_jsonl)
    rows = read_jsonl(retrieval_jsonl, limit=args.max_samples)
    if not rows:
        raise ValueError(f"No retrieval rows found in {retrieval_jsonl}")

    device = torch.device(args.device)
    texts, imu_positive_texts, text_positive_imus, text_records = build_retrieval_pool(rows)
    z_imu = encode_imu_embeddings(
        rows,
        tokenizer,
        model,
        device=device,
        batch_size=int(args.batch_size),
        max_imu_length=int(args.max_imu_length),
    )
    z_text = encode_text_embeddings(
        texts,
        tokenizer,
        model,
        device=device,
        batch_size=int(args.batch_size),
        max_text_length=int(args.max_text_length),
    )
    similarity = torch.matmul(z_imu, z_text.T)
    metrics = compute_bidirectional_retrieval_metrics(similarity, imu_positive_texts, text_positive_imus)

    output_dir = Path(args.output_dir) / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    for imu_index, row in enumerate(rows):
        top_text = top_indices(similarity[imu_index], 10)
        predictions.append(
            {
                "query_id": str(row.get("query_id", imu_index)),
                "direction": "sensor_to_text",
                "rank": _rank_from_scores(similarity[imu_index], imu_positive_texts[imu_index]),
                "positive_text_ids": [f"T{idx:04d}" for idx in sorted(imu_positive_texts[imu_index])],
                "top10_text_ids": [f"T{idx:04d}" for idx in top_text],
                "top10_scores": [float(similarity[imu_index, idx].item()) for idx in top_text],
                "metadata": metadata_without_large_fields(row),
            }
        )
    for text_index, positives in enumerate(text_positive_imus):
        top_imu = top_indices(similarity[:, text_index], 10)
        predictions.append(
            {
                "text_id": f"T{text_index:04d}",
                "direction": "text_to_sensor",
                "rank": _rank_from_scores(similarity[:, text_index], positives) if positives else None,
                "positive_query_ids": [str(rows[idx].get("query_id", idx)) for idx in sorted(positives)],
                "top10_query_ids": [str(rows[idx].get("query_id", idx)) for idx in top_imu],
                "top10_scores": [float(similarity[idx, text_index].item()) for idx in top_imu],
                "text": texts[text_index],
            }
        )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "text_candidates.jsonl", text_records)

    summary = {
        "task": "retrieval",
        "retrieval_protocol": "embedding_similarity_primary_gt_text",
        "retrieval_jsonl": str(retrieval_jsonl),
        "reference_mode": PRIMARY_REFERENCE_MODE,
        "num_sensor_queries": len(rows),
        "num_text_candidates": len(texts),
        "sensor_to_text": metrics["sensor_to_text"],
        "text_to_sensor": metrics["text_to_sensor"],
        "load_info": {
            "pooler_latents": load_info.get("pooler_latents"),
            "embedding_dim": load_info.get("embedding_dim"),
            "text_soft_prompt_length": load_info.get("text_soft_prompt_length", 0),
            "pooler_arch": load_info.get("pooler_arch"),
        },
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def caption_prompt_from_row(row: dict[str, Any]) -> str:
    messages = list(row.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        prompt, _ = format_qwen_plain_lm(messages)
        return prompt
    if messages:
        prompt_messages = [dict(message) for message in messages]
        prompt_messages.append({"role": "assistant", "content": ""})
        prompt, _ = format_qwen_plain_lm(prompt_messages)
        return prompt
    raise ValueError("Caption row is missing messages.")


def clean_generated_caption(text: str) -> str:
    text = str(text)
    text = text.split("<|im_end|>", 1)[0]
    text = text.split("<|endoftext|>", 1)[0]
    return text.strip().strip('"')


def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    if float(temperature) <= 0.0:
        return torch.argmax(logits, dim=-1)
    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        return sorted_indices.gather(-1, next_sorted.unsqueeze(-1)).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _left_pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long)
    for row_index, seq in enumerate(sequences):
        if seq:
            out[row_index, -len(seq) :] = torch.tensor(seq, dtype=torch.long)
    return out


def generate_caption_batch(
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    prompts: list[str],
    *,
    device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    pad_token_id = int(tokenizer.pad_token_id)
    eos_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    input_rows = [tokenizer.encode(prompt, add_special_tokens=False)[-int(max_prompt_length) :] for prompt in prompts]
    input_ids = _left_pad_sequences(input_rows, pad_token_id).to(device)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in prompts]

    for _ in range(int(max_new_tokens)):
        attention_mask = input_ids.ne(pad_token_id).long()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]
        next_tokens = _sample_next_token(logits, temperature=float(temperature), top_p=float(top_p))
        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
        for row_index, token_id in enumerate(next_tokens.detach().cpu().tolist()):
            if not bool(finished[row_index]) and int(token_id) != pad_token_id:
                generated[row_index].append(int(token_id))
        finished = finished | torch.tensor(
            [int(token_id) in eos_token_ids for token_id in next_tokens.detach().cpu().tolist()],
            device=device,
            dtype=torch.bool,
        )
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
        if bool(finished.all()):
            break

    return [clean_generated_caption(tokenizer.decode(ids, skip_special_tokens=False)) for ids in generated]


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def sentence_bleu(candidate: str, reference: str, max_n: int) -> float:
    cand = tokenize_text(candidate)
    ref = tokenize_text(reference)
    if not cand or not ref:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_counts = ngram_counts(cand, n)
        ref_counts = ngram_counts(ref, n)
        total = sum(cand_counts.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
        precisions.append(max(overlap / total, 1e-9))
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(cand)))
    return float(brevity * math.exp(sum(math.log(p) for p in precisions) / max_n))


def rouge_l(candidate: str, reference: str) -> float:
    cand = tokenize_text(candidate)
    ref = tokenize_text(reference)
    if not cand or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for token in cand:
        curr = [0]
        for j, ref_token in enumerate(ref, start=1):
            curr.append(prev[j - 1] + 1 if token == ref_token else max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return float((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)


def meteor_like(candidate: str, reference: str) -> float:
    cand = tokenize_text(candidate)
    ref = tokenize_text(reference)
    if not cand or not ref:
        return 0.0
    ref_counts = Counter(ref)
    matches = 0
    for token in cand:
        if ref_counts[token] > 0:
            matches += 1
            ref_counts[token] -= 1
    if matches == 0:
        return 0.0
    precision = matches / len(cand)
    recall = matches / len(ref)
    return float((10 * precision * recall) / (recall + 9 * precision)) if recall + 9 * precision else 0.0


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str,
    max_length: int,
) -> dict[str, float] | None:
    compute_bertscore.last_error = None
    try:
        from bert_score import BERTScorer
    except Exception as exc:
        compute_bertscore.last_error = repr(exc)
        return None
    try:
        scorer = BERTScorer(
            model_type=model_type,
            lang="en",
            rescale_with_baseline=False,
            use_fast_tokenizer=False,
        )
        if max_length > 0 and hasattr(scorer, "_tokenizer"):
            scorer._tokenizer.model_max_length = int(max_length)
        precision, recall, f1 = scorer.score(predictions, references)
    except Exception as exc:
        compute_bertscore.last_error = repr(exc)
        return None
    return {
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def score_caption_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["references"][0]) for row in rows]
    metrics: dict[str, Any] = {
        "task": "captioning",
        "num_samples": len(rows),
        "bleu_1": float(np.mean([sentence_bleu(p, r, 1) for p, r in zip(predictions, references)])) if rows else 0.0,
        "bleu_4": float(np.mean([sentence_bleu(p, r, 4) for p, r in zip(predictions, references)])) if rows else 0.0,
        "rouge_l": float(np.mean([rouge_l(p, r) for p, r in zip(predictions, references)])) if rows else 0.0,
        "meteor": float(np.mean([meteor_like(p, r) for p, r in zip(predictions, references)])) if rows else 0.0,
        "reference_mode": PRIMARY_REFERENCE_MODE,
        "meteor_note": "Exact-token METEOR-style fallback unless external METEOR package is added.",
    }
    if not args.skip_bertscore:
        bert = compute_bertscore(predictions, references, str(args.bertscore_model), int(args.bertscore_max_length))
        if bert is None:
            metrics["bertscore_available"] = False
            metrics["bertscore_error"] = getattr(compute_bertscore, "last_error", None)
        else:
            metrics["bertscore_available"] = True
            metrics.update(bert)
    else:
        metrics["bertscore_available"] = False
        metrics["bertscore_note"] = "Skipped by --skip-bertscore."
    return metrics


def run_captioning(
    args: argparse.Namespace,
    tokenizer: Any,
    model: AnyMoRuntimeReplacementWrapper,
    load_info: dict[str, Any],
) -> dict[str, Any]:
    captioning_jsonl = resolve_eval_jsonl(args.input_dir, "captioning", args.split, args.captioning_jsonl)
    rows = read_jsonl(captioning_jsonl, limit=args.max_samples)
    if not rows:
        raise ValueError(f"No captioning rows found in {captioning_jsonl}")

    output_dir = Path(args.output_dir) / "captioning"
    predictions_path = output_dir / "predictions.jsonl"
    if args.skip_existing and predictions_path.exists():
        prediction_rows = read_jsonl(predictions_path, limit=args.max_samples)
    else:
        prediction_rows: list[dict[str, Any]] = []
        device = torch.device(args.device)
        for start in tqdm(range(0, len(rows), int(args.batch_size)), desc="caption", unit="batch"):
            batch = rows[start : start + int(args.batch_size)]
            prompts = [caption_prompt_from_row(row) for row in batch]
            captions = generate_caption_batch(
                tokenizer,
                model,
                prompts,
                device=device,
                max_prompt_length=int(args.max_imu_length),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            for row, caption in zip(batch, captions):
                prediction_rows.append(
                    {
                        "query_id": query_id_from_row(row, len(prediction_rows)),
                        "prediction": caption,
                        "references": [primary_reference_text(row)],
                        "metadata": metadata_without_large_fields(row),
                    }
                )
        write_jsonl(predictions_path, prediction_rows)

    metrics = score_caption_rows(prediction_rows, args)
    metrics.update(
        {
            "captioning_jsonl": str(captioning_jsonl),
            "load_info": {
                "pooler_latents": load_info.get("pooler_latents"),
                "embedding_dim": load_info.get("embedding_dim"),
                "text_soft_prompt_length": load_info.get("text_soft_prompt_length", 0),
                "pooler_arch": load_info.get("pooler_arch"),
            },
        }
    )
    save_json(output_dir / "summary.json", metrics)
    return metrics


def score_existing_captioning(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = (
        Path(args.caption_predictions)
        if args.caption_predictions is not None
        else Path(args.output_dir) / "captioning" / "predictions.jsonl"
    )
    rows = read_jsonl(predictions_path, limit=args.max_samples)
    if not rows:
        raise ValueError(f"No caption predictions found in {predictions_path}")
    output_dir = Path(args.output_dir) / "captioning"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = score_caption_rows(rows, args)
    save_json(output_dir / "summary.json", metrics)
    return metrics


def resolve_codebook_artifact_for_heldout_args(args: argparse.Namespace) -> Path:
    if args.codebook_artifact is not None:
        return Path(args.codebook_artifact)
    candidate = Path(args.input_dir) / "imu_codebook_lookup.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find imu_codebook_lookup.pt in {args.input_dir}; pass --codebook-artifact.")


def load_model_for_eval(args: argparse.Namespace):
    device = torch.device(args.device)
    return load_anymo_embedding_model(
        args.model,
        args.checkpoint,
        resolve_codebook_artifact_for_heldout_args(args),
        device=device,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        projection_hidden_size=int(args.projection_hidden_size),
    )


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.task == "captioning_score":
        summaries = {"captioning": score_existing_captioning(args)}
        save_json(Path(args.output_dir) / "summary.json", summaries)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0

    tokenizer, model, load_info = load_model_for_eval(args)
    summaries: dict[str, Any] = {}
    if args.task in {"retrieval", "all"}:
        summaries["retrieval"] = run_retrieval(args, tokenizer, model, load_info)
    if args.task in {"captioning", "all"}:
        summaries["captioning"] = run_captioning(args, tokenizer, model, load_info)
    save_json(Path(args.output_dir) / "summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
