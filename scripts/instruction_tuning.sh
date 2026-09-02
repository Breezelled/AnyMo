REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ANYMO_OUTPUT_ROOT="${ANYMO_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export ANYMO_CACHE_ROOT="${ANYMO_CACHE_ROOT:-${REPO_ROOT}/.cache}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
: "${ANYMO_PRETRAINED_CHECKPOINT:?Set ANYMO_PRETRAINED_CHECKPOINT to the selected pretraining checkpoint}"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

CMD=(python "${REPO_ROOT}/code/contrastive_instruction_tuning_anymo_llm.py")
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CMD=(torchrun --nproc_per_node="${NPROC_PER_NODE}" "${REPO_ROOT}/code/contrastive_instruction_tuning_anymo_llm.py")
fi

"${CMD[@]}" --backbone qwen2_5 \
  --pretrained-checkpoint "${ANYMO_PRETRAINED_CHECKPOINT}" \
  --lm-train-jsonl "${ANYMO_OUTPUT_ROOT}/instruction_tuning/train.jsonl" \
  --contrastive-train-jsonl "${ANYMO_OUTPUT_ROOT}/instruction_tuning/train_contrastive.jsonl" \
  --codebook-artifact "${ANYMO_OUTPUT_ROOT}/instruction_tuning/imu_codebook_lookup.pt" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/anymo" \
  --epochs 1 --learning-rate 2e-5 --batch-size 16 --contrastive-batch-size 16 \
  --gradient-accumulation-steps 4 --grad-cache-steps 4 --max-length 1024 \
  --pooler-latents 128 --lm-loss-weight 1 --contrastive-loss-weight 2 \
  --label-contrastive-loss-weight 2 --temperature 0.1 --text-soft-prompt-length 8 \
  --imu-residual-scale 0.5 --torch-dtype bfloat16 --attn-impl flash_attention_2 \
  --dataloader-num-workers 4 --no-pretokenize --seed 42
