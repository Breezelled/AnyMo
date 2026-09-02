REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ANYMO_OUTPUT_ROOT="${ANYMO_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export ANYMO_CACHE_ROOT="${ANYMO_CACHE_ROOT:-${REPO_ROOT}/.cache}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

python "${REPO_ROOT}/code/pretrain_anymo_llm.py" \
  --backbone qwen2_5 --dataset-dir "${ANYMO_OUTPUT_ROOT}/motion_language_pretraining" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/motion_language_model" \
  --learning-rate 1e-4 --epochs 3 --batch-size 16 --max-length 1024 \
  --torch-dtype bfloat16 --attn-impl flash_attention_2 --save-total-limit 1
