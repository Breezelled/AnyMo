REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ANYMO_OUTPUT_ROOT="${ANYMO_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
: "${ANYMO_CHECKPOINT:?Set ANYMO_CHECKPOINT to the final instruction-tuned checkpoint}"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

CODEBOOK="${ANYMO_OUTPUT_ROOT}/instruction_tuning/imu_codebook_lookup.pt"
python "${REPO_ROOT}/code/evaluate_anymo_har_embedding.py" \
  --eval-root "${ANYMO_OUTPUT_ROOT}/har_evaluation" \
  --checkpoint "${ANYMO_CHECKPOINT}" --codebook-artifact "${CODEBOOK}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/results/har" \
  --label-prompt-mode learned --batch-size 128

python "${REPO_ROOT}/code/evaluate_anymo_heldout.py" --task retrieval --split subset \
  --input-dir "${ANYMO_OUTPUT_ROOT}/nymeria_heldout_evaluation" \
  --checkpoint "${ANYMO_CHECKPOINT}" --codebook-artifact "${CODEBOOK}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/results/nymeria_heldout/100_samples"
python "${REPO_ROOT}/code/evaluate_anymo_heldout.py" --task all --split full \
  --input-dir "${ANYMO_OUTPUT_ROOT}/nymeria_heldout_evaluation" \
  --checkpoint "${ANYMO_CHECKPOINT}" --codebook-artifact "${CODEBOOK}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/results/nymeria_heldout/all_samples"

python "${REPO_ROOT}/code/evaluate_anymo_heldout.py" --task retrieval --split subset \
  --input-dir "${ANYMO_OUTPUT_ROOT}/egoexo4d_zero_shot_evaluation" \
  --checkpoint "${ANYMO_CHECKPOINT}" --codebook-artifact "${CODEBOOK}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/results/egoexo4d_zero_shot/100_samples"
python "${REPO_ROOT}/code/evaluate_anymo_heldout.py" --task all --split full \
  --input-dir "${ANYMO_OUTPUT_ROOT}/egoexo4d_zero_shot_evaluation" \
  --checkpoint "${ANYMO_CHECKPOINT}" --codebook-artifact "${CODEBOOK}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/results/egoexo4d_zero_shot/all_samples"
