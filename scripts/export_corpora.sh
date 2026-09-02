REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ANYMO_DATA_ROOT:?Set ANYMO_DATA_ROOT to the simulated Nymeria root}"
: "${HAR_DATA_ROOT:?Set HAR_DATA_ROOT for the preprocessed HAR datasets}"
: "${EGO4D_ROOT:?Set EGO4D_ROOT}"
: "${MMEA_ROOT:?Set MMEA_ROOT}"
: "${EGOEXO4D_ROOT:?Set EGOEXO4D_ROOT}"
: "${OPENPACK_ROOT:?Set OPENPACK_ROOT}"
export ANYMO_OUTPUT_ROOT="${ANYMO_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

SUMMARY="${ANYMO_DATA_ROOT}/geometry_aware_imu_summary.csv"
ENCODER="${ANYMO_OUTPUT_ROOT}/encoder/encoder_best.pt"
TOKENIZER="${ANYMO_OUTPUT_ROOT}/tokenizer/tokenizer_export.pt"

python "${REPO_ROOT}/code/export.py" tokens \
  --base-dir "${ANYMO_DATA_ROOT}" --summary-csv "${SUMMARY}" \
  --encoder-ckpt "${ENCODER}" --tokenizer-export "${TOKENIZER}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/motion_language_pretraining" \
  --alignment-mode text_labels --batch-size 256 --surface-rotation-augment --device cuda
python "${REPO_ROOT}/code/export.py" instruction \
  --base-dir "${ANYMO_DATA_ROOT}" --summary-csv "${SUMMARY}" \
  --encoder-ckpt "${ENCODER}" --tokenizer-export "${TOKENIZER}" \
  --class-pool-txt "${REPO_ROOT}/metadata/activity_labels_180.txt" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/instruction_tuning" \
  --narration-repeats 3 --mcq-repeats 3 --mcq-min-choices 35 \
  --mcq-max-choices 35 --contrastive-repeats 6 --batch-size 256 \
  --surface-rotation-augment --device cuda
python "${REPO_ROOT}/code/export.py" har \
  --sources all --datasets all --splits test --target-sample-rate-hz 60 \
  --har-data-root "${HAR_DATA_ROOT}" --ego4d-root "${EGO4D_ROOT}" \
  --mmea-root "${MMEA_ROOT}" --egoexo4d-root "${EGOEXO4D_ROOT}" \
  --openpack-root "${OPENPACK_ROOT}" --encoder-ckpt "${ENCODER}" \
  --tokenizer-export "${TOKENIZER}" --output-dir "${ANYMO_OUTPUT_ROOT}/har_evaluation"
python "${REPO_ROOT}/code/export.py" egoexo4d \
  --egoexo4d-root "${EGOEXO4D_ROOT}" --output-dir "${ANYMO_OUTPUT_ROOT}/egoexo4d_prepared"
python "${REPO_ROOT}/code/export.py" heldout \
  --base-dir "${ANYMO_DATA_ROOT}" --summary-csv "${SUMMARY}" \
  --encoder-ckpt "${ENCODER}" --tokenizer-export "${TOKENIZER}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/nymeria_heldout_evaluation" --device cuda
python "${REPO_ROOT}/code/export.py" heldout \
  --egoexo-eval-input-dir "${ANYMO_OUTPUT_ROOT}/egoexo4d_prepared" \
  --encoder-ckpt "${ENCODER}" --tokenizer-export "${TOKENIZER}" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/egoexo4d_zero_shot_evaluation" --device cuda
