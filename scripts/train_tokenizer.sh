REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ANYMO_DATA_ROOT:?Set ANYMO_DATA_ROOT to the simulated Nymeria root}"
export ANYMO_OUTPUT_ROOT="${ANYMO_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

python "${REPO_ROOT}/code/train_pqvae.py" \
  --base-dir "${ANYMO_DATA_ROOT}" \
  --summary-csv "${ANYMO_DATA_ROOT}/geometry_aware_imu_summary.csv" \
  --encoder-ckpt "${ANYMO_OUTPUT_ROOT}/encoder/encoder_best.pt" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/tokenizer" --window-size 300 \
  --batch-size 1024 --epochs 500 --lr 3e-4 --max-visible-nodes 5 \
  --input-dim 256 --bottleneck-dim 128 --num-codebooks 2 \
  --codebook-size 2048 --codebook-dim 64 --ema-decay 0.99 \
  --surface-rotation-augment --surface-rotation-inplane-max-deg 180 \
  --surface-rotation-tilt-max-deg 10 --seed 42
