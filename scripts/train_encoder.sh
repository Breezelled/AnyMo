REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ANYMO_DATA_ROOT:?Set ANYMO_DATA_ROOT to the simulated Nymeria root}"
export ANYMO_OUTPUT_ROOT="${ANYMO_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

python "${REPO_ROOT}/code/train_stgcn.py" \
  --base-dir "${ANYMO_DATA_ROOT}" \
  --summary-csv "${ANYMO_DATA_ROOT}/geometry_aware_imu_summary.csv" \
  --output-dir "${ANYMO_OUTPUT_ROOT}/encoder" \
  --pretrain-loss predictive_infonce --max-visible-nodes 5 \
  --window-size 300 --batch-size 64 --epochs 10 --lr 3e-4 \
  --infonce-temperature 0.1 --surface-rotation-augment \
  --surface-rotation-inplane-max-deg 180 --surface-rotation-tilt-max-deg 10 --seed 42
