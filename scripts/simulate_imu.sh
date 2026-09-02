REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ANYMO_DATA_ROOT:?Set ANYMO_DATA_ROOT to the prepared Nymeria root}"
: "${NYMERIA_TOOLS_ROOT:?Set NYMERIA_TOOLS_ROOT to the official Nymeria tools directory}"

export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"
CANDIDATES="${ANYMO_DATA_ROOT}/body_surface_candidates.npz"
CANDIDATES_WITH_FRAMES="${ANYMO_DATA_ROOT}/body_surface_candidates_with_local_frames.npz"
SUMMARY="${ANYMO_DATA_ROOT}/geometry_aware_imu_summary.csv"

python "${REPO_ROOT}/code/simulate.py" candidates \
  --summary-csv "${ANYMO_DATA_ROOT}/nymeria_mesh_60hz_summary.csv" \
  --output-path "${CANDIDATES}" \
  --nymeria-pkg-root "${NYMERIA_TOOLS_ROOT}" --seed 42
python "${REPO_ROOT}/code/simulate.py" frames \
  --input-npz "${CANDIDATES}" --output-npz "${CANDIDATES_WITH_FRAMES}"
python "${REPO_ROOT}/code/simulate.py" generate \
  --base-dir "${ANYMO_DATA_ROOT}" --selection-npz "${CANDIDATES_WITH_FRAMES}" \
  --sample-rate 60 --seed 42
python "${REPO_ROOT}/code/simulate.py" convert \
  --base-dir "${ANYMO_DATA_ROOT}" \
  --summary-csv "${SUMMARY}"
