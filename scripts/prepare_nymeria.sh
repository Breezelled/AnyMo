REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ANYMO_DATA_ROOT:?Set ANYMO_DATA_ROOT to the authorized Nymeria dataset root}"
: "${NYMERIA_TOOLS_ROOT:?Set NYMERIA_TOOLS_ROOT to the official Nymeria tools directory}"

export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

python "${REPO_ROOT}/code/sync_nymeria.py" imu \
  --base-dir "${ANYMO_DATA_ROOT}" --nymeria-pkg-root "${NYMERIA_TOOLS_ROOT}" --target-hz 60
python "${REPO_ROOT}/code/sync_nymeria.py" motion \
  --base-dir "${ANYMO_DATA_ROOT}" --nymeria-pkg-root "${NYMERIA_TOOLS_ROOT}"
python "${REPO_ROOT}/code/sync_nymeria.py" mesh \
  --base-dir "${ANYMO_DATA_ROOT}" --nymeria-pkg-root "${NYMERIA_TOOLS_ROOT}"
python "${REPO_ROOT}/code/sync_nymeria.py" narration \
  --base-dir "${ANYMO_DATA_ROOT}" --nymeria-pkg-root "${NYMERIA_TOOLS_ROOT}"
