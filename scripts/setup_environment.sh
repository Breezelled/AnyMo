REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_KIND="${1:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_environment.sh {model|simulation} [environment-name]

Examples:
  bash scripts/setup_environment.sh model
  bash scripts/setup_environment.sh simulation
  bash scripts/setup_environment.sh model my-anymo-env
EOF
}

if [[ "${ENV_KIND}" == "-h" || "${ENV_KIND}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${ENV_KIND}" || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

command -v conda >/dev/null 2>&1 || {
  echo "conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
}

case "${ENV_KIND}" in
  model)
    ENV_NAME="${2:-anymo}"
    conda create -y -n "${ENV_NAME}" python=3.10 pip
    conda run -n "${ENV_NAME}" python -m pip install \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cu128
    conda run -n "${ENV_NAME}" python -m pip install -r "${REPO_ROOT}/requirements.txt"
    conda run -n "${ENV_NAME}" python -m pip install packaging psutil ninja
    MAX_JOBS="${MAX_JOBS:-4}" conda run -n "${ENV_NAME}" python -m pip install \
      flash-attn==2.8.3 --no-build-isolation
    conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}/code/ms-swift"
    ;;
  simulation)
    ENV_NAME="${2:-anymo-sim}"
    conda create -y -n "${ENV_NAME}" python=3.10 pip
    conda run -n "${ENV_NAME}" python -m pip install \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu117
    conda run -n "${ENV_NAME}" python -m pip install \
      numpy==1.26.4 pandas==2.3.3 scipy==1.13.1 \
      zarr==2.18.3 numcodecs==0.13.1 tqdm==4.67.1 networkx==3.2.1 \
      PyYAML==6.0.2 pybullet==3.2.7 matplotlib wandb fvcore iopath
    conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
    conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}/code/WIMUSim" --no-deps
    ;;
  *)
    usage
    ;;
esac

echo "Environment '${ENV_NAME}' is ready. Activate it with: conda activate ${ENV_NAME}"
