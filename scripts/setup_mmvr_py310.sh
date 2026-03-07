#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mmvr-py310}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEBUG="${DEBUG:-0}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TIMM_VERSION="${TIMM_VERSION:-1.0.24}"
PYTORCH_INDEX_MODE="${PYTORCH_INDEX_MODE:-mirror}"
TORCH_WHEEL_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu118}"
CAUSAL_CONV1D_WHL_URL="${CAUSAL_CONV1D_WHL_URL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu118torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
MAMBA_SSM_WHL_URL="${MAMBA_SSM_WHL_URL:-https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu118torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
WHEEL_DIR="${WHEEL_DIR:-/tmp/mmvr-py310-wheels}"

if [ "$DEBUG" = "1" ]; then
  set -x
fi

log_step() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

log_info() {
  printf '[info] %s\n' "$1"
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    source "$conda_base/etc/profile.d/conda.sh"
    return
  fi

  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    return
  fi

  if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
    return
  fi

  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
}

run_in_env() {
  conda run --no-capture-output -n "$ENV_NAME" "$@"
}

download_wheel() {
  local url="$1"
  local dest_dir="$2"
  local filename
  filename="$(basename "$url")"
  local output="$dest_dir/$filename"

  mkdir -p "$dest_dir"

  if [ -f "$output" ]; then
    log_info "Using cached wheel: $output"
    printf '%s\n' "$output"
    return
  fi

  log_info "Downloading wheel: $filename"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$output" "$url"
  else
    echo "wget or curl is required to download wheel assets." >&2
    exit 1
  fi

  printf '%s\n' "$output"
}

ensure_conda
export PIP_NO_CACHE_DIR=1

log_step "Preparing conda environment: $ENV_NAME"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 pip -y
else
  log_info "Conda environment already exists: $ENV_NAME"
fi

log_step "Upgrading pip build tooling"
run_in_env python -m pip install --upgrade pip setuptools wheel

log_step "Installing core scientific packages"
run_in_env python -m pip install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  pandas==1.5.3 \
  matplotlib==3.7.5 \
  h5py==3.10.0

log_step "Installing PyTorch ${TORCH_VERSION} (PYTORCH_INDEX_MODE=${PYTORCH_INDEX_MODE})"
case "$PYTORCH_INDEX_MODE" in
  mirror)
    log_info "Using default pip index/mirror (no --index-url override)"
    run_in_env python -m pip install \
      torch=="$TORCH_VERSION" \
      torchvision=="$TORCHVISION_VERSION" \
      torchaudio=="$TORCHAUDIO_VERSION"
    ;;
  official-cu118)
    log_info "Using official PyTorch wheel index: $TORCH_WHEEL_INDEX"
    run_in_env python -m pip install \
      --index-url "$TORCH_WHEEL_INDEX" \
      torch=="$TORCH_VERSION" \
      torchvision=="$TORCHVISION_VERSION" \
      torchaudio=="$TORCHAUDIO_VERSION"
    ;;
  *)
    echo "Invalid PYTORCH_INDEX_MODE: $PYTORCH_INDEX_MODE (expected: mirror or official-cu118)" >&2
    exit 1
    ;;
esac

log_step "Installing Python requirements"
run_in_env python -m pip install -r "$REPO_ROOT/requirements_py310.txt"

log_step "Confirming timm version target"
log_info "timm==$TIMM_VERSION is pinned in requirements_py310.txt"

log_step "Downloading causal-conv1d and mamba-ssm wheels"
causal_wheel="$(download_wheel "$CAUSAL_CONV1D_WHL_URL" "$WHEEL_DIR")"
mamba_wheel="$(download_wheel "$MAMBA_SSM_WHL_URL" "$WHEEL_DIR")"

log_step "Installing local causal-conv1d wheel"
run_in_env python -m pip install "$causal_wheel"

log_step "Installing local mamba-ssm wheel"
run_in_env python -m pip install "$mamba_wheel"

log_step "Running syntax checks"
run_in_env python -m py_compile \
  "$REPO_ROOT/train_kpt.py" \
  "$REPO_ROOT/train_kpt_mamba.py" \
  "$REPO_ROOT/train_cls.py" \
  "$REPO_ROOT/config.py" \
  "$REPO_ROOT/config_mamba.py"

log_step "Running import checks"
run_in_env python - <<PY
import importlib
modules = [
    'torch',
    'torchvision',
    'torchaudio',
    'timm',
    'tensorboardX',
    'cv2',
    'h5py',
    'models.mmVR_Transformer',
    'models.mmVR_Mamba',
]
for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, '__version__', 'n/a')
    print(f'{name}: OK ({version})')
PY

echo
echo "Environment ready: $ENV_NAME"
echo "Torch/Torchvision/Torchaudio: $TORCH_VERSION / $TORCHVISION_VERSION / $TORCHAUDIO_VERSION"
echo "timm: $TIMM_VERSION"
echo "Wheel cache: $WHEEL_DIR"
echo "Activate with: conda activate $ENV_NAME"
