#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mmvr-py310}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TORCH_WHEEL_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu116}"
CAUSAL_CONV1D_WHL_URL="${CAUSAL_CONV1D_WHL_URL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.0.2/causal_conv1d-1.0.2+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
MAMBA_SSM_WHL_URL="${MAMBA_SSM_WHL_URL:-https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
WHEEL_DIR="${WHEEL_DIR:-/tmp/mmvr-py310-wheels}"

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
  conda run -n "$ENV_NAME" "$@"
}

download_wheel() {
  local url="$1"
  local dest_dir="$2"
  local filename
  filename="$(basename "$url")"
  local output="$dest_dir/$filename"

  mkdir -p "$dest_dir"

  if [ ! -f "$output" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget -O "$output" "$url"
    elif command -v curl >/dev/null 2>&1; then
      curl -L -o "$output" "$url"
    else
      echo "wget or curl is required to download wheel assets." >&2
      exit 1
    fi
  fi

  printf '%s\n' "$output"
}

ensure_conda
export PIP_NO_CACHE_DIR=1

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 pip -y
fi

run_in_env python -m pip install --upgrade pip setuptools==69.5.1 wheel
run_in_env python -m pip install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  pandas==1.5.3 \
  matplotlib==3.7.5 \
  h5py==3.10.0
run_in_env python -m pip install \
  --extra-index-url "$TORCH_WHEEL_INDEX" \
  torch==1.13.1+cu116 \
  torchvision==0.14.1+cu116 \
  torchaudio==0.13.1+cu116
run_in_env python -m pip install -r "$REPO_ROOT/requirements_py310.txt"
causal_wheel="$(download_wheel "$CAUSAL_CONV1D_WHL_URL" "$WHEEL_DIR")"
mamba_wheel="$(download_wheel "$MAMBA_SSM_WHL_URL" "$WHEEL_DIR")"
run_in_env python -m pip install "$causal_wheel"
run_in_env python -m pip install "$mamba_wheel"
run_in_env python -m py_compile \
  "$REPO_ROOT/train_kpt.py" \
  "$REPO_ROOT/train_kpt_mamba.py" \
  "$REPO_ROOT/train_cls.py" \
  "$REPO_ROOT/config.py" \
  "$REPO_ROOT/config_mamba.py"
run_in_env python - <<PY
import importlib
modules = [
    'torch',
    'torchvision',
    'torchaudio',
    'tensorboardX',
    'cv2',
    'h5py',
    'models.mmVR_Transformer',
    'models.mmVR_Mamba',
]
for name in modules:
    importlib.import_module(name)
    print(f'{name}: OK')
PY

echo
echo "Environment ready: $ENV_NAME"
echo "Wheel cache: $WHEEL_DIR"
echo "Activate with: conda activate $ENV_NAME"
