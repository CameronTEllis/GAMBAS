#!/bin/bash
# ============================================================================
# setup_env.sh -- build the conda environment on the cluster.
#
# Run this ONCE, interactively, on a node that has the CUDA toolkit and a
# compiler available (often a GPU node -- mamba_ssm and causal-conv1d compile
# CUDA kernels at install time).
#
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2:00:00 \
#        --pty bash
#   bash sr/cluster/setup_env.sh
#
# Notes on the two packages that reliably cause trouble:
#
#   mamba_ssm / causal-conv1d
#       These build CUDA extensions. `pip install mamba_ssm` from source can take
#       30+ min and needs ~16 GB RAM and MAX_JOBS limited, or it OOMs the node.
#       Prefer the prebuilt wheel matching your torch/CUDA/python combo from
#       https://github.com/state-spaces/mamba/releases -- that is what this
#       script tries first.
#       Set MAMBA_FORCE_BUILD=TRUE to force a source build.
#
#   SimpleITK
#       Install with --only-binary to avoid a 40-minute ITK compile.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/config.local.sh" ]; then source "$HERE/config.local.sh"; else source "$HERE/config.sh"; fi

PY_VER="${PY_VER:-3.10}"
TORCH_VER="${TORCH_VER:-2.2.2}"
TORCHVISION_VER="${TORCHVISION_VER:-0.17.2}"
CUDA_TAG="${CUDA_TAG:-cu121}"

if [ -n "$MODULES_TO_LOAD" ]; then
  # shellcheck disable=SC2086
  module load $MODULES_TO_LOAD
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

if conda env list | grep -qE "^${CONDA_ENV}\s"; then
  echo "env '$CONDA_ENV' already exists -- activating it"
else
  conda create -y -n "$CONDA_ENV" "python=$PY_VER"
fi
conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel packaging ninja

echo "--- torch ---"
pip install "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" \
    --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "--- SimpleITK (binary only) ---"
pip install --only-binary=SimpleITK "SimpleITK==2.2.0" || pip install --only-binary=SimpleITK SimpleITK

echo "--- core python deps ---"
pip install \
  "numpy<2" \
  "scipy>=1.10" \
  matplotlib \
  ml-collections \
  tqdm \
  lpips \
  "monai==1.3.0" \
  tensorboard \
  nibabel

echo "--- causal-conv1d + mamba_ssm ---"
# Limit parallel compile jobs so a source build cannot OOM the login/compute node.
export MAX_JOBS="${MAX_JOBS:-4}"
pip install causal-conv1d==1.2.2 || {
  echo "prebuilt causal-conv1d wheel unavailable; building from source"
  CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d==1.2.2
}
pip install mamba_ssm==2.2.2 || {
  echo "prebuilt mamba_ssm wheel unavailable; building from source"
  MAMBA_FORCE_BUILD=TRUE pip install mamba_ssm==2.2.2
}

echo
echo "--- verification ---"
python - <<'EOF'
import importlib, sys
ok = True
for m in ['torch', 'torchvision', 'SimpleITK', 'numpy', 'scipy',
          'ml_collections', 'monai', 'mamba_ssm', 'tensorboard']:
    try:
        mod = importlib.import_module(m)
        print('  ok   %-16s %s' % (m, getattr(mod, '__version__', '')))
    except Exception as e:
        ok = False
        print('  FAIL %-16s %s' % (m, e))
import torch
print('  cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device:', torch.cuda.get_device_name(0))
    try:
        from mamba_ssm import Mamba
        x = torch.randn(1, 64, 128).cuda()
        y = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).cuda()(x)
        print('  mamba forward ok:', tuple(y.shape))
    except Exception as e:
        ok = False
        print('  FAIL mamba forward:', e)
sys.exit(0 if ok else 1)
EOF

echo
echo "environment '$CONDA_ENV' ready."
echo "Set ENV_STYLE=conda and CONDA_ENV=$CONDA_ENV in sr/cluster/config.local.sh"
