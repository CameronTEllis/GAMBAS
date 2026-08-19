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

# ---------------------------------------------------------------------------
# Toolchain preflight
# ---------------------------------------------------------------------------
# The CUDA extensions below need BOTH nvcc and a modern host compiler. nvcc
# shells out to g++ for host code, so an ancient system g++ fails even when nvcc
# is present -- with a confusing cascade of "string_view: No such file or
# directory" and "#error You need C++17" rather than anything naming the
# compiler. Checking here converts a 30-minute failed build into a 1-second
# message.
#
# Set MODULES_TO_LOAD in config.local.sh to fix both at once, e.g.
#   export MODULES_TO_LOAD="gcc/9.1.0 cuda/12.1.1"
GCC_MIN=9
TOOLCHAIN_OK=1

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc     : $(nvcc --version | sed -n 's/^.*release \([0-9.]*\).*$/\1/p' | tail -1)"
else
  echo "nvcc     : NOT FOUND"
  TOOLCHAIN_OK=0
fi

if command -v g++ >/dev/null 2>&1; then
  GCC_MAJOR="$(g++ -dumpversion | cut -d. -f1)"
  echo "g++      : $(g++ -dumpversion)  ($(command -v g++))"
  if [ "${GCC_MAJOR:-0}" -lt "$GCC_MIN" ]; then
    echo "           ^ too old; PyTorch headers need GCC >= $GCC_MIN"
    TOOLCHAIN_OK=0
  fi
else
  echo "g++      : NOT FOUND"
  TOOLCHAIN_OK=0
fi

if [ "$TOOLCHAIN_OK" = "0" ] && [ "${SKIP_MAMBA:-0}" != "1" ]; then
  echo
  echo "=========================================================================="
  echo " Toolchain incomplete -- the CUDA extensions cannot build."
  echo " Try:   module avail gcc ; module avail cuda"
  echo "        module load gcc/9.1.0 cuda/12.1.1"
  echo " then record it in sr/cluster/config.local.sh:"
  echo '        export MODULES_TO_LOAD="gcc/9.1.0 cuda/12.1.1"'
  echo
  echo " Continuing with SKIP_MAMBA=1 so you still get a working environment for"
  echo " stages 1-2 (simulate / build dataset / QC / folds)."
  echo "=========================================================================="
  echo
  SKIP_MAMBA=1
fi

# Point nvcc at the host compiler explicitly. CUDAHOSTCXX is the variable that
# actually controls it -- setting only CXX is not enough.
if [ "${TOOLCHAIN_OK}" = "1" ]; then
  export CC="${CC:-$(command -v gcc)}"
  export CXX="${CXX:-$(command -v g++)}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
fi

# Compile mamba_ssm / causal_conv1d for EVERY GPU architecture the jobs might land
# on -- NOT just the build node's. Sherlock's gpu partition is heterogeneous
# (V100 7.0, T4 7.5, A100 8.0, A40/L40/A6000 8.6, H100 9.0), and a kernel built for
# one arch fails on any other with
#     RuntimeError: CUDA error: no kernel image is available for execution
# which is exactly what a K-fold sweep hits when different folds land on different
# GPUs. Auto-detecting the build node's single arch (the old behaviour) caused this.
# A multi-arch build is a few minutes slower to compile but runs anywhere.
#
# Override TORCH_CUDA_ARCH_LIST (and pin GPU_TYPE / a SLURM --constraint) only if
# you deliberately want a faster single-arch build tied to one GPU model. Add 6.0
# if you must run on P100 nodes.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
  echo "arch list: $TORCH_CUDA_ARCH_LIST (multi-arch default; V100 .. H100)"
else
  echo "arch list: $TORCH_CUDA_ARCH_LIST (from environment)"
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
# transformers is PINNED and the pin is load-bearing. mamba_ssm/__init__.py
# eagerly imports its LM head, which imports
# `transformers.generation.GreedySearchDecoderOnlyOutput` -- a symbol removed in
# transformers 5.x. Unpinned, pip resolves to 5.x, which additionally requires
# torch >= 2.4 and so disables itself against torch 2.2.2, and then
# `from mamba_ssm import Mamba` fails at import with an ImportError that names
# transformers rather than the version conflict. GAMBAS never uses the LM head;
# the pin exists purely to keep that import chain satisfiable.
pip install \
  "numpy<2" \
  "scipy>=1.10" \
  matplotlib \
  ml-collections \
  tqdm \
  lpips \
  "monai==1.3.0" \
  "transformers==${TRANSFORMERS_VER:-4.44.2}" \
  tensorboard \
  nibabel

echo
echo "=========================================================================="
echo " Everything above is all that stages 1-2 need (simulate + build dataset)."
echo " What follows is only required for TRAINING and INFERENCE."
echo "=========================================================================="

if [ "${SKIP_MAMBA:-0}" = "1" ]; then
  echo "SKIP_MAMBA=1 -- stopping before the CUDA extensions."
  echo "You can simulate and build datasets now; rerun without SKIP_MAMBA later."
  MAMBA_STATUS="skipped"
else
  echo "--- causal-conv1d + mamba_ssm ---"
  # These two build CUDA extensions. Notes on why this step is fragile:
  #
  #  * causal-conv1d is OPTIONAL. mamba_ssm imports it in a try/except and falls
  #    back to a plain conv1d when it is missing -- slower, but correct. So a
  #    failure here is not fatal and must not abort the script.
  #  * Version floor is >=1.4.0, per the upstream mamba README. (An earlier
  #    revision of this script pinned ==1.2.2, which does not exist on PyPI at
  #    all -- only 1.2.2.post1 -- and was below the floor anyway.)
  #  * Prebuilt wheels are keyed to an exact torch / CUDA / cpython combination.
  #    If none matches, pip falls back to an sdist and compiles, which needs nvcc
  #    and can take 30+ minutes. MAX_JOBS caps parallelism so that build cannot
  #    OOM the node.
  #  * mamba_ssm is installed FROM GIT, not from PyPI. The PyPI sdist for 2.2.2
  #    is ~85 kB and does not ship the `csrc/` CUDA sources that its own setup.py
  #    references, so a source build dies with
  #      ninja: error: '.../csrc/selective_scan/selective_scan.cpp' ... missing
  #    no matter how good your toolchain is. The git tag contains csrc/.
  #
  # MAX_JOBS default is 2, deliberately low: each mamba CUDA translation unit
  # (selective_scan_*.cu, compiled for every arch) can peak at several GB in cicc,
  # so 4 parallel compiles OOM-kill the node -- the failure looks like
  #    c++: fatal error: Killed signal terminated program cc1plus
  #    sh: line 1: NNNNN Killed  cicc ...
  # which is the OOM killer, not a compile error. Raise it only on a fat-memory
  # node (e.g. MAX_JOBS=4 with --mem>=64G); lower to 1 if you still get killed.
  export MAX_JOBS="${MAX_JOBS:-2}"
  CC1D_SPEC="${CC1D_SPEC:-causal-conv1d>=1.4.0}"
  MAMBA_VER="${MAMBA_VER:-2.2.2}"
  MAMBA_SPEC="${MAMBA_SPEC:-git+https://github.com/state-spaces/mamba.git@v${MAMBA_VER}}"

  # Each install is failure-tolerant. With `set -e`, a bare pip failure here
  # would kill the script before the verification block, leaving you with no
  # report of what actually installed -- which is what happened with the earlier
  # bad pin.
  #
  # NOTE: the retry uses only the package's own FORCE_BUILD env var. Do NOT add
  # `--no-binary=:all:` -- that applies to every dependency in the resolution,
  # including torch, and would try to compile PyTorch from source.
  CC1D_OK=1
  if ! pip install "$CC1D_SPEC"; then
    echo "no matching causal-conv1d wheel; trying a source build (slow, needs nvcc)"
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation "$CC1D_SPEC" \
      || CC1D_OK=0
  fi
  if [ "$CC1D_OK" = "0" ]; then
    echo "causal-conv1d NOT installed. This is OPTIONAL: mamba_ssm imports it in"
    echo "a try/except and falls back to a plain conv1d. Slower, still correct."
  fi

  # `--no-build-isolation` is required, not optional. With isolation, pip builds
  # in a fresh venv that has no numpy and downloads its OWN latest torch (2.6+),
  # so setup.py either crashes on `No module named 'torch'` or probes the wrong
  # torch version. Passing it makes the build see the torch we just installed.
  MAMBA_OK=1
  echo "installing mamba_ssm from $MAMBA_SPEC"
  MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation "$MAMBA_SPEC" || MAMBA_OK=0
  if [ "$MAMBA_OK" = "1" ]; then
    MAMBA_STATUS="installed"
  else
    MAMBA_STATUS="FAILED"
    echo
    echo "mamba_ssm did not install. This one IS required -- "
    echo "models/mamba_modules3D.py line 19 does an unconditional"
    echo "'from mamba_ssm import Mamba', so the generator cannot be built."
    echo "Stages 1-2 still work; training does not."
  fi
fi

echo
echo "--- verification ---"
# Reports the two capabilities separately, because they fail independently and
# the CPU half is useful on its own: stages 1-2 (simulate, build dataset, QC,
# folds) need only numpy + SimpleITK, while training additionally needs torch,
# CUDA and mamba_ssm.
python - <<'EOF'
import importlib, sys

def probe(mods):
    out = {}
    for m in mods:
        try:
            mod = importlib.import_module(m)
            out[m] = getattr(mod, '__version__', 'ok')
            print('  ok   %-16s %s' % (m, out[m]))
        except Exception as e:
            out[m] = None
            print('  FAIL %-16s %s' % (m, e))
    return out

print(' [stages 1-2: simulate / build dataset / QC / folds]')
cpu = probe(['numpy', 'SimpleITK', 'scipy', 'matplotlib'])
cpu_ok = all(cpu[m] for m in ('numpy', 'SimpleITK'))

print(' [training / inference]')
gpu = probe(['torch', 'torchvision', 'ml_collections', 'monai', 'tensorboard',
             'mamba_ssm', 'causal_conv1d'])

train_ok = all(gpu[m] for m in ('torch', 'mamba_ssm'))
if gpu['mamba_ssm'] and not gpu['causal_conv1d']:
    print('  note causal_conv1d absent -- mamba_ssm falls back to a plain '
          'conv1d. Slower, still correct.')

if gpu['torch']:
    import torch
    print('  cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('  device:', torch.cuda.get_device_name(0))
        if gpu['mamba_ssm']:
            try:
                from mamba_ssm import Mamba
                x = torch.randn(1, 64, 128).cuda()
                y = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).cuda()(x)
                print('  mamba forward ok:', tuple(y.shape))
            except Exception as e:
                train_ok = False
                print('  FAIL mamba forward:', e)
    else:
        print('  (no GPU visible from this node -- run the check on a GPU node '
              'before trusting the training half)')

print()
print('  stages 1-2 (simulate, dataset, QC) : %s'
      % ('READY' if cpu_ok else 'NOT READY'))
print('  training / inference               : %s'
      % ('READY' if train_ok else 'NOT READY'))
# Exit non-zero only if the CPU half is broken; a missing CUDA extension is a
# recoverable, partial outcome rather than a failed setup.
sys.exit(0 if cpu_ok else 1)
EOF

echo
# Snapshot the resolved versions. Finding a working combination on this stack
# took several rounds (nvcc, GCC, an incomplete sdist, a transformers conflict),
# and pip resolves every unpinned dependency to a current release against a
# 2024-era torch. This turns "which versions worked" from tribal knowledge into
# a file you can diff or reinstall from.
FROZEN="$HERE/requirements-frozen.txt"
{
  echo "# Generated by setup_env.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# python $(python -V 2>&1 | cut -d' ' -f2)  gcc $(g++ -dumpversion 2>/dev/null || echo n/a)"
  echo "# nvcc $(nvcc --version 2>/dev/null | sed -n 's/^.*release \([0-9.]*\).*$/\1/p' | tail -1 || echo n/a)"
  echo "# TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
  echo "# MODULES_TO_LOAD=${MODULES_TO_LOAD:-<none>}"
  pip freeze
} > "$FROZEN"
echo "wrote $FROZEN"

echo
echo "environment '$CONDA_ENV' built (mamba: ${MAMBA_STATUS:-unknown})."
echo "Set ENV_STYLE=conda and CONDA_ENV=$CONDA_ENV in sr/cluster/config.local.sh"
echo
echo "If the training half is NOT READY you can still run stages 1-2 now:"
echo "  cd \"\$GAMBAS_ROOT\" && python -m sr.simulate_lowres --help"
