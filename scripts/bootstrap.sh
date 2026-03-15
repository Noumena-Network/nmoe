#!/bin/bash
# nmoe bootstrap for cloud GPU instances (cloud GPU providers)
#
# This is the HOST bootstrap path (non-default). The golden path is Docker.
# Use this for quick iteration on cloud instances without Docker.
#
# Tested on:
#   - Prime Intellect 8x B200 (ubuntu_22_cuda_12 image, Ubuntu 24.04, CUDA 12.8)
#
# Usage:
#   git clone https://github.com/noumena-network/nmoe.git
#   cd nmoe && bash scripts/bootstrap.sh
#
# Time estimate: ~8-12 minutes (mostly torch + deps)

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. sudo fallback (containers often run as root without sudo installed)
# -----------------------------------------------------------------------------
if ! command -v sudo &> /dev/null; then
    if [ "$(id -u)" -eq 0 ]; then
        sudo() { "$@"; }
    else
        echo "ERROR: sudo not found. Install sudo or run as root."
        exit 1
    fi
fi

echo "=== nmoe bootstrap (host path) ==="
echo "Started: $(date)"

# -----------------------------------------------------------------------------
# 1. System limits (file descriptors for data prep)
# -----------------------------------------------------------------------------
echo "[1/11] Setting system limits..."

# Increase file descriptor limit for data prep (many shards)
ulimit -n 65536 2>/dev/null || echo "Warning: could not increase ulimit (may need sudo)"

# -----------------------------------------------------------------------------
# 2. Environment variables
# -----------------------------------------------------------------------------
echo "[2/11] Setting up environment..."

# CUDA installation layout varies by image (e.g. /usr/local/cuda vs apt /usr/lib/cuda).
if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d /usr/lib/cuda ]; then
    export CUDA_HOME=/usr/lib/cuda
else
    echo "ERROR: CUDA_HOME not found (expected /usr/local/cuda or /usr/lib/cuda)"
    exit 1
fi
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# Detect GPU architecture (B200=sm100a, H100=sm90) and set build flags.
# NOTE: We keep the repo's primary target as B200; H100 is BF16-only bring-up.
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
if [ -z "${GPU_CAP}" ]; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    case "${GPU_NAME}" in
      *H100*)
        GPU_CAP="9.0"
        ;;
      *B200*|*Blackwell*)
        GPU_CAP="10.0"
        ;;
      *)
        GPU_CAP=""
        ;;
    esac
fi
case "${GPU_CAP}" in
  10.0*)
    export NMOE_CUDA_ARCH="100a"
    export TORCH_CUDA_ARCH_LIST="10.0a"
    export CUTLASS_NVCC_ARCHS="100a"
    ;;
  9.0*)
    export NMOE_CUDA_ARCH="90"
    export TORCH_CUDA_ARCH_LIST="9.0"
    export CUTLASS_NVCC_ARCHS="90"
    ;;
  *)
    echo "ERROR: unsupported GPU compute capability: '${GPU_CAP}'"
    echo "Expected: 10.0 (B200) or 9.0 (H100)."
    exit 1
    ;;
esac

# Persist to bashrc (idempotent)
if ! grep -q 'NMOE_ENV_SET' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'ENVEOF'
# nmoe environment (NMOE_ENV_SET)
export PATH="$HOME/.local/bin:$HOME/.bun/bin:/usr/local/go/bin:$PATH"
# CUDA installation layout varies by image (e.g. /usr/local/cuda vs apt /usr/lib/cuda).
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
elif [ -d /usr/lib/cuda ]; then
  export CUDA_HOME=/usr/lib/cuda
fi
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
ulimit -n 65536 2>/dev/null || true
ENVEOF
fi

# Persist detected arch flags (idempotent per-host).
if ! grep -q 'NMOE_ARCH_SET' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << EOF
# nmoe arch flags (NMOE_ARCH_SET)
export NMOE_CUDA_ARCH="${NMOE_CUDA_ARCH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export CUTLASS_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS}"
EOF
fi

# Verify CUDA
echo "CUDA: $(nvcc --version | grep release)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# -----------------------------------------------------------------------------
# 3. System dependencies
# -----------------------------------------------------------------------------
echo "[3/11] Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ninja-build \
    zlib1g-dev \
    libxml2-dev \
    python3-dev \
    unzip

# Install Go 1.25 (for nmon)
if ! command -v go &> /dev/null || [[ "$(go version)" != *"go1.25"* ]]; then
    echo "Installing Go 1.25..."
    wget -q https://go.dev/dl/go1.25.0.linux-amd64.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf go1.25.0.linux-amd64.tar.gz
    rm go1.25.0.linux-amd64.tar.gz
fi
export PATH=$PATH:/usr/local/go/bin

# Install Node.js 20 (for nviz)
if ! command -v node &> /dev/null || [[ "$(node --version)" != v20* ]]; then
    echo "Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "Node.js: $(node --version)"

# Install bun (for nviz)
if ! command -v bun &> /dev/null; then
    echo "Installing bun..."
    curl -fsSL https://bun.sh/install | bash
fi
export PATH="$HOME/.bun/bin:$PATH"
echo "bun: $(bun --version 2>/dev/null || echo 'missing')"

# Install cloudflared (for tunnels)
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared..."
    sudo curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    sudo chmod +x /usr/local/bin/cloudflared
fi

# -----------------------------------------------------------------------------
# 4. Install uv (fast Python package manager)
# -----------------------------------------------------------------------------
echo "[4/11] Installing uv..."

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found after install"
    exit 1
fi
echo "uv: $(uv --version)"

# -----------------------------------------------------------------------------
# 5. Clone nmoe (if not already in repo)
# -----------------------------------------------------------------------------
echo "[5/11] Setting up nmoe..."

if [ ! -f "pyproject.toml" ]; then
    cd ~
    if [ ! -d "nmoe" ]; then
        git clone https://github.com/noumena-network/nmoe.git
    fi
    cd nmoe
fi

NMOE_ROOT=$(pwd)
echo "nmoe root: $NMOE_ROOT"

# Persist repo-specific PATH so `n` works from anywhere (contract).
mkdir -p "$HOME/.config/nmoe"
cat > "$HOME/.config/nmoe/repo_env.sh" << EOF
# nmoe repo env (generated by bootstrap.sh)
export NMOE_ROOT="${NMOE_ROOT}"
export PATH="\$NMOE_ROOT/.venv/bin:\$PATH"
EOF
source "$HOME/.config/nmoe/repo_env.sh"
if ! grep -q 'NMOE_REPO_ENV' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'ENVEOF'
# nmoe repo env (NMOE_REPO_ENV)
if [ -f "$HOME/.config/nmoe/repo_env.sh" ]; then
  source "$HOME/.config/nmoe/repo_env.sh"
fi
ENVEOF
fi

# -----------------------------------------------------------------------------
# 6. Create venv + install dependencies from pyproject.toml
# -----------------------------------------------------------------------------
echo "[6/11] Creating venv and installing dependencies..."

if [ ! -d ".venv" ]; then
    uv python install 3.12
    uv venv .venv --python 3.12
fi
source .venv/bin/activate

# Install deps from pyproject.toml first
uv sync --extra research

# Ensure project (and console script `n`) is installed into the venv.
uv pip install -e . --no-deps

# Install torch last (nightly cu128 - not in pyproject.toml, uv can't resolve it)
echo "Installing torch nightly cu128..."
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify torch
python -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}, gpus {torch.cuda.device_count()}')"

# Verify n CLI is available
.venv/bin/n --help

# -----------------------------------------------------------------------------
# 7. Verify Triton (installed as a dependency of torch)
# -----------------------------------------------------------------------------
echo "[7/11] Verifying Triton..."
python -c "import triton, triton.runtime.cache; print(f'triton {triton.__version__}')"

# -----------------------------------------------------------------------------
# 8. Vendor FlashAttention and FlashMLA
# -----------------------------------------------------------------------------
echo "[8/11] Vendoring FlashAttention and FlashMLA..."

mkdir -p third_party

# FlashAttention CuTe (FA4)
if [ ! -d "third_party/flash_attn/cute" ]; then
    mkdir -p third_party/flash_attn_upstream
    cd third_party/flash_attn_upstream
    git init
    git remote add origin https://github.com/Dao-AILab/flash-attention.git
    git sparse-checkout init --cone
    git sparse-checkout set flash_attn/cute
    git fetch --depth 1 origin ac9b5f107f2f19cd0ca6e01548d20d072a46335c
    git checkout --detach FETCH_HEAD
    cd $NMOE_ROOT
    mkdir -p third_party/flash_attn
    cp -a third_party/flash_attn_upstream/flash_attn/cute third_party/flash_attn/cute
    rm -rf third_party/flash_attn_upstream
    echo "Vendored FlashAttention CuTe (FA4)"
fi

# FlashMLA (SM100-only; needed for MLA backward on B200).
if [ "${NMOE_CUDA_ARCH}" = "100a" ]; then
    if [ ! -d "third_party/flashmla" ]; then
        git clone https://github.com/deepseek-ai/FlashMLA.git third_party/flashmla
        cd third_party/flashmla
        git checkout 1408756a88e52a25196b759eaf8db89d2b51b5a1
        git submodule update --init --recursive
        rm -rf .git
        find . -name .git -o -name .gitmodules -exec rm -rf {} + 2>/dev/null || true
        rm -rf benchmark docs flash_mla tests setup.py
        rm -rf csrc/sm90 csrc/smxx csrc/sm100/decode csrc/sm100/prefill/sparse csrc/pybind.cpp csrc/params.h 2>/dev/null || true
        cd $NMOE_ROOT
        echo "Vendored FlashMLA (dense SM100 prefill bwd only)"
    fi
else
    echo "Skipping FlashMLA vendoring (NMOE_CUDA_ARCH=${NMOE_CUDA_ARCH})"
fi

# Quack (memory-efficient vocab-scale cross-entropy)
if [ ! -d "third_party/quack" ]; then
    mkdir -p third_party/quack_upstream
    cd third_party/quack_upstream
    git init
    git remote add origin https://github.com/Dao-AILab/quack.git
    git fetch --depth 1 origin 32b51a5b9c4620724158be60ff6a228667bbd391
    git checkout --detach FETCH_HEAD
    cd $NMOE_ROOT
    mkdir -p third_party/quack
    cp -a third_party/quack_upstream/quack third_party/quack/quack
    cp third_party/quack_upstream/LICENSE third_party/quack/LICENSE
    rm -rf third_party/quack_upstream
    echo "Vendored Quack (pinned)"
fi

# -----------------------------------------------------------------------------
# 9. Build CUDA kernels
# -----------------------------------------------------------------------------
echo "[9/11] Building CUDA kernels..."

cd $NMOE_ROOT/nmoe/csrc
make NMOE_CUDA_ARCH="${NMOE_CUDA_ARCH}" || echo "Warning: some kernel builds may have failed (gpu.so optional)"
cd $NMOE_ROOT

# Set PYTHONPATH (idempotent)
export PYTHONPATH=$NMOE_ROOT:$NMOE_ROOT/nmoe/csrc:$NMOE_ROOT/third_party/flash_attn:$NMOE_ROOT/third_party/quack
if ! grep -q 'NMOE_PYTHONPATH_SET' ~/.bashrc 2>/dev/null; then
    echo "# nmoe PYTHONPATH (NMOE_PYTHONPATH_SET)" >> ~/.bashrc
    echo "export PYTHONPATH=$NMOE_ROOT:$NMOE_ROOT/nmoe/csrc:$NMOE_ROOT/third_party/flash_attn:$NMOE_ROOT/third_party/quack" >> ~/.bashrc
fi

# Verify core imports
python -c "from nmoe.csrc import rdep; print('rdep kernel loaded')"
python -c "from nmoe import config, model, train; print('nmoe core modules loaded')"

# -----------------------------------------------------------------------------
# 10. Build monitoring tools (nmon + nviz)
# -----------------------------------------------------------------------------
echo "[10/11] Building monitoring tools..."

# Build nmon (Go TUI)
if [ -d "tools/nmon" ]; then
    cd $NMOE_ROOT/tools/nmon
    go build -o nmon ./cmd/nmon
    echo "nmon built: $NMOE_ROOT/tools/nmon/nmon"
    cd $NMOE_ROOT
fi

# Install nviz dependencies and build (Next.js)
if [ -d "nviz" ]; then
    cd $NMOE_ROOT/nviz
    bun install
    NVIZ_METRICS_DIR=/data/metrics bun run build
    echo "nviz built and ready"
    cd $NMOE_ROOT
fi

# -----------------------------------------------------------------------------
# 11. Create data directory
# -----------------------------------------------------------------------------
echo "[11/12] Setting up data directory..."

sudo mkdir -p /data
DATA_OWNER="${SUDO_USER:-${USER:-$(id -un)}}"
sudo chown "${DATA_OWNER}:${DATA_OWNER}" /data
echo "Data directory ready: /data"

# -----------------------------------------------------------------------------
# 12. Download CORE eval bundle
# -----------------------------------------------------------------------------
echo "[12/12] Downloading CORE eval bundle..."

EVAL_BUNDLE_URL="https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_DIR="/data/eval/eval_bundle"

if [ ! -d "$EVAL_BUNDLE_DIR" ] || [ -z "$(ls -A $EVAL_BUNDLE_DIR 2>/dev/null)" ]; then
    mkdir -p "$(dirname $EVAL_BUNDLE_DIR)"
    echo "Downloading from $EVAL_BUNDLE_URL..."
    curl -L -o /tmp/eval_bundle.zip "$EVAL_BUNDLE_URL"
    unzip -q /tmp/eval_bundle.zip -d "$(dirname $EVAL_BUNDLE_DIR)"
    rm /tmp/eval_bundle.zip
    echo "CORE eval bundle installed: $EVAL_BUNDLE_DIR"
else
    echo "CORE eval bundle already exists: $EVAL_BUNDLE_DIR"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=== Bootstrap complete ==="
echo "Finished: $(date)"
echo ""
echo "Activate environment:"
echo "  cd $NMOE_ROOT && source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download data (100M tokens for quick test):"
echo "     python -m nmoe.data.cli prep \\"
echo "       --source hub_parquet --dataset karpathy/fineweb-edu-100b-shuffle --split train \\"
echo "       --output /data/fineweb_test --name fineweb_test \\"
echo "       --tokenizer gpt2 --vocab-size 50304 --eos-token-id 50256 \\"
echo "       --max-tokens-total 100M --num-shards 32"
echo ""
echo "  2. Run speedrun:"
echo "     n speedrun"
echo ""
echo "  3. Monitor with nmon (TUI):"
echo "     n mon"
echo ""
echo "  4. Monitor with nviz (web):"
echo "     n viz"
echo ""
