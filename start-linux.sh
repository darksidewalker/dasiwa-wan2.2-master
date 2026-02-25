#!/usr/bin/env bash
set -e

# --- 1. CONFIGURATION ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
PATCH_FILE="$PROJECT_DIR/lcpp.patch"
RAMDISK_PATH="/mnt/ramdisk"
RAMDISK_SIZE="55G"

echo "📂 Project Root: $PROJECT_DIR"

# --- 2. RAMDISK SETUP ---
if mountpoint -q "$RAMDISK_PATH"; then
    echo "✅ RAMDisk already mounted at $RAMDISK_PATH"
else
    echo "⚙️ Mounting $RAMDISK_SIZE RAMDisk..."
    sudo mkdir -p "$RAMDISK_PATH"
    sudo mount -t tmpfs -o size=$RAMDISK_SIZE,mode=1777 tmpfs "$RAMDISK_PATH"
fi

# --- 3. CLONE & PATCH LLAMA.CPP ---
if [ ! -d "$LLAMA_DIR" ]; then
    echo "📥 Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cd "$PROJECT_DIR" # Ensure we are in the root to find the patch

# 1. Fix line endings on the patch file IF it exists
if [ -f "lcpp.patch" ]; then
    echo "🧹 Normalizing lcpp.patch line endings..."
    python3 -c "
import os
f = 'lcpp.patch'
with open(f, 'rb') as fh:
    content = fh.read().replace(b'\r\n', b'\n')
with open(f, 'wb') as fh:
    fh.write(content)
"

    # 2. Now try to apply it to the cloned repo
    cd "$LLAMA_DIR"
    git checkout tags/b3962 -f

    if git apply --check "../lcpp.patch" > /dev/null 2>&1; then
        echo "🔧 Applying lcpp.patch..."
        git apply "../lcpp.patch"
        rm -rf build # Force recompile
    else
        echo "✅ Patch already applied or already normalized."
    fi
else
    echo "⚠️ lcpp.patch NOT FOUND. You must provide this file for WAN 2.2 support."
fi

# --- 4. BUILD BINARIES (Optimized for 40/50-Series) ---
# We check for the specific quantizer binary
if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    echo "🔨 Building llama.cpp with CUDA + C++17 support..."
    cd "$LLAMA_DIR"
    mkdir -p build && cd build

    # Force C++17 to satisfy modern CUDA (CCCL) requirements
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=native \
             -DCMAKE_CXX_STANDARD=17 \
             -DCMAKE_CUDA_STANDARD=17

    cmake --build . --config Release -j$(nproc)

    # Check if build actually succeeded
    if [ -f "bin/llama-quantize" ]; then
        echo "✅ Build complete and verified."
    else
        echo "❌ Build FAILED. Check the logs above for C++ or CUDA errors."
        exit 1
    fi
    cd "$PROJECT_DIR"
else
    echo "✅ Patched binaries already exist. Skipping build to save time."
fi

# --- 5. LOCAL VENV SETUP ---
if [ ! -d "$VENV_PATH" ]; then
    echo "⚙️ Creating local virtual environment..."
    uv venv "$VENV_PATH"
fi

echo "📦 Syncing Python dependencies..."
# Use 'uv pip' which automatically detects the local .venv
# as long as we are in the project directory.
uv pip install --refresh -r requirements.txt

echo "💎 Installing FP Quantization Tools..."
# Install silveroxides' quantizer
uv pip install --refresh git+https://github.com/silveroxides/convert_to_quant.git@main#egg=convert_to_quant --no-deps

echo "🍳 Installing Comfy Kitchen [CUBLAS]..."
# Install comfy-kitchen with cublas support for NVFP4 (+Blackwell)
uv pip install --refresh "comfy-kitchen[cublas]"

# --- 6. LAUNCH ---
echo "🚀 Starting DaSiWa WAN 2.2 Master ..."
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

python app.py
