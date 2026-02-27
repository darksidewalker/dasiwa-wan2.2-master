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

# --- 3 & 4. CLONE, PATCH & BUILD LLAMA.CPP (SKIP IF DONE) ---
cd "$PROJECT_DIR"

# Define the path to the final binary we need
QUANT_EXE="$LLAMA_DIR/build/bin/llama-quantize"

if [ -f "$QUANT_EXE" ]; then
    echo "✅ Patched llama-quantize found. Skipping patch/build steps."
else
    echo "🔨 Binary not found or incomplete. Starting build process..."

    # 1. Clone if missing
    if [ ! -d "$LLAMA_DIR" ]; then
        git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    cd "$LLAMA_DIR"
    
    # 2. Hard Reset and Patch (Only if we are actually building)
    echo "🏷️ Preparing source for Wan 2.2..."
    git reset --hard
    git checkout tags/b3962 -f
    
    if [ -f "$PROJECT_DIR/lcpp.patch" ]; then
        echo "🧹 Normalizing and Applying patch..."
        # Fix line endings on the fly
        python3 -c "f=open('$PROJECT_DIR/lcpp.patch','rb');c=f.read().replace(b'\r\n',b'\n');open('$PROJECT_DIR/lcpp.patch','wb').write(c)"
        git apply "$PROJECT_DIR/lcpp.patch"
    fi

    # 3. Build with your modern CUDA + C++17 flags
    mkdir -p build && cd build
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=native \
             -DCMAKE_CXX_STANDARD=17 \
             -DCMAKE_CUDA_STANDARD=17
             
    cmake --build . --config Release -j$(nproc)
    
    cd "$PROJECT_DIR"
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
uv pip install --refresh git+https://github.com/silveroxides/convert_to_quant.git@main#egg=convert_to_quant --no-deps --force-reinstall

echo "🍳 Installing Comfy Kitchen [CUBLAS]..."
# Install comfy-kitchen with cublas support for NVFP4 (+Blackwell)
uv pip install --refresh "comfy-kitchen[cublas]"

# --- 6. LAUNCH ---
echo "🚀 Starting DaSiWa WAN 2.2 Master ..."
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

python app.py
