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
    echo "📥 Cloning llama.cpp (Tag b3962)..."
    git clone --branch b3962 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

# Apply the patch if it exists and hasn't been applied yet
if [ -f "$PATCH_FILE" ]; then
    cd "$LLAMA_DIR"
    if patch -p1 --dry-run < "$PATCH_FILE" > /dev/null 2>&1; then
        echo "🔧 Applying lcpp.patch..."
        patch -p1 < "$PATCH_FILE"
    else
        echo "✅ Patch already applied or incompatible."
    fi
    cd "$PROJECT_DIR"
else
    echo "⚠️ lcpp.patch not found in $PROJECT_DIR. Skipping patch step."
fi

# --- 4. BUILD BINARIES ---
if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    echo "🔨 Building llama.cpp with CUDA support..."
    cd "$LLAMA_DIR"
    mkdir -p build && cd build
    cmake .. -DGGML_CUDA=ON
    cmake --build . --config Release -j$(nproc)
    cd "$PROJECT_DIR"
    echo "✅ Build complete."
else
    echo "✅ Binaries already exist. Skipping build."
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
