# 🌀 DaSiWa WAN 2.2 Master: Merger & Quantizer

**DaSiWa WAN 2.2 Master** is a high-performance toolkit designed for merging and quantizing WAN 2.2 Video Models. It is specifically optimized for systems with 64GB RAM and NVIDIA 40-series/50-series (Blackwell) GPUs, leveraging a high-speed 55GB RAMDisk to minimize SSD wear and maximize throughput.

## ✨ Key Features

🧬 Model Merger: Advanced JSON-based merging using the Master-Engine. Supports multi-pass pipelines and custom recipe loading.

📦 GGUF Quantizer: Specialized WAN 2.2 quantization with Self-Healing 5D logic to preserve video tensor shapes.

💎 FP Quants (FP8/NV4): Integration with convert_to_quant and comfy-kitchen for native Blackwell (NVFP4/MXFP8) and Ada (FP8 E4M3) formats.

⚡ RAMDisk Integration: Direct I/O to /mnt/ramdisk for lightning-fast conversions and reduced SSD latency.

🛡️ 64GB Safety Logic: Intelligent memory management that prevents system crashes by monitoring RAMDisk overhead.

### 🚀 Quick Start
1. Prerequisites

Ensure you have uv installed for high-speed dependency management:

Bash
`curl -LsSf https://astral.sh/uv/install.sh | sh`

2. Installation & Launch

The included start.sh handles RAMDisk mounting, llama.cpp building, and dependency syncing automatically.
Bash
```
chmod +x start.sh
./start.sh
```
3. Directory Structure

    models/: Place your .safetensors base models here.

    recipes/: Store your .json merge configurations here.

    /mnt/ramdisk/: Used as the workspace for active quantizations.

### 🛠️ Quantization Guide
| Format | Target | Hardware |
|-------:|-------:|---------:|
GGUF | (Q2-Q8) AMD & old NV | and VRAM poor - Best for limited deployment.|
FP8 | (E4M3) RTX 40-series | Native Ada acceleration; best quality/speed balance.|
NVFP4 | RTX 50-series | Blackwell native 4-bit; maximum VRAM savings.|
MXFP8 | RTX 50-series | Microscaled 8-bit; near-lossless video quality.|

⚠️ Memory Management for 64GB Systems

Because the 55GB RAMDisk consumes a large portion of system memory, follow these rules:

1. Clear First: Use the "Clear RAMDisk" button before starting a new FP8 conversion if a large model is already present.
2. Move to SSD: Once a conversion is finished, click "Move RD Models to SSD" to free up system RAM for inference.
3. Vitals Monitor: Keep an eye on the Vitals bar at the top of the GUI to ensure RAM usage stays below 90%.

### 🤝 Credits
Quantization: 
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [City96](https://github.com/city96/ComfyUI-GGUF/tree/main/tools)

Utilities: 
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) for Blackwell/NVFP4 support.
