# 🌀 DaSiWa WAN 2.2 Master: Merger & Quantizer

DaSiWa WAN 2.2 Master is a high-performance industrial toolkit designed for merging and quantizing Wan 2.2 (14B) Video Models. Specifically engineered for systems with 64GB RAM and NVIDIA Ada (40-series) or Blackwell (50-series) GPUs, it leverages a 55GB RAMDisk to eliminate SSD bottlenecking and maximize tensor throughput.

## ✨ Key Features

🧬 Action-Master Engine: Adaptive Injection Pipeline - The Action-Master Engine is a high-precision merging framework specifically architected for Wan 2.2 (14B). It moves beyond simple weight stacking by treating LoRAs as conceptual data streams that are filtered, gated, and injected based on the base model’s existing knowledge.

**Key Features of the New Approach:**

- Dual-Path Processing: * addition: High-fidelity raw transfer for Distilled or Foundation LoRAs where 100% precision is required.
    - `injection` Activates the custom Adaptive Noise Gate and Dynamic Limiter for complex concept integration.

- Adaptive Knowledge Injection: The engine automatically detects "Conceptual Gaps." If the LoRA introduces a new texture (like NSFW fluids or specific anatomy) that the   base model lacks, it lowers the sensitivity gate to allow deep carving of the new weights.

- Intelligent Noise Gating: For established concepts (lighting, style, generic motion), the engine applies a strict gate to strip away "latent chatter," ensuring the merge remains sharp and artifact-free.

- Saturation Guard (Dynamic Limiter): Prevents "deep-fried" or over-saturated videos by monitoring weight variance and capping peak intensities before they destabilize the 14B architecture.

- Universal Key Mapping: A normalized "slug-based" matching system that allows seamless merging regardless of whether the LoRAs were trained with dots, underscores, or different naming conventions.

- Real-Time Injection Summary: Provides a CLI-based health report for every pass, calculating Knowledge Injection % (Sparsity) and Total Model Shift to ensure stability before final quantization.

📦 GGUF MoE Specialist: Native Wan 2.2 GGUF quantization with Self-Healing 5D Expert Injection to preserve video tensor shapes and prevent "gray-screen" outputs.

💎 Next-Gen FP Quants: Native support for NVFP4 (Blackwell) and FP8 E4M3 (Ada) via optimized convert_to_quant integration.

⚡ RAMDisk-First I/O: Maps all heavy-lift operations to /mnt/ramdisk to protect SSD health and increase conversion speed by up to 4x.

🛡️ 64GB Safety Logic: Intelligent memory flushing and subprocess monitoring to prevent OOM (Out of Memory) crashes during 14B model handling.

## 🚀 Quick Start

### Prerequisites

Ensure you have uv installed for high-speed dependency management:
Bash
```
curl -LsSf https://astral.sh/uv/install.sh | sh
``` 

### Installation & Launch

The included start.sh handles RAMDisk mounting, environment syncing, and build requirements automatically.
Bash
```
chmod +x start.sh
./start.sh
```

#### 🛠️ Quantization Guide

|Format|Target|Hardware|Recommendation|
|-----:|-----:|-------:|-------------:|
| GGUF (Q2-Q8) | Universal / CPU | Best for VRAM-constrained systems (8GB - 12GB).
| FP8 (E4M3) | RTX 40-Series | Native Ada acceleration; best quality/speed balance.
| NVFP4 | RTX 50-Series | Blackwell native 4-bit; extreme VRAM savings for 14B models.
| MXFP8 | RTX 50-Series | Microscaled 8-bit; near-lossless video quality.

### 📂 Directory Structure

    models/: Place your .safetensors base models and LoRAs here.

    recipes/: Store your .json merge configurations here.

    logs/: Automated session logs for debugging merge weights.

    /mnt/ramdisk/: Active workspace for quantization (volatile).

### ⚠️ Memory Management for 64GB Systems

The 55GB RAMDisk consumes a large portion of system memory. To ensure stability:

    Intermediate Cache: The engine saves a high-precision BF16 Master to SSD before quantizing to RAMDisk. If a quant fails, you can restart from this cache.

    Vitals Monitor: Always monitor the Station Health bar in the GUI. If RAM usage exceeds 92%, terminate the process using the 🛑 STOP button.

    Sync to SSD: Once a conversion is successful, use the "Manual Move to SSD" button to free up RAMDisk space before starting the next run.

### 🤝 Credits
Quantization: 
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [City96](https://github.com/city96/ComfyUI-GGUF/tree/main/tools)

Utilities: 
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) for Blackwell/NVFP4 support.
