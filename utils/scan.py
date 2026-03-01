import torch
from safetensors.torch import load_file
import os

def scan_model_structure(model_path, output_log="model_structure_report.txt"):
    with open(output_log, "w", encoding="utf-8") as f:
        # Scan Base Model
        f.write(f"=== BASE MODEL SCAN: {os.path.basename(model_path)} ===\n")
        base_sd = load_file(model_path)
        keys = list(base_sd.keys())
        f.write(f"Total Keys: {len(keys)}\n")
        f.write("First 20 Keys:\n")
        for k in keys[:20]:
            f.write(f"  {k} | Shape: {list(base_sd[k].shape)}\n")
        f.write("\n... [Skipped Middle] ...\n\n")
        f.write("Last 20 Keys:\n")
        for k in keys[-20:]:
            f.write(f"  {k} | Shape: {list(base_sd[k].shape)}\n")
        f.write("\n" + "="*50 + "\n\n")
        del base_sd

        # Scan LoRA Directory
        lora_dir = "loras/WAN22/F/" # Change this to your actual path
        f.write(f"=== LORA DIRECTORY SCAN ===\n")
        for filename in os.listdir(lora_dir):
            if filename.endswith(".safetensors"):
                f.write(f"FILE: {filename}\n")
                try:
                    l_sd = load_file(os.path.join(lora_dir, filename))
                    l_keys = list(l_sd.keys())
                    # Just sample a few to see the naming convention
                    for k in l_keys[:10]:
                        f.write(f"  {k} | Shape: {list(l_sd[k].shape)}\n")
                    f.write("-" * 30 + "\n")
                    del l_sd
                except Exception as e:
                    f.write(f"  ❌ Error reading {filename}: {e}\n")

    print(f"✅ Scan Complete. Analysis saved to {output_log}")

# Run this once
scan_model_structure("models/wan2.2_i2v_high_noise_14B_fp16.safetensors")