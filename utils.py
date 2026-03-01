import psutil
import torch
import os
import subprocess
import re
import json

def get_sys_info():
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    gpu_load, vram_info = "0%", "0.0/0.0GB"
    if torch.cuda.is_available():
        try:
            v_used = torch.cuda.memory_reserved() / 1e9
            v_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_info = f"{v_used:.1f}/{v_total:.1f}GB"
            res = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], encoding='utf-8')
            gpu_load = f"{res.strip()}%"
        except: gpu_load = "ERR%"
    return f"🖥️ CPU: {cpu}% | RAM: {ram}%\\n📟 GPU: {gpu_load} | VRAM: {vram_info}"

def instant_validate(recipe_name, base_model):
    if not recipe_name or not base_model:
        return "### 🛡️ Status: Waiting..."
    m_high = "high_noise" in base_model.lower()
    r_high = "high" in recipe_name.lower()
    if m_high != r_high:
        return "### ⚠️ WARNING: Noise Mismatch detected!"
    return "### ✅ VALIDATED: Ready."

def get_final_summary_string(summary_data, role_label):
    lines = ["="*105, f"{'SECTION':<40} | {'LAYERS':<10} | {'SHIFT':<20}", "-"*105]
    total_delta = 0
    for s in summary_data:
        lines.append(f"{s['pass']:<40} | {s['layers']:<10} | {s['delta']:>18.8f}")
        total_delta += s['delta']
    lines.append("-" * 105)
    lines.append(f"TOTAL SHIFT: {total_delta:.8f} | ROLE: {role_label}")
    return "\n".join(lines)

def verify_model_integrity(modified_tensors):
    """Stand-alone diagnostic updated for 14B ActionMaster Engine"""
    yield "  🛡️ INTEGRITY CHECK..."
    for key, tensor in modified_tensors.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            yield f"  ❌ Error in {key}"