import psutil
import torch
import os
import subprocess
import shutil
import re
import json
import gradio as gr
from config import MODELS_DIR, RECIPES_DIR

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

    return f"🖥️ CPU: {cpu}% | RAM: {ram}%\n📟 GPU: {gpu_load} | VRAM: {vram_info}\n"

def instant_validate(recipe_name, base_model):
    if not recipe_name or not base_model:
        return "### 🛡️ Status: Waiting for selection..."
    model_is_high = "high_noise" in base_model.lower()
    recipe_is_high = "high" in recipe_name.lower()
    model_label = "MOTION (High)" if model_is_high else "REFINER (Low)"
    try:
        recipe_path = os.path.join(RECIPES_DIR, recipe_name)
        with open(recipe_path, 'r') as f:
            content = f.read().lower()
            if model_is_high:
                if "low_noise" in content or "_low" in content:
                    return f"### ❌ MISMATCH: {model_label} Model vs. LOW LoRAs in JSON"
                if not recipe_is_high:
                    return f"### ❌ MISMATCH: {model_label} Model vs. LOW Recipe filename"
            else:
                if "high_noise" in content or "_high" in content:
                    return f"### ❌ MISMATCH: {model_label} Model vs. HIGH LoRAs in JSON"
                if recipe_is_high:
                    return f"### ❌ MISMATCH: {model_label} Model vs. HIGH Recipe filename"
        return f"### ✅ VALIDATED: {model_label} Architecture Alignment"
    except Exception as e:
        return f"### ⚠️ Status: Validation Error ({str(e)})"

def sync_ram_to_ssd(path):
    if not path or not os.path.exists(path): return "❌ Source missing."
    dest = os.path.join(MODELS_DIR, os.path.basename(path))
    shutil.move(path, dest)
    return f"✅ MOVED: {os.path.basename(dest)}"

def get_final_summary_string(summary_data, role_label):
    # Header Construction
    lines = ["\n" + "="*105, f"📊 14B MODEL MERGE SUMMARY: {role_label}", "="*105]
    # Added 'RANK TIER' for better tracking of the Smart Gate decisions
    lines.append(f"{'PASS NAME':<18} | {'METHOD':<10} | {'RANK TIER':<15} | {'KEPT %':<8} | {'PEAKS':<8} | {'SHIFT'}")
    lines.append("-" * 105)
    
    total_delta = 0
    for s in summary_data:
        # Pulling the new 'tier' key we added to the dictionary
        tier_label = s.get('tier', 'N/A')
        kept_val = s.get('kept', 0.0)
        
        lines.append(f"{s['pass']:<18} | {s['method']:<10} | {tier_label:<15} | {kept_val:>7.1f}% | {s['peaks']:<8} | {s['delta']:.8f}")
        total_delta += s['delta']
    
    lines.append("-" * 105)

    # RECALIBRATED STABILITY FOR WAN 2.2 14B
    # These thresholds ensure the Mixture of Experts (MoE) doesn't collapse
    match total_delta:
        case d if d < 0.002: status, icon = "STABLE (EXCELLENT)", "✅"
        case d if d < 0.008: status, icon = "SATURATED (HEAVY)", "⚠️"
        case d if d < 0.015: status, icon = "VOLATILE (SENSITIVE)", "🔥"
        case _:              status, icon = "CRITICAL (EXPLODED)", "💀"

    lines.append(f"{'TOTAL CUMULATIVE MODEL SHIFT':<76} | {total_delta:.8f}")
    lines.append(f"{'STABILITY CHECK (14B CALIBRATION)':<76} | {icon} {status}")
    lines.append("="*105 + "\n")
    return "\n".join(lines)