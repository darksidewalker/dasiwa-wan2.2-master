import psutil
import torch
import os
import subprocess
import shutil
import re
import json
import gradio as gr
from config import RAMDISK_PATH, MODELS_DIR, RECIPES_DIR

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
    
    rd_status = "💾 RD: [OFFLINE]"
    if os.path.exists(RAMDISK_PATH):
        try:
            usage = psutil.disk_usage(RAMDISK_PATH)
            rd_status = f"💾 RD: {usage.used/1e9:.1f}/{usage.total/1e9:.1f}GB"
        except: rd_status = "💾 RD: [ERR]"
    return f"🖥️ CPU: {cpu}% | RAM: {ram}%\n📟 GPU: {gpu_load} | VRAM: {vram_info}\n{rd_status}"

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
    """
    Standalone version of the Extended Summary logic for utils.py.
    Pass 'engine.summary_data' and 'engine.role_label' as arguments.
    """
    lines = ["\n" + "="*85, f"📊 FINAL MERGE SUMMARY: {role_label}", "="*85]
    lines.append(f"{'PASS NAME':<15} | {'METHOD':<10} | {'LAYERS':<8} | {'KNOWLEDGE %':<12} | {'PEAKS':<6} | {'SHIFT'}")
    lines.append("-" * 85)
    total_delta = 0
    for s in summary_data:
        lines.append(f"{s['pass']:<15} | {s['method']:<10} | {s['layers']:<8} | {s['inj']:>10.1f}% | {s['peaks']:<6} | {s['delta']:.8f}")
        total_delta += s['delta']
    
    lines.append("-" * 85)
    # Extended Stability Tiers preserved from your original code
    status = "STABLE" if total_delta < 0.015 else ("SATURATED" if total_delta < 0.030 else "VOLATILE")
    lines.append(f"{'TOTAL MODEL SHIFT':<52} | {total_delta:.8f}")
    lines.append(f"{'STABILITY CHECK':<52} | {status}")
    lines.append("="*85 + "\n")
    return "\n".join(lines)