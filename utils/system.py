# utils/system.py
import psutil
import torch
import platform
import subprocess

def get_sys_info():
    # 1. Physical RAM & CPU
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    
    # 2. Swap / Pagefile Logic
    swap = psutil.swap_memory()
    swap_used_gb = swap.used / (1024**3)
    swap_total_gb = swap.total / (1024**3)
    swap_percent = swap.percent
    
    gpu_load = "N/A"
    vram_info = "0.0/0.0GB"
    
    # 3. GPU Logic (NVIDIA, AMD, Intel)
    if torch.cuda.is_available():
        try:
            # Gets hardware-level Free/Total VRAM
            free_b, total_b = torch.cuda.mem_get_info()
            used_gb = (total_b - free_b) / (1024**3)
            total_gb = total_b / (1024**3)
            vram_info = f"{used_gb:.1f}/{total_gb:.1f}GB"
            
            # Get GPU Utilization
            if "nvidia" in torch.cuda.get_device_name(0).lower():
                # Attempt to get real load via torch internal
                gpu_load = f"{torch.cuda.utilization():>3}%"
            else:
                gpu_load = "Active"
        except Exception:
            gpu_load = "ERR%"

    # 4. Formatting the Output for the UI
    # Line 1: CPU and Physical RAM
    # Line 2: Swap / Pagefile (Critical for low-VRAM offloading)
    # Line 3: GPU & VRAM
    return (
        f"🖥️ CPU: {cpu:>3}% | RAM: {ram:>3}% | OS: {platform.system()[:3]}\n"
        f"🔄 SWAP: {swap_percent:>2}% ({swap_used_gb:.1f}/{swap_total_gb:.1f}GB)\n"
        f"📟 GPU: {gpu_load} | VRAM: {vram_info}"
    )