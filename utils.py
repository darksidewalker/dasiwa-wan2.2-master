import psutil, torch, os, subprocess

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