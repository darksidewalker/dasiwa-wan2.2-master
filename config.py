import os
import torch
import psutil

# --- HARDWARE OPTIMIZATIONS ---
# Enable TF32 for ~2x faster math on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- DIRECTORIES ---
MODELS_DIR = "models"
RECIPES_DIR = "recipes"
RAMDISK_PATH = "/mnt/ramdisk"
LOGS_DIR = "logs"

# --- UI ASSETS (Gradio 6 Optimized) ---
CSS_STYLE = """
#terminal textarea { 
    background-color: #0d1117 !important; 
    color: #00ff41 !important; 
    font-family: 'Fira Code', monospace !important; 
    font-size: 13px !important;
}
.vitals-card { 
    border: 1px solid #30363d; 
    padding: 15px; 
    border-radius: 8px; 
    background: #0d1117; 
}
.primary-button {
    background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%) !important;
    color: white !important;
    border: 1px solid #145523 !important;
    font-weight: bold !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

.primary-button:hover {
    background: #218838 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Keep the Stop button Red */
.stop-button {
    background: #dc3545 !important;
    color: white !important;
}
"""

JS_AUTO_SCROLL = """
() => {
    const el = document.querySelector('#terminal textarea');
    if (el) { el.scrollTop = el.scrollHeight; }
}
"""

def get_ram_threshold_met():
    """Returns True if available RAM is below 15% (Critical for 14B models)"""
    return psutil.virtual_memory().percent > 85

def ensure_dirs():
    for d in [MODELS_DIR, RECIPES_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)