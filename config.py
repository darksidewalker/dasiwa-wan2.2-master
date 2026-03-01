import os, torch, psutil

# --- DIRECTORIES ---
MODELS_DIR = "models"
LOGS_DIR = "logs"

# --- UI ASSETS ---
CSS_STYLE = """
#terminal textarea { 
    background-color: #0d1117 !important; 
    color: #00ff41 !important; 
    font-family: 'Fira Code', monospace !important; 
    font-size: 13px !important;
}
.vitals-card { border: 1px solid #30363d; padding: 15px; border-radius: 8px; background: #0d1117; }
.primary-button {
    background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%) !important;
    color: white !important;
    font-weight: bold !important;
}
"""

JS_AUTO_SCROLL = """
(x) => {
    const el = document.getElementById('terminal');
    if (el) {
        const textarea = el.querySelector('textarea');
        if (textarea) textarea.scrollTop = textarea.scrollHeight;
    }
}
"""

def ensure_dirs():
    for d in [MODELS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)