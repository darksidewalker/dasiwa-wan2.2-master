import gradio as gr
import psutil
import torch
import os
import json
import re
import gc
import subprocess
import shutil
import time
import datetime
from engine import ActionMasterEngine

# --- 1. CONFIGURATION & DIRECTORIES ---
MODELS_DIR = "models"
RECIPES_DIR = "recipes"
RAMDISK_PATH = "/mnt/ramdisk"
LOGS_DIR = "logs"

for d in [MODELS_DIR, RECIPES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 2. THE TERMINAL & VITALS STYLE (EXTENDED) ---
CSS_STYLE = """
#terminal textarea { 
    background-color: #0d1117 !important; 
    color: #00ff41 !important; 
    font-family: 'Fira Code', 'Cascadia Code', monospace !important; 
    font-size: 13px !important;
    line-height: 1.5 !important;
    border: 1px solid #30363d !important;
}
#terminal textarea::-webkit-scrollbar { width: 10px; }
#terminal textarea::-webkit-scrollbar-track { background: #0d1117; }
#terminal textarea::-webkit-scrollbar-thumb { background: #238636; border-radius: 5px; }
.vitals-card { 
    border: 1px solid #444; 
    padding: 15px; 
    border-radius: 12px; 
    background: #0a0a0a;
    box-shadow: inset 0 0 10px #000;
}
.vitals-card textarea {
    background-color: transparent !important;
    color: #00ff41 !important;
    font-family: 'Fira Code', monospace !important;
    border: none !important;
    font-size: 14px !important;
    resize: none !important;
}
.sync-box {
    margin-top: 15px;
    padding: 12px;
    border: 1px dashed #555;
    border-radius: 8px;
    background: #161b22;
}
"""

JS_AUTO_SCROLL = """
() => {
    const textarea = document.querySelector('#terminal textarea');
    if (textarea) { textarea.scrollTop = textarea.scrollHeight; }
}
"""

# --- 3. CORE UTILITIES ---

def get_sys_info():
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    vram = f"{torch.cuda.memory_reserved()/1e9:.1f}GB" if torch.cuda.is_available() else "0.0G"
    rd_status = "💾 RD: [OFFLINE]"
    if os.path.exists(RAMDISK_PATH):
        try:
            usage = psutil.disk_usage(RAMDISK_PATH)
            rd_status = f"💾 RD:  {usage.used/1e9:>5.1f} / {usage.total/1e9:.1f}GB"
        except:
            rd_status = "💾 RD: [LOCKED]"
    return f"🖥️ CPU: {cpu:>5}% | RAM: {ram:>3}%\n📟 VRAM: {vram:>9}\n{rd_status}"

def list_files():
    try:
        m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.bin', '.gguf'))])
        r = sorted([f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')])
        return gr.update(choices=m), gr.update(choices=r)
    except:
        return gr.update(choices=[]), gr.update(choices=[])

def load_recipe_text(name):
    if not name: return ""
    with open(os.path.join(RECIPES_DIR, name), 'r') as f:
        return f.read()

def sync_ram_to_ssd(path):
    if not path or not os.path.exists(path): return "❌ SYNC ERROR: Path invalid."
    dest = os.path.join(MODELS_DIR, os.path.basename(path))
    try:
        shutil.move(path, dest)
        return f"✅ SYNC SUCCESS: {os.path.basename(dest)} moved to SSD."
    except Exception as e: return f"❌ SYNC FAILED: {str(e)}"

# --- 4. THE MASTER PIPELINE ---

def run_pipeline(recipe_json, base_model, q_format, recipe_name, auto_move, progress=gr.Progress()):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "-"*60 + "\n"
    
    try:
        # STAGE 1: PARSING & INITIALIZATION
        if not base_model or not recipe_name:
            raise ValueError("Base model or Recipe name missing.")

        progress(0.05, desc="🧬 Parsing Recipe...")
        clean_json = re.sub(r'//.*', '', recipe_json)
        recipe_dict = json.loads(clean_json)
        recipe_dict['paths'] = recipe_dict.get('paths', {})
        recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
        
        engine = ActionMasterEngine(recipe_dict)
        log_acc += f"🧬 ENGINE: {engine.role_label} Architecture Detected.\n"

        # --- ENHANCED DYNAMIC CACHE LOGIC ---
        is_gguf = q_format.startswith("GGUF_")
        
        # Create a unique slug: ModelName_RecipeName
        model_slug = os.path.splitext(base_model)[0][:12]
        recipe_slug = recipe_name.replace(".json", "")
        
        # File: e.g., "cache_Wan2.2-14B_HighRes_flattened.safetensors"
        cache_name = f"cache_{model_slug}_{recipe_slug}_{'flattened' if is_gguf else 'native'}.safetensors"
        temp_path = os.path.join(MODELS_DIR, cache_name)

        # CHECK FOR EXISTING INTERMEDIATE
        if os.path.exists(temp_path):
            log_acc += f"♻️ CACHE HIT: Found intermediate for [{recipe_slug}] at {temp_path}\n"
            log_acc += "⏭️ SKIPPING MERGE: Using existing FP16 data for new quantization.\n"
            yield log_acc, temp_path
        else:
            # STAGE 2: THE MERGE LOOP
            pipeline = recipe_dict.get('pipeline', [])
            for i, step in enumerate(pipeline):
                p_name = step.get('pass_name', f"Pass {i+1}")
                progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}...")
                log_acc += f"▶️ EXECUTING: {p_name}...\n"
                engine.process_pass(step, 1.0)
                yield log_acc, ""

            # STAGE 3: DATA PREPARATION (Using Dynamic Path)
            progress(0.8, desc="📐 Preparing Tensors...")
            log_acc += "-"*60 + "\n"
            if is_gguf:
                log_acc += f"📐 GGUF DETECTED: Flattening to {temp_path} (SSD)...\n"
                temp_path = engine.save_and_patch(use_ramdisk=False, custom_path=temp_path)
            else:
                log_acc += f"💎 FP8 DETECTED: Saving Native 5D to {temp_path} (SSD)...\n"
                temp_path = engine.save_pure_5d(use_ramdisk=False, custom_path=temp_path)
            
            log_acc += f"✅ INTERMEDIATE SAVED: {temp_path}\n"
            yield log_acc, temp_path

        # STAGE 4: QUANTIZATION CLI HANDOFF
        progress(0.9, desc=f"🏗️ Conversion: {q_format}...")
        
        out_prefix = recipe_dict['paths'].get('output_prefix', 'Wan22_Output')
        if is_gguf:
            q_type = q_format.replace("GGUF_", "")
            final_output_path = f"{out_prefix}_{recipe_slug}_{q_type}.gguf"
            cmd = ["python", "convert_to_gguf.py", temp_path, "--out", final_output_path, "--quant", q_type]
        else:
            final_output_path = f"{out_prefix}_{recipe_slug}_{q_format}.safetensors"
            fmt_flag = [] if q_format == "fp8" else [f"--{q_format}"]
            cmd = ["convert_to_quant", "-i", temp_path, "-o", final_output_path, "--comfy_quant", "--wan"] + fmt_flag

        log_acc += f"📂 TARGET PATH: {final_output_path}\n"
        log_acc += f"🖥️ CLI EXEC: {' '.join(cmd)}\n"
        yield log_acc, temp_path

        # --- ACTIVE PROCESS MONITORING ---
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        log_acc += "🚀 Quantizer Started... Monitoring logs:\n"
        for line in process.stdout:
            log_acc += f"  [QUANT] {line}"
            yield log_acc, temp_path
        
        process.wait()

        if process.returncode == 0:
            log_acc += f"\n✨ SUCCESS: Final model created at {final_output_path}\n"
            # NOTE: We NO LONGER delete temp_path so it stays for future sessions/quants
            log_acc += f"💾 PERSISTENT CACHE: {temp_path} preserved on SSD.\n"
        else:
            log_acc += f"\n❌ ERROR: Quantizer failed (Code {process.returncode}).\n"

        if auto_move and os.path.exists(final_output_path):
            log_acc += f"{sync_ram_to_ssd(final_output_path)}\n"

        log_acc += "-"*60 + "\n✨ [STATION MASTER] SESSION COMPLETE.\n"
        yield log_acc, final_output_path

    except Exception as e:
        log_acc += f"\n❌ CRITICAL ERROR: {str(e)}\n"
        yield log_acc, ""

# --- 5. UI ARCHITECTURE ---

with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**Direct-to-SSD 5D-Patching & GGUF/SVD Quantization**")
        with gr.Column(scale=2, elem_classes=["vitals-card"]):
            vitals_box = gr.Textbox(label="Environment Stats", value=get_sys_info(), lines=3, interactive=False, container=False)
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                gr.Markdown("### 📂 Asset Management")
                base_dd = gr.Dropdown(label="Base Model")
                recipe_dd = gr.Dropdown(label="Recipe (JSON)")
                refresh_btn = gr.Button("🔄 Refresh Assets")
            with gr.Group():
                gr.Markdown("### 💎 Quantization Target")
                quant_select = gr.Dropdown(choices=["fp8", "nvfp4", "GGUF_Q8_0", "GGUF_Q5_K_M", "GGUF_Q4_K_M"], value="fp8", label="Target Format")
                auto_move_toggle = gr.Checkbox(label="🚀 Auto-move to SSD on finish", value=False)
                start_btn = gr.Button("🔥 START PIPELINE", variant="primary")
            with gr.Group(elem_classes=["sync-box"]):
                gr.Markdown("### 📦 SSD Synchronization")
                last_path_state = gr.State("")
                sync_trigger = gr.Button("📤 Move RAMDisk -> SSD")
                sync_output = gr.Markdown("Status: Idle")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("💻 Live Terminal"):
                    terminal_box = gr.Textbox(label="CLI Live Stream", lines=28, interactive=False, elem_id="terminal")
                with gr.Tab("📝 Recipe Editor"):
                    recipe_editor = gr.Code(label="JSON Configuration", language="json", lines=25)

    # EVENT BINDINGS
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    
    # Ensure recipe_dd is passed to run_pipeline
    start_btn.click(
        fn=run_pipeline, 
        inputs=[recipe_editor, base_dd, quant_select, recipe_dd, auto_move_toggle], 
        outputs=[terminal_box, last_path_state],
    ).then(fn=None, js=JS_AUTO_SCROLL)

    sync_trigger.click(fn=sync_ram_to_ssd, inputs=[last_path_state], outputs=[sync_output])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS_STYLE)