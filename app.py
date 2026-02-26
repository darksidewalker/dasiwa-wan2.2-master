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
#terminal textarea::-webkit-scrollbar-thumb:hover { background: #2ea043; }

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
    line-height: 1.7 !important;
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
    """Deep hardware monitoring including RAMDisk and CPU load."""
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    vram = f"{torch.cuda.memory_reserved()/1e9:.1f}GB" if torch.cuda.is_available() else "0.0G"
    
    rd_status = "RD: [OFFLINE]"
    if os.path.exists(RAMDISK_PATH):
        try:
            usage = psutil.disk_usage(RAMDISK_PATH)
            rd_status = f"💾 RD:  {usage.used/1e9:>5.1f} / {usage.total/1e9:.1f}GB"
        except:
            rd_status = "RD: [LOCKED]"
            
    return f"🖥️ CPU: {cpu:>5}% | RAM: {ram:>3}%\n📟 VRAM: {vram:>9}\n{rd_status}"

def list_files():
    """Rescans filesystem for assets."""
    try:
        m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.bin'))])
        r = sorted([f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')])
        return gr.update(choices=m), gr.update(choices=r)
    except Exception as e:
        return gr.update(choices=[]), gr.update(choices=[])

def load_recipe_text(name):
    if not name: return ""
    try:
        with open(os.path.join(RECIPES_DIR, name), 'r') as f:
            return f.read()
    except Exception as e:
        return f"// Error loading recipe: {str(e)}"

def sync_ram_to_ssd(ram_path):
    if not ram_path or not os.path.exists(ram_path):
        return "❌ SYNC ERROR: Path invalid or file missing from RAMDisk."
    
    filename = os.path.basename(ram_path)
    dest = os.path.join(MODELS_DIR, filename)
    
    start_time = time.time()
    try:
        shutil.move(ram_path, dest)
        duration = time.time() - start_time
        return f"✅ SYNC SUCCESS: {filename} moved to SSD in {duration:.1f}s"
    except Exception as e:
        return f"❌ SYNC FAILED: {str(e)}"

# --- 4. THE MASTER PIPELINE (UNABRIDGED) ---

def run_pipeline(recipe_json, base_model, q_format, auto_move, progress=gr.Progress()):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER INITIALIZING...\n"
    log_acc += f"[{timestamp}] 🛠️ COMPONENT: v2.2-Wan Master\n"
    log_acc += "-"*60 + "\n"
    
    final_output_path = ""
    
    try:
        if not base_model:
            raise ValueError("Execution halted: No base model selected.")

        # STAGE 1: PARSING & INITIALIZATION
        progress(0.05, desc="🧬 Parsing Recipe...")
        clean_json = re.sub(r'//.*', '', recipe_json)
        try:
            recipe_dict = json.loads(clean_json)
        except json.JSONDecodeError as je:
            raise ValueError(f"JSON Syntax Error: {str(je)}")

        recipe_dict['paths'] = recipe_dict.get('paths', {})
        recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
        
        log_acc += f"📦 LOADING BASE: {base_model}\n"
        engine = ActionMasterEngine(recipe_dict)
        log_acc += f"🧬 ENGINE: {'High-Res' if engine.is_high_res else 'Low-Res'} Architecture Detected.\n"
        yield log_acc, ""

        # STAGE 2: THE MERGE LOOP
        pipeline = engine.recipe.get('pipeline', [])
        for i, step in enumerate(pipeline):
            p_name = step.get('name', f"Pass {i+1}")
            progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}...")
            
            log_acc += f"▶️ EXECUTING: {p_name}...\n"
            conflicts = engine.process_pass(step, 1.0)
            
            # Conflict Heatmap Analysis
            early, mid, late = 0, 0, 0
            for c in conflicts:
                m = re.search(r'blocks\.(\d+)\.', c)
                if m:
                    blk = int(m.group(1))
                    if blk <= 8: early += 1
                    elif 9 <= blk <= 30: mid += 1
                    else: late += 1
            
            log_acc += f"  └ Conflicts -> E:[{early}] M:[{mid}] L:[{late}]\n"
            log_acc += f"  └ VRAM Usage: {torch.cuda.memory_allocated()/1e9:.2f}GB\n"
            
            torch.cuda.empty_cache()
            gc.collect()
            yield log_acc, ""

        # STAGE 3: 5D PATCHING & RAMDISK DUMP
        progress(0.8, desc="📐 Patching 5D Tensors...")
        log_acc += "-"*60 + "\n"
        log_acc += "📐 STAGE 3: RESHAPING 5D VIDEO TENSORS TO 4D INTERMEDIATE...\n"
        log_acc += "💾 INJECTING: ComfyUI metadata & orig_shape keys...\n"
        
        temp_ram_path = engine.save_and_patch(use_ramdisk=True)
        log_acc += f"✅ INTERMEDIATE SAVED: {temp_ram_path}\n"
        yield log_acc, ""

        # STAGE 4: QUANTIZATION CLI HANDOFF
        progress(0.9, desc=f"🏗️ Conversion: {q_format}...")
        log_acc += f"🏗️ STAGE 4: COMMENCING {q_format.upper()} QUANTIZATION...\n"
        
        if q_format.startswith("GGUF_"):
            q_type = q_format.replace("GGUF_", "")
            final_output_path = temp_ram_path.replace(".safetensors", f"_{q_type}.gguf")
            cmd = ["python", "convert_to_gguf.py", temp_ram_path, "--out", final_output_path, "--quant", q_type]
        else:
            final_output_path = temp_ram_path.replace("_PATCHED.safetensors", f"_{q_format}.safetensors")
            # Default to FP8 if format is blank
            fmt_flag = [] if q_format == "fp8" else [f"--{q_format}"]
            cmd = ["convert_to_quant", "-i", temp_ram_path, "-o", final_output_path, "--comfy_quant", "--wan"] + fmt_flag

        log_acc += f"🖥️ CLI EXEC: {' '.join(cmd)}\n"
        yield log_acc, ""

        # Subprocess Management with real-time stream
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ""):
            log_acc += line
            yield log_acc, ""

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Quantization failed with exit code {process.returncode}")

        # STAGE 5: POST-PROCESS & CLEANUP
        if os.path.exists(temp_ram_path):
            os.remove(temp_ram_path)
            log_acc += "🧹 CLEANUP: Intermediate FP16 purged from RAMDisk.\n"

        if auto_move:
            log_acc += "🚀 AUTO-SYNC: Moving to persistent SSD...\n"
            sync_result = sync_ram_to_ssd(final_output_path)
            log_acc += f"{sync_result}\n"
        
        log_acc += "-"*60 + "\n"
        log_acc += "✨ [STATION MASTER] ALL TASKS COMPLETED SUCCESSFULLY.\n"
        yield log_acc, final_output_path

    except Exception as e:
        err_msg = f"\n❌ CRITICAL PIPELINE ERROR: {str(e)}\n"
        log_acc += err_msg
        yield log_acc, ""

# --- 5. UI ARCHITECTURE (GRADIO 6.0 COMPLIANT) ---

# REMOVED css=CSS_STYLE from here
with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**Direct-to-RAM 5D-Patching & GGUF/SVD Quantization**")
        with gr.Column(scale=2, elem_classes=["vitals-card"]):
            vitals_box = gr.Textbox(
                label="Environment Stats", 
                value=get_sys_info(), 
                lines=3, 
                interactive=False, 
                container=False
            )
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                gr.Markdown("### 📂 Asset Management")
                base_dd = gr.Dropdown(label="Base Model (Safetensors)")
                recipe_dd = gr.Dropdown(label="Recipe (JSON)")
                refresh_btn = gr.Button("🔄 Refresh Asset Directories")
            
            with gr.Group():
                gr.Markdown("### 💎 Quantization Target")
                quant_select = gr.Dropdown(
                    choices=[
                        "fp8", "nvfp4", "mxfp8", "int8",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", 
                        "GGUF_Q4_K_M", "GGUF_Q3_K_L", "GGUF_Q2_K"
                    ], 
                    value="fp8", 
                    label="Target Format"
                )
                auto_move_toggle = gr.Checkbox(label="🚀 Auto-move to SSD on finish", value=False)
                start_btn = gr.Button("🔥 START PIPELINE", variant="primary")
            
            with gr.Group(elem_classes=["sync-box"]):
                gr.Markdown("### 📦 SSD Synchronization")
                last_path_state = gr.State("")
                sync_trigger = gr.Button("📤 Move RAMDisk -> SSD")
                sync_output = gr.Markdown("Status: Idle")

        with gr.Column(scale=2):
            with gr.Tabs() as work_tabs:
                # 1. Terminal is FIRST and ACTIVE by default
                with gr.Tab("💻 Live Terminal", id="terminal_tab"):
                    terminal_box = gr.Textbox(
                        label="CLI Live Stream", 
                        lines=28, 
                        interactive=False, 
                        elem_id="terminal"
                    )
                # 2. Editor is SECOND
                with gr.Tab("📝 Recipe Editor", id="editor_tab"):
                    recipe_editor = gr.Code(
                        label="JSON Configuration", 
                        language="json", 
                        lines=25
                    )

    # --- 6. EVENT BINDINGS ---
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    
    recipe_dd.change(
        fn=load_recipe_text, 
        inputs=[recipe_dd], 
        outputs=[recipe_editor]
    )
    
    # FIXED: Changed _js to js
    start_btn.click(
        fn=run_pipeline, 
        inputs=[recipe_editor, base_dd, quant_select, auto_move_toggle], 
        outputs=[terminal_box, last_path_state],
    ).then(fn=None, js=JS_AUTO_SCROLL) 

    sync_trigger.click(
        fn=sync_ram_to_ssd, 
        inputs=[last_path_state], 
        outputs=[sync_output]
    )

# --- 7. LAUNCH (CSS MOVED HERE) ---
if __name__ == "__main__":
    # CSS must be passed here in Gradio 6.0
    demo.launch(theme=gr.themes.Soft(), css=CSS_STYLE)