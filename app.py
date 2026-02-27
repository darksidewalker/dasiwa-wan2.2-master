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

active_process = None

for d in [MODELS_DIR, RECIPES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 2. STYLING (Gradio 6 Optimized) ---
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
"""

JS_AUTO_SCROLL = """
() => {
    const el = document.querySelector('#terminal textarea');
    if (el) { el.scrollTop = el.scrollHeight; }
}
"""

# --- 3. SYSTEM UTILITIES ---
def get_sys_info():
    """Gradio 6.0 compatible system health stream."""
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

def list_files():
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.gguf'))])
    r = sorted([f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')])
    return gr.update(choices=m), gr.update(choices=r)

def load_recipe_text(name):
    if not name: return ""
    with open(os.path.join(RECIPES_DIR, name), 'r') as f: return f.read()

def sync_ram_to_ssd(path):
    if not path or not os.path.exists(path): return "❌ Source missing."
    dest = os.path.join(MODELS_DIR, os.path.basename(path))
    shutil.move(path, dest)
    return f"✅ MOVED: {os.path.basename(dest)}"

def terminate_pipeline():
    global active_process
    if active_process:
        active_process.terminate()
        active_process = None
        return "🛑 EMERGENCY STOP: Pipeline Terminated."
    return "ℹ️ Idle."

# --- 4. THE MASTER PIPELINE ---
def run_pipeline(recipe_json, base_model, q_format, recipe_name, auto_move, progress=gr.Progress()):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "="*60 + "\n"
    global active_process
    
    try:
        # 1. SETUP
        progress(0.05, desc="Initializing Engine...")
        clean_json = re.sub(r'//.*', '', recipe_json)
        recipe_dict = json.loads(clean_json)
        recipe_dict['paths'] = recipe_dict.get('paths', {})
        recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
        
        engine = ActionMasterEngine(recipe_dict)
        log_acc += f"🧬 ENGINE: {engine.role_label} Architecture Detected.\n"

        is_gguf = q_format.startswith("GGUF_")
        recipe_slug = recipe_name.replace(".json", "")
        mode_suffix = "flattened" if is_gguf else "native"
        
        # 2. CACHE LOGIC
        model_slug = os.path.splitext(base_model)[0][:8]
        cache_name = f"cache_{model_slug}_{recipe_slug}_{mode_suffix}.safetensors"
        temp_path = os.path.join(MODELS_DIR, cache_name)

        if os.path.exists(temp_path):
            log_acc += f"♻️ CACHE HIT: Reusing {cache_name}\n"
            yield log_acc, temp_path
        else:
            # 3. MERGING PASSES
            pipeline = recipe_dict.get('pipeline', [])
            for i, step in enumerate(pipeline):
                p_name = step.get('pass_name', f"Pass {i+1}")
                progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}")
                log_acc += f"▶️ {p_name.upper()}: Merging Tensors...\n"
                engine.process_pass(step, 1.0)
                yield log_acc, ""

            # STAGE 4: TENSOR PATTERN PREP
            progress(0.8, desc="Saving Master...")
            
            # This is the line to change for the GUI:
            log_acc += f"💾 EXPORT: Saving high-precision Master (BF16) to SSD...\n"
            
            engine.save_master(temp_path)
            log_acc += f"✅ INTERMEDIATE SAVED: {cache_name}\n"
            yield log_acc, temp_path
        
        # 5. QUANTIZATION
        progress(0.9, desc=f"Quantizing to {q_format}")
        output_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
        out_prefix = recipe_dict['paths'].get('output_prefix', 'Wan22_Merge')
        
        if is_gguf:
            q_type = q_format.replace("GGUF_", "")
            final_output_path = os.path.join(output_dir, f"{out_prefix}_{recipe_slug}_{q_type}.gguf")
            cmd = ["python", "convert.py", "--path", temp_path, "--dst", final_output_path]
            log_acc += "📦 GGUF: Converting to F16 (Experts will be extracted)...\n"
        else:
            final_output_path = os.path.join(output_dir, f"{out_prefix}_{recipe_slug}_{q_format}.safetensors")
            fmt_flag = ["--nvfp4"] if q_format == "nvfp4" else (["--int8"] if q_format == "int8" else [])
            cmd = ["convert_to_quant", "-i", temp_path, "-o", final_output_path, "--comfy_quant", "--wan"] + fmt_flag
            log_acc += "💎 NATIVE: Using convert_to_quant --wan preset...\n"

        # 6. EXECUTION & LOG STREAMING
        active_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in active_process.stdout:
            if not any(x in line for x in ["Optimizing", "worse_count"]):
                log_acc += f"  [QUANT] {line}"
                yield log_acc, temp_path
        active_process.wait()

        # 7. GGUF EXPERT RE-INJECTION
        if is_gguf and active_process.returncode == 0:
            log_acc += "💉 Stage 7: Re-injecting 5D MoE Experts...\n"
            fix_cmd = ["python", "fix_5d_tensors.py", "--src", final_output_path, "--dst", final_output_path, "--overwrite"]
            subprocess.run(fix_cmd)
            log_acc += "✅ EXPERTS RESTORED.\n"

        if active_process.returncode == 0:
            log_acc += f"✅ SUCCESS: {os.path.basename(final_output_path)} created.\n"
            if auto_move: log_acc += f"{sync_ram_to_ssd(final_output_path)}\n"
        else:
            log_acc += f"❌ FAILED: Exit Code {active_process.returncode}\n"

        active_process = None
        yield log_acc, final_output_path

    except Exception as e:
        log_acc += f"\n🔥 CRITICAL FAILURE: {str(e)}\n"
        yield log_acc, ""

# --- 5. UI CONSTRUCTION (Gradio 6.0 Compliant) ---
with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**Gradio 6.0 Stable | 14B High-Precision MoE Pipeline**")
        with gr.Column(scale=2, elem_classes=["vitals-card"]):
            vitals_box = gr.Textbox(label="Health", value=get_sys_info(), lines=3, interactive=False)
            # Automatic timer for system stats
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                base_dd = gr.Dropdown(label="Base Component")
                recipe_dd = gr.Dropdown(label="Active Recipe")
                refresh_btn = gr.Button("🔄 Refresh Assets")
            
            with gr.Group():
                quant_select = gr.Dropdown(
                    choices=["fp8", "nvfp4", "int8", "GGUF_Q8_0", "GGUF_Q4_K_M", "GGUF_Q2_K"], 
                    value="fp8", 
                    label="Format"
                )
                auto_move_toggle = gr.Checkbox(label="🚀 Move to SSD on Success", value=False)
                start_btn = gr.Button("🔥 START PIPELINE", variant="primary")
                stop_btn = gr.Button("🛑 STOP", variant="stop")
            
            last_path_state = gr.State("")
            sync_trigger = gr.Button("📤 Manual Move to SSD")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("💻 Terminal Feed"):
                    terminal_box = gr.Textbox(lines=25, interactive=False, elem_id="terminal")
                with gr.Tab("📝 Recipe Editor"):
                    recipe_editor = gr.Code(language="json", lines=25)

    # --- EVENT BINDINGS ---
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    
    start_btn.click(
        fn=run_pipeline, 
        inputs=[recipe_editor, base_dd, quant_select, recipe_dd, auto_move_toggle], 
        outputs=[terminal_box, last_path_state]
    )

    stop_btn.click(fn=terminate_pipeline, outputs=[terminal_box])
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL)
    sync_trigger.click(fn=sync_ram_to_ssd, inputs=[last_path_state], outputs=[terminal_box])

# --- 7. LAUNCH (CSS Moved Here for Gradio 6.0) ---
if __name__ == "__main__":
    demo.launch(css=CSS_STYLE)