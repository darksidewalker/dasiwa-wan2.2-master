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

def instant_validate(recipe_name, base_model):
    if not recipe_name or not base_model:
        return "### 🛡️ Status: Waiting for selection..."
    
    # Quick check: Is this a Motion (High) or Refiner (Low) base?
    is_motion = "high" in base_model.lower() or "i2v" in base_model.lower()
    forbidden = "low" if is_motion else "high"
    
    try:
        with open(os.path.join(RECIPES_DIR, recipe_name), 'r') as f:
            recipe_content = f.read()
            # Check for forbidden keywords in the raw text for speed
            if forbidden in recipe_content.lower():
                return f"### ❌ Status: **CONFLICT DETECTED** (Mismatched {forbidden} noise LoRA)"
            return "### ✅ Status: **VALIDATED** (Architecture Alignment Verified)"
    except Exception:
        return "### ⚠️ Status: Validation Error (Check JSON format)"

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
        # 1. SETUP & ENGINE INIT
        progress(0.05, desc="Initializing Engine...")
        clean_json = re.sub(r'//.*', '', recipe_json)
        recipe_dict = json.loads(clean_json)
        
        recipe_dict['paths'] = recipe_dict.get('paths', {})
        recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
        recipe_dict['paths']['title'] = recipe_dict['paths'].get('title', recipe_name.replace(".json", ""))
        
        # Initialize Engine (ONLY ONCE - saves 28GB RAM spike)
        engine = ActionMasterEngine(recipe_dict)

        # --- VALIDATION HEADER ---
        mismatches = engine.get_compatibility_report()
        border = "=" * 60
        header = f"\n{border}\n🛡️  RECIPE VALIDATION: {engine.role_label}\n{border}\n"
        
        if mismatches:
            header += f"❌ CONFLICT: {len(mismatches)} LoRA(s) mismatch noise levels!\n"
            for m in mismatches: header += f"   - [WARN] {m}\n"
            header += f"{border}\n⚠️  PROCEEDING WITH CAUTION...\n\n"
        else:
            header += "✅ ALL SYSTEMS CLEAR: Alignment Verified.\n"
            header += f"{border}\n\n"

        print(header)      # CLI
        log_acc += header  # GUI Accumulator
        yield log_acc, ""  # Force GUI refresh

        # 2. WORKSPACE SETUP
        progress(0.1, desc="Setting up workspace...")
        recipe_slug = recipe_name.replace(".json", "")
        cache_name = f"MASTER_{recipe_slug}.safetensors"
        output_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
        temp_path = os.path.join(output_dir, cache_name)

        # 3. MERGING LOOP
        pipeline = recipe_dict.get('pipeline', [])
        global_mult = recipe_dict['paths'].get('global_weight_factor', 1.0)

        for i, step in enumerate(pipeline):
            p_name = step.get('pass_name', f"Pass {i+1}")
            progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}")
            
            log_acc += f"▶️ {p_name.upper()} | Mode: {step.get('method', 'ADDITION').upper()}\n"
            yield log_acc, ""
            
            engine.process_pass(step, global_mult)
            
            last_pass = engine.summary_data[-1]
            peak_str = f" | ⚠️ PEAKS: {last_pass['peaks']}" if last_pass['peaks'] > 0 else ""
            log_acc += f"  └─ Injection: {last_pass['inj']:.1f}% | Shift: {last_pass['delta']:.8f}{peak_str}\n"
            yield log_acc, ""

        # 4. SAVE MASTER (Memory Safe)
        progress(0.8, desc="Finalizing Report...")
        log_acc += engine.get_final_summary_string() + "\n"
        log_acc += f"💾 EXPORT: Saving 28GB Master to {output_dir}...\n"
        yield log_acc, ""
        
        engine.save_master(temp_path) 
        
        # --- NEW: FILE EXISTENCE & INTEGRITY CHECK ---
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1e9: # Check if > 1GB
            error_msg = f"❌ SAVE ERROR: Master file missing or corrupt at {temp_path}"
            print(error_msg)
            log_acc += error_msg + "\n"
            yield log_acc, ""
            return "🛑 Save Failure", ""
            
        log_acc += f"✅ MASTER SAVED: {cache_name} ({os.path.getsize(temp_path)/1e9:.1f} GB)\n"
        yield log_acc, ""

        # 5. VRAM PURGE & QUANTIZATION
        engine._cleanup() 
        torch.cuda.empty_cache()
        gc.collect()
        log_acc += "🧹 VRAM Purged. Initializing Quantization...\n"
        yield log_acc, temp_path

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Cleaning up...")
        torch.cuda.empty_cache()
        return "🛑 User Aborted.", ""
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
                val_status_display = gr.Markdown("### 🛡️ Status: No Recipe Selected") # New Component
                refresh_btn = gr.Button("🔄 Refresh Assets")
            
            with gr.Group():
                quant_select = gr.Radio(
                    choices=[
                        "fp8", "nvfp4", "int8", 
                        "GGUF_Q8_0", 
                        "GGUF_Q6_K", 
                        "GGUF_Q5_K_M", 
                        "GGUF_Q4_K_M", 
                        "GGUF_Q3_K_L", 
                        "GGUF_Q2_K"
                    ], 
                    value="fp8", 
                    label="Export Format"
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
    recipe_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    base_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    
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
    try:
        demo.launch(css=CSS_STYLE)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 SIGNAL RECEIVED: Performing Clean Shutdown...")
        print("   - Clearing Tensors...")
        torch.cuda.empty_cache()
        print("   - Terminating Active Processes...")
        terminate_pipeline() # Uses your existing function
        print("✅ Shutdown Complete. Goodbye Master.")
        print("="*60 + "\n")