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
    
    # STRICT DETECTION: Look for noise tags only
    model_is_high = "high_noise" in base_model.lower()
    recipe_is_high = "high" in recipe_name.lower()
    
    model_label = "MOTION (High)" if model_is_high else "REFINER (Low)"
    
    try:
        # Load JSON to check internal LoRA paths
        recipe_path = os.path.join(RECIPES_DIR, recipe_name)
        with open(recipe_path, 'r') as f:
            content = f.read().lower()
            
            # Scenario A: High Model check
            if model_is_high:
                if "low_noise" in content or "_low" in content:
                    return f"### ❌ MISMATCH: {model_label} Model vs. LOW LoRAs found in JSON"
                if not recipe_is_high:
                    return f"### ❌ MISMATCH: {model_label} Model vs. LOW Recipe filename"
            
            # Scenario B: Low Model check
            else:
                if "high_noise" in content or "_high" in content:
                    return f"### ❌ MISMATCH: {model_label} Model vs. HIGH LoRAs found in JSON"
                if recipe_is_high:
                    return f"### ❌ MISMATCH: {model_label} Model vs. HIGH Recipe filename"

        return f"### ✅ VALIDATED: {model_label} Architecture Alignment"
    except Exception as e:
        return f"### ⚠️ Status: Validation Error ({str(e)})"

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
    
    # Path for intermediate master (Always SSD to prevent OOM)
    recipe_slug = recipe_name.replace(".json", "")
    cache_name = f"MASTER_{recipe_slug}.safetensors"
    temp_path = os.path.join(MODELS_DIR, cache_name)
    
    # Path for final GGUF (Always RAM Disk if available)
    final_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
    
    # --- SMART SKIP LOGIC ---
    # Check if a valid master already exists on SSD
    master_exists = os.path.exists(temp_path) and os.path.getsize(temp_path) > 1e9
    # Skip merging if file exists AND we aren't explicitly asking for a new FP16 Master
    skip_merge = master_exists and q_format != "None (FP16 Master)"
    
    try:
        if skip_merge:
            log_acc += f"⚡ FAST TRACK: Found existing Master: {cache_name}\n"
            log_acc += "⏭️ Skipping Merge Loop and jumping to Quantization...\n\n"
            yield log_acc, "", "Fast Tracking..."
        else:
            # 1. SETUP & ENGINE INIT
            progress(0.05, desc="Initializing Engine...")
            yield log_acc, "", "Initializing Engine..."
            
            clean_json = re.sub(r'//.*', '', recipe_json)
            recipe_dict = json.loads(clean_json)
            
            recipe_dict['paths'] = recipe_dict.get('paths', {})
            recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
            recipe_dict['paths']['title'] = recipe_dict['paths'].get('title', recipe_slug)
            
            # Initialize Engine
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

            log_acc += header
            yield log_acc, "", "Merging Layers..."

            # 2. MERGING LOOP
            pipeline = recipe_dict.get('pipeline', [])
            global_mult = recipe_dict['paths'].get('global_weight_factor', 1.0)

            for i, step in enumerate(pipeline):
                p_name = step.get('pass_name', f"Pass {i+1}")
                progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}")
                
                log_acc += f"▶️ {p_name.upper()} | Mode: {step.get('method', 'ADDITION').upper()}\n"
                yield log_acc, "", f"Merging: {p_name}"
                
                engine.process_pass(step, global_mult)
                
                last_pass = engine.summary_data[-1]
                peak_str = f" | ⚠️ PEAKS: {last_pass['peaks']}" if last_pass['peaks'] > 0 else ""
                log_acc += f"  └─ Injection: {last_pass['inj']:.1f}% | Shift: {last_pass['delta']:.8f}{peak_str}\n"
                yield log_acc, "", f"Merging: {p_name}"

            # 3. SAVE MASTER (SSD)
            progress(0.8, desc="Exporting to SSD...")
            log_acc += engine.get_final_summary_string() + "\n"
            log_acc += f"💾 EXPORT: Writing 28GB Master to SSD: {temp_path}...\n"
            log_acc += "⚠️ UI may pause briefly during I/O write...\n"
            yield log_acc, "", "Exporting to SSD..."
            
            engine.save_master(temp_path) 
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1e9:
                error_msg = f"❌ SAVE ERROR: Master file missing or corrupt at {temp_path}"
                log_acc += error_msg + "\n"
                yield log_acc, "", "Save Failed"
                return "🛑 Save Failure", "", "Error"
                
            log_acc += f"✅ MASTER SAVED: {os.path.getsize(temp_path)/1e9:.1f} GB\n"
            yield log_acc, "", "Export Complete"

        # 4. QUANTIZATION / EXPORT (RAM Disk)
        torch.cuda.empty_cache()
        gc.collect()

        if q_format != "None (FP16 Master)":
            if "GGUF_" in q_format:
                q_type = q_format.replace("GGUF_", "")
                final_name = f"WAN22_{recipe_slug}_{q_type}.gguf"
                final_path = os.path.join(final_dir, final_name)
                
                yield log_acc + f"🔨 GGUF: {q_type} -> {final_path}\n", "", f"GGUF {q_type}..."
                cmd = ["python", "convert.py", "--path", temp_path, "--dst", final_path, "--outtype", q_type]
            
            else:
                final_name = f"WAN22_{recipe_slug}_{q_format}.safetensors"
                final_path = os.path.join(final_dir, final_name)
                
                yield log_acc + f"🔨 PIP PACKAGE: {q_format.upper()} -> {final_path}\n", "", f"Quantizing {q_format}..."
                
                cmd = ["convert_to_quant", "-i", temp_path, "--comfy_quant", "--wan"]
                
                if q_format == "int8":
                    cmd += ["--int8", "--block_size", "128"]
                elif q_format == "nvfp4":
                    cmd += ["--nvfp4"]

            subprocess.run(cmd, check=True)
            log_acc += f"✅ EXPORT COMPLETE: {final_name}\n"
            yield log_acc, final_path, "Process Finished"
        else:
            yield log_acc + "✅ MASTER FP16 READY ON SSD.\n", temp_path, "Finished"

    except KeyboardInterrupt:
        yield log_acc + "\n⚠️ Interrupted by user.\n", "", "Aborted"
    except Exception as e:
        log_acc += f"\n🔥 CRITICAL FAILURE: {str(e)}\n"
        yield log_acc, "", "Critical Error"
    finally:
        global active_process
        active_process = None

# --- 5. UI CONSTRUCTION (Gradio 6.0 Compliant) ---
with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**14B High-Precision MoE Pipeline**")
        
        # COLUMN 2: RESTORED VITALS
        with gr.Column(scale=3):
            with gr.Group(elem_classes="vitals-card"):
                vitals_box = gr.Textbox(label="System Health", value=get_sys_info(), lines=3, interactive=False)
                # Keep the timer active so it updates every 2 seconds
                gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        
        # COLUMN 3: STAGE & PROGRESS
        with gr.Column(scale=3):
            with gr.Group(elem_classes="vitals-card"):
                pipeline_status = gr.Label(label="Current Stage", value="Idle")
                # This anchors the progress bar here so it stays out of the terminal
                main_progress = gr.Progress(track_tqdm=True)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📂 Assets")
                base_dd = gr.Dropdown(label="Base Model")
                recipe_dd = gr.Dropdown(label="Active Recipe")
                val_status_display = gr.Markdown("### 🛡️ Status: No Recipe Selected")
                refresh_btn = gr.Button("🔄 Refresh Assets", size="sm")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Export Configuration")
                quant_select = gr.Radio(
                    # FIX: Simplified string to match pipeline logic exactly
                    choices=[
                        "None (FP16 Master)", "fp8", "nvfp4", "int8", 
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q4_K_M"
                    ], 
                    value="None (FP16 Master)", 
                    label="Target Format"
                )
                auto_move_toggle = gr.Checkbox(label="🚀 Move to SSD on Success", value=False)
                
            with gr.Row():
                start_btn = gr.Button("🔥 START", variant="primary", scale=2)
                stop_btn = gr.Button("🛑 STOP", variant="stop", scale=1)
            
            sync_trigger = gr.Button("📤 Manual Move to SSD", variant="secondary")
            last_path_state = gr.State("")

        # RIGHT: Feed & Code
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("💻 Terminal Feed"):
                    terminal_box = gr.Textbox(lines=28, interactive=False, elem_id="terminal", show_label=False)
                with gr.Tab("📝 Recipe Editor"):
                    recipe_editor = gr.Code(language="json", lines=28)

    # --- EVENT BINDINGS ---
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    
    # Validator logic (Updated to check recipe contents)
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    recipe_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    base_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    
    # Start Pipeline (3 Outputs to match run_pipeline yield)
    start_btn.click(
        fn=run_pipeline, 
        inputs=[recipe_editor, base_dd, quant_select, recipe_dd, auto_move_toggle], 
        outputs=[terminal_box, last_path_state, pipeline_status],
        show_progress="hidden" # This prevents the overlay on Textboxes
    )

    stop_btn.click(fn=terminate_pipeline, outputs=[terminal_box])
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL)
    sync_trigger.click(fn=sync_ram_to_ssd, inputs=[last_path_state], outputs=[terminal_box])

# --- 7. LAUNCH (CSS Moved Here for Gradio 6.0) ---
if __name__ == "__main__":
    try:
        # The 'show_api=False' helps prevent some internal thread conflicts
        demo.launch(css=CSS_STYLE) 
    except KeyboardInterrupt:
        print("\n" + "!"*60)
        print("🛑 SIGNAL RECEIVED: Performing Clean Shutdown...")
        
        # 1. Kill the background merge/quant process if it exists
        if active_process:
            print("   - Terminating active subprocess...")
            active_process.terminate()
            
        # 2. Clear GPU memory to prevent driver hanging
        print("   - Flushing VRAM...")
        torch.cuda.empty_cache()
        
        print("✅ Shutdown Complete. Terminal Safe.")
        print("!"*60 + "\n")
        # Exit without letting the error bubble up to the OS
        os._exit(0)