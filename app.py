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

# Global variable to track the active quantization process
active_process = None

# Ensure directories exist
for d in [MODELS_DIR, RECIPES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 2. STYLING & SCRIPTS ---
CSS_STYLE = """
#terminal textarea { 
    background-color: #0d1117 !important; 
    color: #00ff41 !important; 
    font-family: 'Fira Code', 'Cascadia Code', monospace !important; 
    font-size: 13px !important;
    line-height: 1.5 !important;
    border: 1px solid #30363d !important;
    overflow-y: auto !important;
    white-space: pre !important;
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
    const el = document.querySelector('#terminal textarea');
    if (el) { 
        // Only scroll if we aren't manually looking at history
        // This check allows you to scroll up without it snapping back constantly
        const threshold = 100; 
        const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
        if (isAtBottom) {
            el.scrollTop = el.scrollHeight; 
        }
    }
}
"""

# --- 3. UTILITIES ---
def get_sys_info():
    # System Stats
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    
    # GPU Stats
    gpu_load = "0%"
    vram_info = "0.0/0.0GB"
    
    if torch.cuda.is_available():
        try:
            # Get VRAM: Used / Total
            v_used = torch.cuda.memory_reserved() / 1e9
            v_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_info = f"{v_used:.1f}/{v_total:.1f}GB"
            
            # Get GPU Utilization % (KDE Style) via nvidia-smi
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                encoding='utf-8'
            )
            gpu_load = f"{res.strip()}%"
        except:
            gpu_load = "ERR%"

    # RAMDisk Stats
    rd_status = "💾 RD: [OFFLINE]"
    if os.path.exists(RAMDISK_PATH):
        try:
            usage = psutil.disk_usage(RAMDISK_PATH)
            rd_status = f"💾 RD: {usage.used/1e9:>5.1f} / {usage.total/1e9:.1f}GB"
        except:
            rd_status = "💾 RD: [LOCKED]"

    return (f"🖥️ CPU: {cpu:>3}% | RAM: {ram:>3}%\n"
            f"📟 GPU: {gpu_load:>4} | VRAM: {vram_info}\n"
            f"{rd_status}")

def list_files():
    try:
        m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.bin', '.gguf'))])
        r = sorted([f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')])
        return gr.update(choices=m), gr.update(choices=r)
    except: return gr.update(choices=[]), gr.update(choices=[])

def load_recipe_text(name):
    if not name: return ""
    with open(os.path.join(RECIPES_DIR, name), 'r') as f: return f.read()

def sync_ram_to_ssd(path):
    if not path or not os.path.exists(path): return "❌ SYNC ERROR: Path invalid."
    dest = os.path.join(MODELS_DIR, os.path.basename(path))
    try:
        shutil.move(path, dest)
        return f"✅ SYNC SUCCESS: {os.path.basename(dest)} moved to SSD."
    except Exception as e: return f"❌ SYNC FAILED: {str(e)}"

def terminate_pipeline():
    global active_process
    if active_process and active_process.poll() is None:
        # Kill the subprocess group (the quantizer)
        active_process.terminate()
        active_process = None
        return "🛑 SYSTEM MANUALLY TERMINATED. Subprocess killed."
    return "ℹ️ No active process to stop."

# --- 4. THE MASTER PIPELINE ---
def run_pipeline(recipe_json, base_model, q_format, recipe_name, auto_move, progress=gr.Progress()):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "-"*60 + "\n"
    global active_process
    
    try:
        # STAGE 1: PARSING
        clean_json = re.sub(r'//.*', '', recipe_json)
        recipe_dict = json.loads(clean_json)
        recipe_dict['paths'] = recipe_dict.get('paths', {})
        recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
        
        engine = ActionMasterEngine(recipe_dict)
        log_acc += f"🧬 ENGINE: {engine.role_label} Architecture Detected.\n"

        # --- DEEP CACHE LOOKUP ---
        is_gguf = q_format.startswith("GGUF_")
        recipe_slug = recipe_name.replace(".json", "")
        mode_suffix = "flattened" if is_gguf else "native"
        
        found_cache = None
        for f in os.listdir(MODELS_DIR):
            if recipe_slug in f and mode_suffix in f and f.endswith(".safetensors"):
                found_cache = os.path.join(MODELS_DIR, f)
                break
        
        if found_cache:
            temp_path = found_cache
            log_acc += f"♻️ DEEP CACHE HIT: Found intermediate at {temp_path}\n"
            log_acc += "⏭️ SKIPPING MERGE: Using existing data.\n"
            yield log_acc, temp_path
        else:
            model_slug = os.path.splitext(base_model)[0][:10]
            cache_name = f"cache_{model_slug}_{recipe_slug}_{mode_suffix}.safetensors"
            temp_path = os.path.join(MODELS_DIR, cache_name)
            
            # STAGE 2: MERGE LOOP
            pipeline = recipe_dict.get('pipeline', [])
            for i, step in enumerate(pipeline):
                p_name = step.get('pass_name', f"Pass {i+1}")
                progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}...")
                log_acc += f"▶️ EXECUTING: {p_name}...\n"
                engine.process_pass(step, 1.0)
                yield log_acc, ""

            # STAGE 3: ROBUST TENSOR PRESERVATION
            progress(0.8, desc="📐 Preserving 5D Structure...")
            
            # We avoid 'save_pure_5d' because it can strip distilled metadata.
            # 'save_and_patch' keeps the original model's tensor indexing.
            log_acc += f"📐 Mode: {'GGUF' if is_gguf else 'FP8'} | Preservation: Robust\n"
            
            # 64GB SAFETY: Save to SSD to keep RAM clear for the merging math
            raw_out = engine.save_and_patch(use_ramdisk=False)
            
            # Verify the file was created before moving on
            if os.path.exists(raw_out):
                os.rename(raw_out, temp_path)
                log_acc += f"✅ INTERMEDIATE VERIFIED: {temp_path}\n"
            else:
                raise Exception("Engine failed to export the patched model.")
                
            yield log_acc, temp_path
        
        # Clean up VRAM/RAM before the heavy lifting    
        gc.collect()
        torch.cuda.empty_cache()
        log_acc += "🧹 MEMORY: VRAM flushed for Quantization.\n"
        yield log_acc, temp_path

        # STAGE 4: QUANTIZATION (HYBRID STORAGE + LOG FILTER)
        progress(0.9, desc=f"🏗️ Conversion: {q_format}...")
        
        # --- PRE-QUANT MEMORY FLUSH ---
        gc.collect()
        torch.cuda.empty_cache()
        log_acc += "🧹 MEMORY: VRAM flushed for Quantization.\n"
        yield log_acc, temp_path

        out_prefix = recipe_dict['paths'].get('output_prefix', 'Wan22_Output')
        
        # Determine the directory: prioritize RAMDisk if it exists
        output_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
        
        if is_gguf:
            q_type = q_format.replace("GGUF_", "")
            final_name = f"{out_prefix}_{recipe_slug}_{q_type}.gguf"
        else:
            final_name = f"{out_prefix}_{recipe_slug}_{q_format}.safetensors"
            
        # FORCE FULL PATH
        final_output_path = os.path.abspath(os.path.join(output_dir, final_name))
        
        if is_gguf:
            cmd = ["python", "convert_to_gguf.py", temp_path, "--out", final_output_path, "--quant", q_type]
        else:
            fmt_flag = [] if q_format == "fp8" else [f"--{q_format}"]
            cmd = ["convert_to_quant", "-i", temp_path, "-o", final_output_path, "--comfy_quant", "--wan"] + fmt_flag

        log_acc += f"🖥️ CLI EXEC: {' '.join(cmd)}\n"
        log_acc += f"🛰️ OUTPUT TARGET: {final_output_path}\n"
        yield log_acc, temp_path

        # --- MONITORING LOOP ---
        global active_process
        active_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        last_log_time = time.time()
        for line in active_process.stdout:
            is_spam = "worse_count" in line or "Optimizing" in line
            is_important = any(k in line for k in ["Error", "Success", "Saved", "Loading", "Finalizing"])

            if is_important or not is_spam:
                log_acc += f"  [QUANT] {line}"
                yield log_acc, temp_path
            else:
                if time.time() - last_log_time > 8:
                    log_acc += "  [QUANT] ...still optimizing weights...\n"
                    last_log_time = time.time()
                    yield log_acc, temp_path
        
        active_process.wait()

        # FINAL VERIFICATION
        if os.path.exists(final_output_path):
            log_acc += f"✅ SUCCESS: File confirmed at {final_output_path}\n"
            # IMPORTANT: We update last_path_state so the "Sync" button knows what to move
            yield log_acc, final_output_path
        else:
            log_acc += f"❌ ERROR: Quantizer finished but {final_name} was not found in target dir.\n"
            yield log_acc, ""

        active_process = None

        if auto_move and os.path.exists(final_output_path):
            # If it's already on the SSD (because RAMDisk didn't exist), sync_ram_to_ssd handles it
            log_acc += f"🚀 AUTO-MOVE: {sync_ram_to_ssd(final_output_path)}\n"
            yield log_acc, final_output_path

# --- 5. UI ARCHITECTURE ---
with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**Direct-to-SSD 5D-Patching & GGUF/SVD Quantization**")
        with gr.Column(scale=2, elem_classes=["vitals-card"]):
            vitals_box = gr.Textbox(label="Environment Stats", value=get_sys_info(), lines=4, interactive=False, container=False)
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
                quant_select = gr.Dropdown(
                    choices=[
                        "fp8", "nvfp4", "int8",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", 
                        "GGUF_Q4_K_M", "GGUF_Q3_K_L", "GGUF_Q2_K"
                    ], 
                    value="fp8", 
                    label="Target Format"
                )
                auto_move_toggle = gr.Checkbox(label="🚀 Auto-move to SSD on finish", value=False)
                start_btn = gr.Button("🔥 START PIPELINE", variant="primary")
                stop_btn = gr.Button("🛑 EMERGENCY STOP", variant="stop")
            
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

    # --- 6. EVENT BINDINGS ---
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    
    start_btn.click(
        fn=run_pipeline, 
        inputs=[recipe_editor, base_dd, quant_select, recipe_dd, auto_move_toggle], 
        outputs=[terminal_box, last_path_state],
        show_progress="minimal"
    )

    stop_btn.click(fn=terminate_pipeline, outputs=[terminal_box])

    terminal_box.change(fn=None, js=JS_AUTO_SCROLL)

    sync_trigger.click(fn=sync_ram_to_ssd, inputs=[last_path_state], outputs=[sync_output])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS_STYLE)