import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info, instant_validate, get_final_summary_string, sync_ram_to_ssd
from engine import ActionMasterEngine

# Global handle for the background process
active_process = None
ensure_dirs()

# --- HELPER FUNCTIONS ---

def list_files():
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.gguf'))])
    r = sorted([f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')])
    return gr.update(choices=m), gr.update(choices=r)

def load_recipe_text(name):
    if not name: return ""
    try:
        with open(os.path.join(RECIPES_DIR, name), 'r') as f: return f.read()
    except: return "❌ Error loading recipe."

def stop_pipeline():
    global active_process
    if active_process:
        print("🛑 STOP SIGNAL: Terminating subprocess...")
        active_process.terminate()
        active_process = None
    torch.cuda.empty_cache()
    gc.collect()
    return "🛑 PROCESS TERMINATED BY USER\n" + "-"*60, "Idle"

# --- MAIN PIPELINE ---
def run_pipeline(recipe_json, base_model, q_formats, recipe_name, progress=gr.Progress()):
    if not q_formats:
        yield "❌ ERROR: No export formats selected. Please check at least one.", "", "Idle"
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "="*60 + "\n"
    global active_process
    
    recipe_slug = recipe_name.replace(".json", "") if recipe_name else "custom_merge"
    cache_name = f"MASTER_{recipe_slug}.safetensors"
    temp_path = os.path.join(MODELS_DIR, cache_name)

    ROOT_DIR = os.getcwd()
    LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    CONVERT_PY = os.path.join(ROOT_DIR, "convert.py")
    FIX_5D_PY = os.path.join(ROOT_DIR, "fix_5d_tensors.py")
    
    master_exists = os.path.exists(temp_path) and os.path.getsize(temp_path) > 1e9

    skip_merge = master_exists and not (len(q_formats) == 1 and "None (FP16 Master)" in q_formats)
    
    try:
        if skip_merge:
            log_acc += f"⚡ FAST TRACK: Found existing Master: {cache_name}\n"
            log_acc += "⏭️ Skipping Merge Loop and jumping to Batch Export...\n\n"
            yield log_acc, "", "Fast Tracking..."
        else:
            # --- 1. ENGINE INIT ---
            progress(0.05, desc="Initializing Engine...")
            clean_json = re.sub(r'//.*', '', recipe_json)
            recipe_dict = json.loads(clean_json)
            recipe_dict['paths'] = recipe_dict.get('paths', {})
            recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
            recipe_dict['paths']['title'] = recipe_dict['paths'].get('title', recipe_slug)
            
            engine = ActionMasterEngine(recipe_dict)

            mismatches = engine.get_compatibility_report()
            log_acc += f"\n{'='*60}\n🛡️  RECIPE VALIDATION: {engine.role_label}\n{'='*60}\n"
            if mismatches:
                log_acc += f"❌ CONFLICT: {len(mismatches)} LoRA(s) mismatch noise levels!\n"
                for m in mismatches: log_acc += f"   - [WARN] {m}\n"
            else:
                log_acc += "✅ ALL SYSTEMS CLEAR: Alignment Verified.\n"
            log_acc += f"{'='*60}\n\n"
            yield log_acc, "", "Merging Layers..."

            # --- 2. MERGING LOOP ---
            pipeline = recipe_dict.get('pipeline', [])
            global_mult = recipe_dict['paths'].get('global_weight_factor', 1.0)

            for i, step in enumerate(pipeline):
                p_name = step.get('pass_name', f"Pass {i+1}")
                progress(0.1 + (i/len(pipeline) * 0.5), desc=f"Merging {p_name}")
                for message in engine.process_pass(step, global_mult):
                    log_acc += message + "\n"
                    yield log_acc, "", f"Working: {p_name}"

            # --- 3. SAFETY GATE (INTEGRITY CHECK) ---
            progress(0.65, desc="Verifying 14B Integrity...")
            log_acc += "\n" + "="*60 + "\n"
            from utils import verify_model_integrity
            try:
                # Scan tensors in GPU/RAM before committing to SSD
                for diag_msg in verify_model_integrity(engine.base_dict, engine.base_keys, engine.router_regex):
                    log_acc += diag_msg + "\n"
                    yield log_acc, "", "🛡️ Verifying Tensors..."
            except Exception as integrity_error:
                log_acc += f"\n🔥 INTEGRITY FAILURE: {str(integrity_error)}\n"
                yield log_acc, "", "Verification Failed"
                return 

            # --- 4. SAVE MASTER ---
            progress(0.75, desc="Writing Master File...")
            summary_table = get_final_summary_string(engine.summary_data, engine.role_label)
            log_acc += "\n" + summary_table + "\n"
            log_acc += f"💾 EXPORT: Writing Source Master to SSD: {temp_path}...\n"
            yield log_acc, "", "💾 WRITING MASTER..." 
            
            engine.save_master(temp_path) 
            
            # Wipe engine for 64GB RAM clearance
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            log_acc += f"✅ SOURCE MASTER READY.\n\n"

        # --- 5. BATCH EXPORT QUEUE ---
        log_acc += "📦 STARTING BATCH EXPORT QUEUE\n" + "-"*60 + "\n"
        
        # Track if we created a BF16 intermediate for cleanup
        bf16_created_path = None

        for fmt in q_formats:
            try:
                if fmt == "None (FP16 Master)":
                    log_acc += "🔹 FP16 Master: Ready (Source)\n"
                    yield log_acc, temp_path, "FP16 Ready"
                    continue

                # --- CASE A: GGUF WITH 5D-FIX CHAIN ---
                if "GGUF_" in fmt:
                    q_type = fmt.replace("GGUF_", "")
                    bf16_gguf = temp_path.replace(".safetensors", "-BF16.gguf")
                    quant_gguf = temp_path.replace(".safetensors", f"-{q_type}.gguf")
                    
                    # 1. Create BF16 source if missing
                    if not os.path.exists(bf16_gguf):
                        yield log_acc + f"📦 GGUF {q_type}: Creating BF16 GGUF...\n", "", "Step 1/3"
                        # We use execute_export_logic to keep the STOP button active during long runs
                        cmd_bf16 = ["python", CONVERT_PY, "--src", temp_path]
                        for update in execute_export_logic(cmd_bf16, "BF16_Base", bf16_gguf, "BF16", False, MODELS_DIR, log_acc):
                            if isinstance(update, tuple): yield update
                        bf16_created_path = bf16_gguf

                    # 2. llama-quantize from build folder
                    yield log_acc + f"🔨 GGUF {q_type}: Quantizing...\n", "", "Step 2/3"
                    cmd_quant = [LLAMA_BIN, bf16_gguf, quant_gguf, q_type]
                    # Direct subprocess for binary logic
                    subprocess.run(cmd_quant, check=True)

                    # 3. Apply 5D Fix (Overwrites intermediate)
                    yield log_acc + f"🔧 GGUF {q_type}: Applying 5D Fix...\n", "", "Step 3/3"
                    cmd_fix = ["python", FIX_5D_PY, "--src", quant_gguf, "--dst", quant_gguf]
                    subprocess.run(cmd_fix, check=True)
                    
                    log_acc += f"✅ GGUF {q_type} Successfully Fixed and Exported.\n"
                    yield log_acc, quant_gguf, f"Finished {q_type}"

                # --- CASE B: SAFETENSORS (convert_to_quant PIP) ---
                else:
                    suffix = fmt.lower().replace(" ", "_")
                    final_path = temp_path.replace(".safetensors", f"_{suffix}.safetensors")
                    
                    # Command Logic (Default is FP8)
                    cmd = ["convert_to_quant", "-i", temp_path, "-o", final_path, "--comfy_quant", "--wan"]
                    if "int8" in fmt.lower(): cmd += ["--int8", "--block_size", "128"]
                    elif "nvfp4" in fmt.lower(): cmd += ["--nvfp4"]

                    yield log_acc + f"🚀 Exporting {fmt} via PIP Tool...\n", "", f"Working: {fmt}"
                    for update in execute_export_logic(cmd, suffix, final_path, fmt, False, MODELS_DIR, log_acc):
                        if isinstance(update, tuple):
                            log_acc = update[0]
                            yield update
                    
                    log_acc += f"✅ Finished: {fmt}\n"

            except Exception as e:
                log_acc += f"❌ ERROR in {fmt}: {str(e)}\n"
                yield log_acc, "", f"Skipped {fmt}"
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        if bf16_created_path and os.path.exists(bf16_created_path):
            os.remove(bf16_created_path)
            log_acc += "🧹 Cleaned up intermediate BF16 GGUF.\n"

        log_acc += f"\n{'='*60}\n✨ ALL BATCH TASKS COMPLETE.\n{'='*60}\n"
        yield log_acc, temp_path, "Process Finished"

    except Exception as e:
        log_acc += f"\n🔥 CRITICAL FAILURE: {str(e)}\n"
        yield log_acc, "", "Critical Error"
    finally:
        try:
            log_filename = f"merge_{recipe_slug}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
            with open(os.path.join(LOGS_DIR, log_filename), "w", encoding="utf-8") as f:
                f.write(log_acc)
        except: pass
        active_process = None
        torch.cuda.empty_cache()

# --- REUSABLE EXPORT HELPER ---
def execute_export_logic(cmd, final_name, final_path, q_format, auto_move, final_dir, log_acc):
    global active_process
    yield log_acc + f"🔨 STARTING EXPORT: {final_name}...\n", "", f"Quantizing {q_format}..."
    
    # Use Popen to keep it interruptible by the STOP button
    active_process = subprocess.Popen(cmd)
    active_process.wait()
    
    # Check if it was stopped by the user or crashed
    if active_process is None: # Means stop_pipeline was called
         yield log_acc + "🛑 EXPORT CANCELLED BY USER.\n", "", "Stopped"
         return

    if active_process.returncode != 0:
        raise Exception(f"Quantization failed (Code {active_process.returncode})")
    
    active_process = None
    log_acc += f"✅ EXPORT COMPLETE: {final_name}\n"
    yield log_acc, final_path, "Process Finished"

# --- UI LAYOUT ---

with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**14B High-Precision MoE Pipeline**")
        with gr.Column(scale=3):
            vitals_box = gr.Textbox(label="System Health", value=get_sys_info(), lines=3, interactive=False)
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        with gr.Column(scale=3):
            pipeline_status = gr.Label(label="Current Stage", value="Idle")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                base_dd = gr.Dropdown(label="Base Model")
                recipe_dd = gr.Dropdown(label="Active Recipe")
                val_status_display = gr.Markdown("### 🛡️ Status: No Recipe Selected")
                refresh_btn = gr.Button("🔄 Refresh Assets", size="sm")
            with gr.Group():
                q_format = gr.CheckboxGroup(
                    choices=[
                        "None (FP16 Master)", 
                        "FP8 (SVD)", 
                        "INT8 (Block-wise)", 
                        "NVFP4 (Blackwell)",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", 
                        "GGUF_Q4_K_M", "GGUF_Q3_K_M", "GGUF_Q2_K"
                    ],
                    label="Batch Export: FP8, INT8, NVFP4 & GGUF Q8-Q2",
                    value=["None (FP16 Master)"],
                    elem_classes=["quant-selector"]
                )
            with gr.Row():
                run_btn = gr.Button(
                    "🧩 RUN", 
                    variant="primary", 
                    elem_classes=["primary-button"] # This triggers the green gradient in config.py
                )
                stop_btn = gr.Button("🛑 STOP", variant="stop", scale=1)
            
            last_path_state = gr.State("")

        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("💻 Terminal"): terminal_box = gr.Textbox(lines=28, interactive=False, show_label=False, elem_id="terminal")
                with gr.Tab("📝 Editor"): recipe_editor = gr.Code(language="json", lines=28)

    # --- BINDINGS ---
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    recipe_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    base_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    
    run_event = run_btn.click(
        fn=run_pipeline,
        inputs=[
            recipe_editor,
            base_dd,
            q_format,
            recipe_dd
        ],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status], cancels=[run_event])
    
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL)

if __name__ == "__main__":
    try:
        demo.launch(css=CSS_STYLE) 
    except KeyboardInterrupt:
        print("\n" + "!"*60)
        print("🛑 SIGNAL RECEIVED: Performing Clean Shutdown...")
        if active_process:
            print("   - Terminating active subprocess...")
            active_process.terminate()
        print("   - Flushing VRAM...")
        torch.cuda.empty_cache()
        print("✅ Shutdown Complete. Terminal Safe.")
        print("!"*60 + "\n")
        os._exit(0)