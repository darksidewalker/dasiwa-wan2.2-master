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
    """
    Refactored Station Master Pipeline.
    Supports Batch Export (Checkboxes) and Surgical MoE Shielding.
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "="*60 + "\n"
    global active_process
    
    recipe_slug = recipe_name.replace(".json", "") if recipe_name else "custom_merge"
    cache_name = f"MASTER_{recipe_slug}.safetensors"
    temp_path = os.path.join(MODELS_DIR, cache_name)
    final_dir = MODELS_DIR
    
    # Check if a Master source already exists to avoid re-merging
    master_exists = os.path.exists(temp_path) and os.path.getsize(temp_path) > 1e9
    
    # Logic change: Skip merge ONLY if the Master exists AND FP16 isn't the only thing requested
    skip_merge = master_exists and not (len(q_formats) == 1 and "None (FP16 Master)" in q_formats)
    
    try:
        if skip_merge:
            log_acc += f"⚡ FAST TRACK: Found existing Master: {cache_name}\n"
            log_acc += "⏭️ Skipping Merge Loop and jumping to Batch Export...\n\n"
            yield log_acc, "", "Fast Tracking..."
        else:
            # --- 1. SETUP & ENGINE INIT ---
            progress(0.05, desc="Initializing Engine...")
            clean_json = re.sub(r'//.*', '', recipe_json)
            recipe_dict = json.loads(clean_json)
            recipe_dict['paths'] = recipe_dict.get('paths', {})
            recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
            recipe_dict['paths']['title'] = recipe_dict['paths'].get('title', recipe_slug)
            
            engine = ActionMasterEngine(recipe_dict)

            # Compatibility Reporting
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

            # --- 3. FINAL INTEGRITY CHECK (PRE-SAVE GATE) ---
            progress(0.65, desc="Verifying 14B Integrity...")
            log_acc += "\n" + "="*60 + "\n"
            from utils import verify_model_integrity
            
            try:
                for diag_msg in verify_model_integrity(engine.base_dict, engine.base_keys, engine.router_regex):
                    log_acc += diag_msg + "\n"
                    yield log_acc, "", "🛡️ Verifying Tensors..."
            except Exception as integrity_error:
                log_acc += f"\n🔥 INTEGRITY FAILURE: {str(integrity_error)}\n"
                yield log_acc, "", "Verification Failed"
                return 

            # --- 4. SAVE MASTER SOURCE ---
            progress(0.75, desc="Finalizing Master File...")
            summary_table = get_final_summary_string(engine.summary_data, engine.role_label)
            log_acc += "\n" + summary_table + "\n"
            log_acc += f"💾 EXPORT: Writing Source Master to SSD: {temp_path}...\n"
            yield log_acc, "", "💾 WRITING MASTER..." 
            
            engine.save_master(temp_path) 
            
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            log_acc += f"✅ SOURCE MASTER READY.\n\n"

        # --- 5. BATCH EXPORT LOOP (With Error Isolation) ---
        log_acc += "📦 STARTING BATCH EXPORT QUEUE\n" + "-"*60 + "\n"
        
        for idx, fmt in enumerate(q_formats):
            progress(0.8 + (idx/len(q_formats) * 0.2), desc=f"Exporting {fmt}...")
            
            if fmt == "None (FP16 Master)":
                log_acc += "🔹 FP16 Master: Verified (Source File)\n"
                yield log_acc, temp_path, "FP16 Ready"
                continue

            try:
                log_acc += f"🚀 Processing Format: {fmt}...\n"
                
                if "GGUF_" in fmt:
                    q_type = fmt.replace("GGUF_", "")
                    final_name = f"WAN22_{recipe_slug}_{q_type}.gguf"
                    final_path = os.path.join(final_dir, final_name)
                    cmd = ["python", "convert.py", "--src", temp_path, "--dst", final_path, "--outtype", q_type]
                else:
                    final_name = f"WAN22_{recipe_slug}_{fmt}.safetensors"
                    final_path = os.path.join(final_dir, final_name)
                    cmd = ["convert_to_quant", "-i", temp_path, "-o", final_path, "--comfy_quant", "--wan"]
                    if fmt == "int8": cmd += ["--int8", "--block_size", "128"]
                    elif fmt == "nvfp4": cmd += ["--nvfp4"]

                for update in execute_export_logic(cmd, final_name, final_path, fmt, auto_move, final_dir, log_acc):
                    if isinstance(update, tuple):
                        log_acc = update[0]
                        yield update
                
                log_acc += f"✅ Successfully Finished: {final_name}\n"

            except Exception as item_error:
                log_acc += f"❌ ERROR in {fmt}: {str(item_error)}\n"
                log_acc += "⚠️ Skipping to next format in queue...\n"
                yield log_acc, "", f"Skipped {fmt}"
            
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        log_acc += f"\n{'='*60}\n✨ BATCH TASKS FINISHED.\n{'='*60}\n"

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
                    choices=["None (FP16 Master)", "GGUF_Q4_K_M", "GGUF_Q8_0", "int8", "nvfp4"],
                    label="Batch Export Formats",
                    value=["None (FP16 Master)"]
                )
            with gr.Row():
                start_btn = gr.Button("🔥 START", variant="primary", scale=2)
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
    
    # Logic for Start/Stop
    run_event = start_btn.click(
        fn=run_pipeline, 
        inputs=[recipe_editor, base_dd, quant_select, recipe_dd], # 4 args
        outputs=[terminal_box, last_path_state, pipeline_status],
        show_progress="hidden"
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