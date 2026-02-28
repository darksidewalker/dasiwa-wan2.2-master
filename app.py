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

def run_pipeline(recipe_json, base_model, q_format, recipe_name, auto_move, progress=gr.Progress()):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "="*60 + "\n"
    global active_process
    
    recipe_slug = recipe_name.replace(".json", "") if recipe_name else "custom_merge"
    cache_name = f"MASTER_{recipe_slug}.safetensors"
    temp_path = os.path.join(MODELS_DIR, cache_name)
    final_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
    
    master_exists = os.path.exists(temp_path) and os.path.getsize(temp_path) > 1e9
    skip_merge = master_exists and q_format != "None (FP16 Master)"
    
    try:
        if skip_merge:
            log_acc += f"⚡ FAST TRACK: Found existing Master: {cache_name}\n"
            log_acc += "⏭️ Skipping Merge Loop and jumping to Quantization...\n\n"
            yield log_acc, "", "Fast Tracking..."
        else:
            # 1. SETUP & ENGINE INIT
            progress(0.05, desc="Initializing Engine...")
            clean_json = re.sub(r'//.*', '', recipe_json)
            recipe_dict = json.loads(clean_json)
            recipe_dict['paths'] = recipe_dict.get('paths', {})
            recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
            recipe_dict['paths']['title'] = recipe_dict['paths'].get('title', recipe_slug)
            
            engine = ActionMasterEngine(recipe_dict)

            # VALIDATION HEADER
            mismatches = engine.get_compatibility_report()
            log_acc += f"\n{'='*60}\n🛡️  RECIPE VALIDATION: {engine.role_label}\n{'='*60}\n"
            if mismatches:
                log_acc += f"❌ CONFLICT: {len(mismatches)} LoRA(s) mismatch noise levels!\n"
                for m in mismatches: log_acc += f"   - [WARN] {m}\n"
            else:
                log_acc += "✅ ALL SYSTEMS CLEAR: Alignment Verified.\n"
            log_acc += f"{'='*60}\n\n"
            yield log_acc, "", "Merging Layers..."

            # 2. MERGING LOOP (CLEAN & INTERRUPTIBLE)
            pipeline = recipe_dict.get('pipeline', [])
            global_mult = recipe_dict['paths'].get('global_weight_factor', 1.0)

            for i, step in enumerate(pipeline):
                p_name = step.get('pass_name', f"Pass {i+1}")
                progress(0.1 + (i/len(pipeline) * 0.6), desc=f"Merging {p_name}")
                
                # RUN THE ENGINE ONCE
                # This loop handles all the real-time "Analyzing..." messages
                for message in engine.process_pass(step, global_mult):
                    log_acc += message + "\n"
                    yield log_acc, "", f"Working: {p_name}"
                
                # ACCESS SUMMARY ONLY AFTER THE ENGINE FINISHES THE PASS
                if engine.summary_data:
                    last_pass = engine.summary_data[-1]
                    peak_str = f" | ⚠️ PEAKS: {last_pass['peaks']}" if last_pass['peaks'] > 0 else ""
                    
                    log_acc += "-"*60 + "\n"
                    
                    yield log_acc, "", f"Finished: {p_name}"
                else:
                    log_acc += f"⚠️ {p_name} produced no summary data. Check LoRA files.\n"
                    yield log_acc, "", "Warning: Empty Pass"

            # 3. SAVE MASTER (SSD)
            progress(0.8, desc="Finalizing Tensors...")
            summary_table = get_final_summary_string(engine.summary_data, engine.role_label)
            log_acc += summary_table + "\n"

            log_acc += f"💾 EXPORT: Writing 14B Master to SSD: {temp_path}...\n"
            log_acc += "⚠️ SYSTEM MAY BECOME UNRESPONSIVE DURING WRITE\n"
            yield log_acc, "", "💾 WRITING TO SSD..." # This keeps the UI active
            
            engine.save_master(temp_path) 
            
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            
            log_acc += f"✅ MASTER SAVED SUCCESSFULLY.\n"
            yield log_acc, temp_path, "Process Finished"

        match q_format:
            case "None (FP16 Master)":
                yield log_acc + "✅ MASTER FP16 READY ON SSD.\n", temp_path, "Finished"
            
            case str(f) if "GGUF_" in f:
                q_type = f.replace("GGUF_", "")
                final_name = f"WAN22_{recipe_slug}_{q_type}.gguf"
                final_path = os.path.join(final_dir, final_name)
                cmd = ["python", "convert.py", "--src", temp_path, "--dst", final_path, "--outtype", q_type]
                yield from execute_export_logic(cmd, final_name, final_path, q_format, auto_move, final_dir, log_acc)
            
            case _:
                final_name = f"WAN22_{recipe_slug}_{q_format}.safetensors"
                final_path = os.path.join(final_dir, final_name)
                cmd = ["convert_to_quant", "-i", temp_path, "-o", final_path, "--comfy_quant", "--wan"]
                if q_format == "int8": cmd += ["--int8", "--block_size", "128"]
                elif q_format == "nvfp4": cmd += ["--nvfp4"]
                yield from execute_export_logic(cmd, final_name, final_path, q_format, auto_move, final_dir, log_acc)

    except Exception as e:
        log_acc += f"\n🔥 CRITICAL FAILURE: {str(e)}\n"
        yield log_acc, "", "Critical Error"
    finally:
        try:
            log_filename = f"merge_{recipe_slug}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
    if auto_move and final_dir == RAMDISK_PATH:
        shutil.move(final_path, os.path.join(MODELS_DIR, final_name))
        final_path = os.path.join(MODELS_DIR, final_name)
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
                quant_select = gr.Radio(
                    choices=["None (FP16 Master)", "fp8", "nvfp4", "int8", "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q4_K_M", "GGUF_Q2_K"], 
                    value="None (FP16 Master)", label="Target Format"
                )
                auto_move_toggle = gr.Checkbox(label="🚀 Move to SSD on Success", value=False)
            with gr.Row():
                start_btn = gr.Button("🔥 START", variant="primary", scale=2)
                stop_btn = gr.Button("🛑 STOP", variant="stop", scale=1)
            
            sync_trigger = gr.Button("📤 Manual Move to SSD", variant="secondary")
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
        inputs=[recipe_editor, base_dd, quant_select, recipe_dd, auto_move_toggle], 
        outputs=[terminal_box, last_path_state, pipeline_status],
        show_progress="hidden"
    )
    
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status], cancels=[run_event])
    sync_trigger.click(fn=sync_ram_to_ssd, inputs=[last_path_state], outputs=[terminal_box])
    
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