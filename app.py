import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info, instant_validate, get_final_summary_string, sync_ram_to_ssd
from engine import ActionMasterEngine

active_process = None
ensure_dirs()

def list_files():
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(('.safetensors', '.gguf'))])
    r = sorted([f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')])
    return gr.update(choices=m), gr.update(choices=r)

def load_recipe_text(name):
    if not name: return ""
    with open(os.path.join(RECIPES_DIR, name), 'r') as f: return f.read()

def run_pipeline(recipe_json, base_model, q_format, recipe_name, auto_move, progress=gr.Progress()):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "="*60 + "\n"
    global active_process
    
    # PATH SEPARATION
    recipe_slug = recipe_name.replace(".json", "")
    cache_name = f"MASTER_{recipe_slug}.safetensors"
    temp_path = os.path.join(MODELS_DIR, cache_name) # Always SSD
    final_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR # Always RAM Disk
    
    # SMART SKIP LOGIC
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
            yield log_acc, "", "Initializing Engine..."
            
            clean_json = re.sub(r'//.*', '', recipe_json)
            recipe_dict = json.loads(clean_json)
            
            recipe_dict['paths'] = recipe_dict.get('paths', {})
            recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
            recipe_dict['paths']['title'] = recipe_dict['paths'].get('title', recipe_slug)
            
            engine = ActionMasterEngine(recipe_dict)

            # VALIDATION HEADER
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
            log_acc += get_final_summary_string(engine.summary_data, engine.role_label) + "\n"
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
                # Use --src and --dst as required by convert.py
                cmd = ["python", "convert.py", "--src", temp_path, "--dst", final_path, "--outtype", q_type]
            else:
                final_name = f"WAN22_{recipe_slug}_{q_format}.safetensors"
                final_path = os.path.join(final_dir, final_name)
                
                yield log_acc + f"🔨 PIP PACKAGE: {q_format.upper()} -> {final_path}\n", "", f"Quantizing {q_format}..."
                
                # SPECIFIC QUANT FLAGS
                cmd = ["convert_to_quant", "-i", temp_path, "-o", final_path, "--comfy_quant", "--wan"]
                if q_format == "int8":
                    cmd += ["--int8", "--block_size", "128"]
                elif q_format == "nvfp4":
                    cmd += ["--nvfp4"]

            subprocess.run(cmd, check=True)
            
            # AUTO MOVE
            if auto_move and final_dir == RAMDISK_PATH:
                shutil.move(final_path, os.path.join(MODELS_DIR, final_name))
                final_path = os.path.join(MODELS_DIR, final_name)

            log_acc += f"✅ EXPORT COMPLETE: {final_name}\n"
            yield log_acc, final_path, "Process Finished"

            log_filename = f"merge_{recipe_slug}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            log_path = os.path.join(LOGS_DIR, log_filename)

            # Write the terminal feed to the logs folder
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_acc)

        else:
            yield log_acc + "✅ MASTER FP16 READY ON SSD.\n", temp_path, "Finished"

    except Exception as e:
        log_acc += f"\n🔥 CRITICAL FAILURE: {str(e)}\n"
        yield log_acc, "", "Critical Error"
    finally:
        active_process = None

with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# ⚜️ DaSiWa WAN 2.2 Master\n**14B High-Precision MoE Pipeline**")
        with gr.Column(scale=3):
            with gr.Group(elem_classes="vitals-card"):
                vitals_box = gr.Textbox(label="System Health", value=get_sys_info(), lines=3, interactive=False)
                gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        with gr.Column(scale=3):
            with gr.Group(elem_classes="vitals-card"):
                pipeline_status = gr.Label(label="Current Stage", value="Idle")
                main_progress = gr.Progress(track_tqdm=True)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                base_dd = gr.Dropdown(label="Base Model")
                recipe_dd = gr.Dropdown(label="Active Recipe")
                val_status_display = gr.Markdown("### 🛡️ Status: No Recipe Selected")
                refresh_btn = gr.Button("🔄 Refresh Assets", size="sm")
            with gr.Group():
                quant_select = gr.Radio(
                    choices=[
                        "None (FP16 Master)", 
                        "fp8", "nvfp4", "int8", 
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", 
                        "GGUF_Q4_K_M", "GGUF_Q3_K_M", "GGUF_Q2_K"
                    ], 
                    value="None (FP16 Master)", 
                    label="Target Format"
                )
                auto_move_toggle = gr.Checkbox(label="🚀 Move to SSD on Success", value=False)
            with gr.Row():
                start_btn = gr.Button("🔥 START", variant="primary", scale=2)
                stop_btn = gr.Button("🛑 STOP", variant="stop", scale=1)
            last_path_state = gr.State("")

        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("💻 Terminal Feed"): terminal_box = gr.Textbox(lines=28, interactive=False, elem_id="terminal", show_label=False)
                with gr.Tab("📝 Recipe Editor"): recipe_editor = gr.Code(language="json", lines=28)

    # Bindings
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    recipe_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    start_btn.click(fn=run_pipeline, inputs=[recipe_editor, base_dd, quant_select, recipe_dd, auto_move_toggle], outputs=[terminal_box, last_path_state, pipeline_status])
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL)
    sync_trigger.click(fn=sync_ram_to_ssd, inputs=[last_path_state], outputs=[terminal_box])

# --- THE CLEAN SHUTDOWN ---
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