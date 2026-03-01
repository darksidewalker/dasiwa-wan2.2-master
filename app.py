import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info, instant_validate, get_final_summary_string
from engine import ActionMasterEngine

# Global handle for the background process
active_process = None
ensure_dirs()

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
        active_process.kill() 
        active_process = None
    torch.cuda.empty_cache()
    gc.collect()
    return "🛑 BASH PROCESS TERMINATED\n" + "-"*60, "Idle"

def run_pipeline(recipe_json, base_model, q_formats, recipe_name):
    global active_process
    if not q_formats:
        yield "❌ ERROR: No export formats selected.", "", "Idle"
        return

    progress = gr.Progress() 
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] ⚜️ DaSiWa STATION MASTER ACTIVE\n" + "="*60 + "\n"
    
    recipe_slug = recipe_name.replace(".json", "") if recipe_name else "custom_merge"
    temp_path = os.path.join(MODELS_DIR, f"MASTER_{recipe_slug}.safetensors")
    
    # Tool Path Resolution
    ROOT_DIR = os.getcwd()
    LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    CONVERT_PY = os.path.join(ROOT_DIR, "convert.py")
    FIX_5D_PY = os.path.join(ROOT_DIR, "fix_5d_tensors.py")
    
    try:
        # --- 1. ENGINE INITIALIZATION ---
        yield log_acc + "Initializing Engine (FP32 Mode)...\n", "", "Initializing..."
        recipe_dict = json.loads(re.sub(r'//.*', '', recipe_json))
        recipe_dict['paths'] = recipe_dict.get('paths', {})
        recipe_dict['paths']['base_model'] = os.path.join(MODELS_DIR, base_model)
        
        engine = ActionMasterEngine(recipe_dict)
        
        # UI Validation Report
        mismatches = engine.get_compatibility_report()
        log_acc += f"\n{'='*60}\n🛡️  RECIPE VALIDATION: {engine.role_label}\n{'='*60}\n"
        if mismatches:
            for m in mismatches: log_acc += f"   - [WARN] {m}\n"
        else:
            log_acc += "✅ ALL SYSTEMS CLEAR: Alignment Verified.\n"
        log_acc += f"{'='*60}\n\n"
        yield log_acc, "", "Merging..."

        # --- 2. MULTI-ROUND SECTIONAL MERGE ---
        pipeline = recipe_dict.get('pipeline', [])
        total_steps = len(pipeline)

        for idx, step in enumerate(pipeline):
            current_status = f"Section {idx+1}/{total_steps}: {step.get('pass_name', 'Unknown')}"
            
            # Internal engine call ensures FP32 precision and bit-parity with Silveroxides
            for message in engine.process_pass(step, recipe_dict['paths'].get('global_weight_factor', 1.0)):
                log_acc += message + "\n"
                yield log_acc, "", current_status

        # --- 3. FINAL INTEGRITY CHECK ---
        progress(0.7, desc="🛡️ Integrity Scan")
        for diag_msg in engine.run_pre_save_check():
            log_acc += diag_msg + "\n"
            yield log_acc, "", "🛡️ Verifying..."

        # --- 4. STREAMING SAVE (FP32 -> BF16) ---
        progress(0.85, desc="💾 Writing Master")
        engine.save_master(temp_path)
        
        # Get final Shift Data from Engine
        log_acc += engine.get_final_summary("BF16") + "\n"
        
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        log_acc += "✅ SOURCE MASTER READY.\n\n"

        # --- 5. BATCH EXPORT QUEUE ---
        quants_to_process = [f for f in q_formats if "None" not in f]
        if not quants_to_process:
            log_acc += "✨ Process Finished. No quantizations requested."
            yield log_acc, temp_path, "Idle"
            return

        log_acc += f"📦 STARTING BATCH EXPORT QUEUE ({len(quants_to_process)} formats)\n" + "-"*60 + "\n"
        
        for idx, fmt in enumerate(q_formats):
            if "None" in fmt: continue
            batch_status = f"Exporting {fmt} ({idx+1}/{len(q_formats)})"
            
            if "GGUF_" in fmt:
                q_type = fmt.replace("GGUF_", "")
                final_path = temp_path.replace(".safetensors", f"-{q_type}.gguf")
                bf16_gguf = temp_path.replace(".safetensors", "-BF16.gguf")
                
                # Step 1: Convert to GGUF (BF16)
                log_acc += f"📦 Converting to BF16 GGUF...\n"
                yield log_acc, "", batch_status
                active_process = subprocess.Popen(["python", CONVERT_PY, "--src", temp_path])
                active_process.wait()

                # Step 2: Llama-Quantize
                log_acc += f"🔨 Quantizing to {q_type}...\n"
                yield log_acc, "", batch_status
                active_process = subprocess.Popen([LLAMA_BIN, bf16_gguf, final_path, q_type])
                active_process.wait()

                # Step 3: Fix 5D Tensors (Surgical Fix for Wan 2.1/2.2)
                log_acc += f"🔧 Applying 5D Expert Tensor Fix...\n"
                yield log_acc, "", batch_status
                active_process = subprocess.Popen(["python", FIX_5D_PY, "--src", final_path, "--dst", final_path])
                active_process.wait()
                
                if os.path.exists(bf16_gguf): os.remove(bf16_gguf)
            else:
                # Standard FP8/INT8 logic
                suffix = fmt.lower().replace(" ", "_")
                final_path = temp_path.replace(".safetensors", f"_{suffix}.safetensors")
                cmd = ["convert_to_quant", "-i", temp_path, "-o", final_path, "--comfy_quant", "--wan"]
                if "int8" in fmt.lower(): cmd += ["--int8", "--block_size", "128"]
                elif "nvfp4" in fmt.lower(): cmd += ["--nvfp4"]
                
                log_acc += f"🚀 Running {fmt} export...\n"
                yield log_acc, "", batch_status
                active_process = subprocess.Popen(cmd)
                active_process.wait()

            log_acc += f"✅ FINISHED: {fmt}\n"
            yield log_acc, final_path, batch_status

        log_acc += "\n✨ ALL TASKS COMPLETED SUCCESSFULLY."
        yield log_acc, temp_path, "Idle"

    except Exception as e:
        yield log_acc + f"\n🔥 CRITICAL FAILURE: {str(e)}", "", "Error"
    finally:
        active_process = None
        torch.cuda.empty_cache()

# --- GRADIO UI LAYOUT (Gradio 6 Optimized) ---
with gr.Blocks(title="ActionMaster STATION") as demo:
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# ⚜️ ActionMaster STATION MASTER\n**High-Precision Wan 2.1/2.2 14B Merging Engine**")
        with gr.Column(scale=3):
            vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=3, interactive=False)
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        with gr.Column(scale=3):
            pipeline_status = gr.Label(label="Process State", value="Idle")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                base_dd = gr.Dropdown(label="Base Model (Safetensors)", allow_custom_value=True)
                recipe_dd = gr.Dropdown(label="Active Recipe (JSON)", allow_custom_value=True)
                val_status_display = gr.Markdown("### 🛡️ Status: Standby")
                refresh_btn = gr.Button("🔄 Refresh Assets", size="sm")
            with gr.Group():
                q_format = gr.CheckboxGroup(
                    choices=[
                        "None (BF16 Master)", "FP8 (SVD)", "INT8 (Block-wise)", 
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", "GGUF_Q4_K_M"
                    ],
                    label="Export Formats",
                    value=["None (BF16 Master)"]
                )
            with gr.Row():
                run_btn = gr.Button("🧩 RUN PIPELINE", variant="primary", elem_classes=["primary-button"])
                stop_btn = gr.Button("🛑 STOP", variant="stop")
            last_path_state = gr.State("")

        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("💻 Terminal Log"):
                    terminal_box = gr.Textbox(lines=28, interactive=False, show_label=False, elem_id="terminal")
                with gr.Tab("📝 Recipe Editor"):
                    recipe_editor = gr.Code(language="json", lines=28)

    # --- BINDINGS ---
    demo.load(list_files, outputs=[base_dd, recipe_dd])
    refresh_btn.click(list_files, outputs=[base_dd, recipe_dd])
    recipe_dd.change(load_recipe_text, inputs=[recipe_dd], outputs=[recipe_editor])
    recipe_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    base_dd.change(instant_validate, inputs=[recipe_dd, base_dd], outputs=[val_status_display])
    
    run_event = run_btn.click(
        fn=run_pipeline,
        inputs=[recipe_editor, base_dd, q_format, recipe_dd],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status], cancels=[run_event])
    
    # Auto-scroll JS binding
    terminal_box.change(fn=None, js="(x) => { document.getElementById('terminal').querySelector('textarea').scrollTop = 9999999; }")

if __name__ == "__main__":
    demo.launch(css=CSS_STYLE)