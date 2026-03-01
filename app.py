import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info

# Global handle for the background process
active_process = None
ensure_dirs()

def list_files():
    # Only show safetensors as source for conversion
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')])
    return gr.update(choices=m)

def stop_pipeline():
    global active_process
    if active_process:
        active_process.kill() 
        active_process = None
    torch.cuda.empty_cache()
    gc.collect()
    return "🛑 CONVERSION TERMINATED\n" + "-"*60, "Idle"

def run_conversion(base_model, q_formats):
    global active_process
    if not q_formats:
        yield "❌ ERROR: No export formats selected.", "", "Idle"
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] 📦 Q-QUANT STATION ACTIVE\n" + "="*60 + "\n"
    
    source_path = os.path.join(MODELS_DIR, base_model)
    ROOT_DIR = os.getcwd()
    LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    CONVERT_PY = os.path.join(ROOT_DIR, "convert.py")
    FIX_5D_PY = os.path.join(ROOT_DIR, "fix_5d_tensors.py")
    
    try:
        log_acc += f"Source Model: {base_model}\n\n"
        yield log_acc, "", "Processing..."

        for idx, fmt in enumerate(q_formats):
            batch_status = f"Exporting {fmt} ({idx+1}/{len(q_formats)})"
            
            # --- GGUF Q-QUANT PIPELINE ---
            if "GGUF_" in fmt:
                q_type = fmt.replace("GGUF_", "")
                final_path = source_path.replace(".safetensors", f"-{q_type}.gguf")
                bf16_gguf = source_path.replace(".safetensors", "-BF16.gguf")
                
                steps = [
                    (f"📦 Step 1: Converting to BF16 GGUF...", ["python", CONVERT_PY, "--src", source_path]),
                    (f"🔨 Step 2: Llama-Quantize to {q_type}...", [LLAMA_BIN, bf16_gguf, final_path, q_type]),
                    (f"🔧 Step 3: Applying 5D Expert Tensor Fix...", ["python", FIX_5D_PY, "--src", final_path, "--dst", final_path])
                ]
                
                for step_msg, cmd in steps:
                    log_acc += f"{step_msg}\n"
                    yield log_acc, "", batch_status
                    active_process = subprocess.Popen(cmd)
                    active_process.wait()
                    if active_process.returncode != 0:
                        log_acc += f"❌ FAILED at {step_msg}\n"
                        break
                
                if os.path.exists(bf16_gguf): os.remove(bf16_gguf)

            # --- SAFETENSORS QUANTIZATION (FP8/INT8/NVFP4) ---
            else:
                # Clean suffix naming
                suffix = fmt.lower().replace(" ", "_").split("(")[0].strip()
                final_path = source_path.replace(".safetensors", f"_{suffix}.safetensors")
                
                cmd = ["convert_to_quant", "-i", source_path, "-o", final_path, "--comfy_quant", "--wan"]
                if "int8" in fmt.lower(): cmd += ["--int8", "--block_size", "128"]
                elif "nvfp4" in fmt.lower(): cmd += ["--nvfp4"]
                
                log_acc += f"🚀 Running {fmt} export...\n"
                yield log_acc, "", batch_status
                active_process = subprocess.Popen(cmd)
                active_process.wait()

            log_acc += f"✅ FINISHED: {fmt}\n"
            yield log_acc, final_path, batch_status

        log_acc += "\n✨ ALL QUANTIZATIONS COMPLETE."
        yield log_acc, source_path, "Idle"

    except Exception as e:
        yield log_acc + f"\n🔥 ERROR: {str(e)}", "", "Error"
    finally:
        active_process = None

# --- UI LAYOUT ---
with gr.Blocks(title="Conversion Station", css=CSS_STYLE) as demo:
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# 📦 DaSiWa Quant Station\n**Direct GGUF & Safetensors Quantization**")
        with gr.Column(scale=3):
            vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=3, interactive=False, elem_classes=["vitals-card"])
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        with gr.Column(scale=3):
            pipeline_status = gr.Label(label="Process State", value="Idle")

    with gr.Row():
        with gr.Column(scale=3):
            base_dd = gr.Dropdown(label="Select Source Safetensors", allow_custom_value=True)
            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            with gr.Group():
                gr.Markdown("### ⚖️ Select Formats")
                q_format = gr.CheckboxGroup(
                    choices=[
                        "FP8 (SVD)", "INT8 (Block-wise)", "NVFP4",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", "GGUF_Q5_0",
                        "GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q3_K_M", "GGUF_Q2_K"
                    ],
                    label="Available Quants",
                    value=["FP8 (SVD)"]
                )
            
            with gr.Row():
                run_btn = gr.Button("🧩 START BATCH", variant="primary", elem_classes=["primary-button"])
                stop_btn = gr.Button("🛑 STOP", variant="stop", elem_classes=["stop-button"])
            
            last_path_state = gr.State("")

        with gr.Column(scale=7):
            terminal_box = gr.Textbox(lines=26, interactive=False, show_label=False, elem_id="terminal")

    # --- BINDINGS ---
    demo.load(list_files, outputs=[base_dd])
    refresh_btn.click(list_files, outputs=[base_dd])
    
    run_btn.click(
        fn=run_conversion,
        inputs=[base_dd, q_format],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status])
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL, inputs=[terminal_box])

if __name__ == "__main__":
    demo.launch()