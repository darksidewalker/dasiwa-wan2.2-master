import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info, inject_metadata, get_metadata

# Global handle for the background process
active_process = None
ensure_dirs()

# --- INTERNAL METADATA TEMPLATE ---
METADATA_TEMPLATE = {
    "modelspec.title": "DaSiWa WAN 2.2 I2V {model_name}",
    "modelspec.author": "Darksidewalker",
    "modelspec.description": "Multi-Expert Image-to-Video diffusion model quantized via DaSiWa Station.",
    "modelspec.date": "{date}",
    "modelspec.architecture": "wan_2.2_14b_i2v",
    "modelspec.implementation": "https://github.com/Wan-Video/Wan2.2",
    "modelspec.tags": "image-to-video, moe, diffusion, wan2.2, DaSiWa",
    "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
    "quantization.tool": "https://github.com/darksidewalker/dasiwa-wan2.2-master",
    "quantization.version": "1.0.0",
    "quantization.bits": "{bits}"
}

# --- HELPER FUNCTIONS ---

def list_files():
    """Lists available models in the models directory."""
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')])
    return gr.update(choices=m, value=m[0] if m else None)

def update_metadata_preview(name):
    """Updates the live JSON preview for metadata."""
    if not name or name == "Enter Name...": name = "Model-Name"
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d")
    preview = {}
    for k, v in METADATA_TEMPLATE.items():
        preview[k] = v.replace("{model_name}", name).replace("{date}", curr_date).replace("{bits}", "SELECTED_QUANT")
    return json.dumps(preview, indent=4)

def stop_pipeline():
    global active_process
    if active_process:
        active_process.kill()
        active_process = None
    torch.cuda.empty_cache()
    gc.collect()
    return "🛑 Process Terminated by User.\n" + "-"*60, "Idle"

def handle_injection(file_name, json_str):
    if not file_name: return "❌ Select a model first."
    try:
        data = json.loads(json_str)
        path = os.path.join(MODELS_DIR, file_name)
        success, msg = inject_metadata(path, data)
        return f"✅ {msg}" if success else f"❌ {msg}"
    except Exception as e:
        return f"🔥 JSON Error: {str(e)}"

def read_selected_metadata(file_name):
    if not file_name: return "❌ Select a model first."
    path = os.path.join(MODELS_DIR, file_name)
    meta, err = get_metadata(path)
    if err: return f"❌ Error: {err}"
    return f"🔍 METADATA FOR: {file_name}\n" + "-"*40 + "\n" + json.dumps(meta, indent=4)

# --- CORE CONVERSION ENGINE ---

def run_conversion(base_model, q_formats, model_name, custom_options):
    global active_process
    
    # 1. STANDARD FLAG MAPPING
    FLAG_MAP = {
        "FP8": ["--comfy_quant"],
        "INT8 Block-wise": ["--int8", "--block_size", "128", "--comfy_quant"],
        "NVFP4": ["--nvfp4", "--comfy_quant"],
        "Fast Mode (Simple)": ["--simple"],
        "Auto-Quality (Heur)": ["--heur"],
        "Low Memory Mode": ["--low-memory"]
    }

    # 2. ULTRA-QUALITY OPTIMIZER PRESET
    ULTRA_PRESET = [
        "--comfy_quant", 
        "--save-quant-metadata", 
        "--wan", 
        "--lr_schedule", "plateau", 
        "--lr_patience", "2", 
        "--lr_factor", "0.96", 
        "--lr_min", "9e-9", 
        "--lr_cooldown", "0", 
        "--lr_threshold", "1e-11", 
        "--num_iter", "9000", 
        "--calib_samples", "45000", 
        "--lr", "9.916700000002915715e-3", 
        "--top_p", "0.05", 
        "--min_k", "64", 
        "--max_k", "256", 
        "--early-stop-stall", "20000", 
        "--early-stop-lr", "1e-8", 
        "--early-stop-loss", "9e-8", 
        "--lr-shape-influence", "3.5"
    ]

    if not q_formats:
        yield "❌ ERROR: No export formats selected.", "", "Idle"
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] 📦 Q-QUANT STATION ACTIVE\n" + "="*60 + "\n"
    source_path = os.path.join(MODELS_DIR, base_model)
    
    try:
        for idx, fmt in enumerate(q_formats):
            batch_status = f"Exporting {fmt} ({idx+1}/{len(q_formats)})"
            suffix = fmt.lower().replace(" ", "_").split("(")[0].strip()
            final_path = source_path.replace(".safetensors", f"_{suffix}.safetensors")
            
            cmd = ["convert_to_quant", "-i", source_path, "-o", final_path]

            # --- CASE LOGIC: ULTRA VS STANDARD ---
            if "Ultra-Quality (Optimizer)" in custom_options:
                log_acc += "💎 ULTRA MODE ACTIVE: Using surgical optimizer settings...\n"
                cmd.extend(ULTRA_PRESET)
            else:
                # Apply standard selected flags
                if fmt in FLAG_MAP: cmd.extend(FLAG_MAP[fmt])
                if custom_options:
                    for opt in custom_options:
                        if opt in FLAG_MAP: cmd.extend(FLAG_MAP[opt])
                # Auto-Wan Detection if not in Ultra
                if "wan" in base_model.lower(): cmd.append("--wan")

            log_acc += f"🚀 Command: {' '.join(cmd)}\n"
            yield log_acc, "", batch_status
            
            active_process = subprocess.Popen(cmd)
            active_process.wait()

            # Metadata Injection Step
            if os.path.exists(final_path) and final_path.endswith('.safetensors'):
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                meta = {k: v.replace("{model_name}", model_name).replace("{date}", current_date).replace("{bits}", fmt) for k, v in METADATA_TEMPLATE.items()}
                success, msg = inject_metadata(final_path, meta)
                log_acc += f"📝 Metadata: {msg}\n"

            log_acc += f"✅ FINISHED: {fmt}\n"
            yield log_acc, final_path, batch_status

        log_acc += "\n✨ ALL BATCH PROCESSES COMPLETE."
        yield log_acc, source_path, "Idle"

    except Exception as e:
        yield log_acc + f"\n🔥 ERROR: {str(e)}", "", "Error"
    finally:
        active_process = None

# --- UI LAYOUT ---

with gr.Blocks(title="Conversion Station") as demo:
    # Row 1: App Header
    with gr.Row():
        gr.Markdown("# 📦 DaSiWa Quant Station\n**Direct GGUF & Safetensors Quantization for Wan 2.2**")

    # Row 2: Control Panel (3-Column Grid)
    with gr.Row():
        # Column 1: Source Files
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### 📥 Source Settings")
                base_dd = gr.Dropdown(label="Select Source Safetensors", allow_custom_value=True)
                friendly_name = gr.Textbox(label="Model Display Name", placeholder="e.g. Cinema-Mix-V1")
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            with gr.Row():
                run_btn = gr.Button("🧩 START BATCH", variant="primary", elem_classes=["primary-button"])
                stop_btn = gr.Button("🛑 STOP", variant="stop", elem_classes=["stop-button"])
            
            last_path_state = gr.State("")

        # Column 2: Quantization Formats (Full List)
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### ⚖️ Select Formats")
                q_format = gr.CheckboxGroup(
                    choices=[
                        "FP8", "INT8 Block-wise", "NVFP4",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", "GGUF_Q5_0",
                        "GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q3_K_M", "GGUF_Q2_K"
                    ],
                    label="Target Format",
                    info="Choose precision level. GGUF requires a multi-step build.",
                    value=["Standard FP8 (ComfyUI)"]
                )

        # Column 3: Optimizations & Vitals
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 🛠️ Optimization Flags")
                extra_flags = gr.CheckboxGroup(
                    choices=["Ultra-Quality (Optimizer)", "Auto-Quality (Heur)", "Fast Mode (Simple)", "Low Memory Mode"],
                    label="Quantization Tweaks",
                    info="For FP-Quants - Ultra: 9k iterations (Pro Quality). Heur: Fixes black screens.",
                    value=["Auto-Quality (Heur)"]
                )
            
            vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=2, interactive=False, elem_classes=["vitals-card"])
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
            pipeline_status = gr.Label(label="Process State", value="Idle")

    # Row 3: Output Terminal & Metadata Tools
    with gr.Row():
        with gr.Column(scale=6):
            terminal_box = gr.Textbox(lines=22, interactive=False, show_label=False, elem_id="terminal")
        
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 📝 Metadata Injector")
                metadata_input = gr.Code(
                    value=update_metadata_preview("Enter Name..."), 
                    language="json",
                    label="Header Preview",
                    interactive=True
                )
                with gr.Row():
                    inject_btn = gr.Button("💉 Inject to Source")
                    read_btn = gr.Button("🔍 Read Current Header")

    # --- BINDINGS ---
    demo.load(list_files, outputs=[base_dd])
    friendly_name.change(fn=update_metadata_preview, inputs=[friendly_name], outputs=[metadata_input])
    
    run_btn.click(
        fn=run_conversion,
        inputs=[base_dd, q_format, friendly_name, extra_flags],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status])
    refresh_btn.click(list_files, outputs=[base_dd])
    inject_btn.click(fn=handle_injection, inputs=[base_dd, metadata_input], outputs=[terminal_box])
    read_btn.click(fn=read_selected_metadata, inputs=[base_dd], outputs=[terminal_box])

    # Auto-scroll terminal
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL, inputs=[terminal_box])

if __name__ == "__main__":
    demo.launch(css=CSS_STYLE)