import gradio as gr
import psutil, torch, time, os, json, re, gc, sys, subprocess, shutil
import gguf
from tqdm import tqdm
from safetensors.torch import load_file
from engine import ActionMasterEngine

# --- Configuration ---
RECIPES_DIR = "recipes"
MODELS_DIR = "models"
RAMDISK_PATH = "/mnt/ramdisk"
VENV_PYTHON = sys.executable

VENV_BIN = os.path.join(os.path.dirname(sys.executable), "convert_to_quant")

for d in [RECIPES_DIR, MODELS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

class ModelWan:
    arch = "wan"
    keys_hiprec = [".modulation", "text_embedding.2.weight", "pos_embedder", "head.modulation"]
    MAX_DIMS = 4

# --- Utility Functions ---
def get_ramdisk_status():
    if not os.path.exists(RAMDISK_PATH): return "RD: N/A"
    try:
        st = os.statvfs(RAMDISK_PATH)
        free = (st.f_bavail * st.f_frsize) / (1024**3)
        total = (st.f_blocks * st.f_frsize) / (1024**3)
        return f"RD: {total-free:.1f}/{total:.1f}G"
    except: return "RD: Error"

def clear_ramdisk():
    if os.path.exists(RAMDISK_PATH):
        for f in os.listdir(RAMDISK_PATH):
            try: os.remove(os.path.join(RAMDISK_PATH, f))
            except: pass
        return "✅ RAMDisk Cleared"
    return "❌ No RAMDisk"

def move_rd_to_ssd():
    if not os.path.exists(RAMDISK_PATH): return "❌ RAMDisk not found."
    moved = []
    # Scans for both GGUF and Safetensors (FP8/NV4)
    valid_exts = (".gguf", ".safetensors")
    for f in os.listdir(RAMDISK_PATH):
        if f.endswith(valid_exts):
            src = os.path.join(RAMDISK_PATH, f)
            dst = os.path.join(MODELS_DIR, f)
            try:
                shutil.move(src, dst)
                moved.append(f)
            except Exception as e:
                return f"❌ Move failed: {str(e)}"
    return f"✅ Moved: {', '.join(moved)}" if moved else "ℹ️ No models found in RAMDisk."

def get_sys_info():
    ram = psutil.virtual_memory().percent
    vram = f"{torch.cuda.memory_reserved() / 1e9:.1f}G" if torch.cuda.is_available() else "0G"
    return f"RAM: {ram}% | VRAM: {vram} | {get_ramdisk_status()}"

def get_model_list():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')]
    return sorted(files) if files else ["No models found"]

def get_recipe_list():
    files = [f for f in os.listdir(RECIPES_DIR) if f.endswith('.json')]
    return sorted(files) if files else ["No recipes found"]

def load_selected_recipe(filename):
    if filename == "No recipes found" or not filename: return ""
    path = os.path.join(RECIPES_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f: return f.read()

def save_active_recipe(filename, content):
    if not filename or filename == "No recipes found": return "❌ Select a name."
    path = os.path.join(RECIPES_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    return f"✅ Saved to {filename}"

# --- Merger Logic ---
def run_merge_pipeline(editor_content, model_filename, progress=gr.Progress()):
    if not editor_content.strip(): return "❌ Editor empty."
    model_path = os.path.join(MODELS_DIR, model_filename)
    try:
        clean_json = re.sub(r'//.*', '', editor_content)
        recipe_data = json.loads(clean_json)
        recipe_data['paths']['base_model'] = model_path

        temp_path = "session_recipe.json"
        with open(temp_path, 'w') as f: json.dump(recipe_data, f)

        gc.collect()
        torch.cuda.empty_cache()

        progress(0, desc="🚀 Initializing Engine...")
        engine = ActionMasterEngine(temp_path)
        logs = [f"🧬 Engine: {'HIGH' if engine.is_high_res else 'LOW'}"]

        for i, step in enumerate(engine.recipe['pipeline']):
            progress((i/len(engine.recipe['pipeline'])), desc=f"Merging Pass {i+1}...")
            engine.process_pass(step, engine.paths.get('global_weight_factor', 1.0))
            logs.append(f"✅ Pass {i+1} complete.")

        final_file = engine.save()
        return "\n".join(logs) + f"\n\n✨ SAVED: {final_file}"
    except Exception as e:
        return f"❌ Merger Error: {str(e)}"

# --- Quantizer Logic ---
def run_unified_quantization(model_filename, quant_choice, keep_in_ram, progress=gr.Progress()):
    logs = []
    five_d_counter = 0
    try:
        src_path = os.path.join(MODELS_DIR, model_filename)
        orig_size = os.path.getsize(src_path) / (1024**3)
        base_name = os.path.splitext(model_filename)[0]

        work_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
        temp_gguf = os.path.join(work_dir, f"{base_name}_{quant_choice}.gguf")
        final_dest = os.path.join(MODELS_DIR, f"{base_name}_{quant_choice}.gguf")

        logs.append(f"📦 Loading: {model_filename}")
        state_dict = load_file(src_path)
        writer = gguf.GGUFWriter(path=temp_gguf, arch=ModelWan.arch)

        target_quant = getattr(gguf.GGMLQuantizationType, quant_choice)
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_file_type(target_quant)

        tensor_keys = list(state_dict.keys())
        for i, key in enumerate(tensor_keys):
            data = state_dict[key]
            progress((i/len(tensor_keys)), desc=f"Quantizing {key[:15]}...")
            data = data.to(torch.float32).numpy() if data.dtype == torch.bfloat16 else data.numpy()

            target_dtype = gguf.GGMLQuantizationType.F32 if (any(hp in key for hp in ModelWan.keys_hiprec) or len(data.shape) == 1) else target_quant

            if len(data.shape) == 5:
                writer.add_array(f"comfy.gguf.orig_shape.{key}", list(data.shape))
                data = data.reshape(-1, *data.shape[-3:])
                five_d_counter += 1

            try:
                writer.add_tensor(key, gguf.quants.quantize(data, target_dtype), raw_dtype=target_dtype)
            except:
                writer.add_tensor(key, gguf.quants.quantize(data, gguf.GGMLQuantizationType.F16), raw_dtype=gguf.GGMLQuantizationType.F16)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        if not keep_in_ram:
            progress(0.99, desc="🚚 Moving to SSD...")
            shutil.move(temp_gguf, final_dest)
            loc = "SSD (models/)"
        else:
            loc = "RAMDisk (/mnt/ramdisk/)"

        final_path = final_dest if not keep_in_ram else temp_gguf
        final_size = os.path.getsize(final_path) / (1024**3)
        logs.append(f"🌀 Tagged {five_d_counter} 5D video tensors.")
        logs.append(f"📊 Size: {orig_size:.1f}GB ➡️ {final_size:.1f}GB")
        return "\n".join(logs) + f"\n\n✨ SUCCESS! Model saved in {loc}"
    except Exception as e: return f"❌ Quant Error: {str(e)}"

def run_fp_quantization(model_filename, format_choice, use_wan_preset, keep_in_ram, progress=gr.Progress()):
    try:
        # MEMORY CHECK: 64GB is tight. We check RD usage first.
        if os.path.exists(RAMDISK_PATH):
            st = os.statvfs(RAMDISK_PATH)
            used_gb = (st.f_blocks - st.f_bavail) * st.f_frsize / (1024**3)
            if used_gb > 1:
                return "❌ RAMDisk must be EMPTY to run FP Quants on a 64GB system. Please click 'Clear RAMDisk' first."

        src_path = os.path.join(MODELS_DIR, model_filename)
        base_name = os.path.splitext(model_filename)[0]

        # We write to RAMDisk for speed, but READ from SSD to save RAM
        work_dir = RAMDISK_PATH if os.path.exists(RAMDISK_PATH) else MODELS_DIR
        temp_output = os.path.join(work_dir, f"{base_name}_{format_choice}.safetensors")
        final_dest = os.path.join(MODELS_DIR, f"{base_name}_{format_choice}.safetensors")

        # Build the CLI command
        # Use VENV_BIN to ensure we hit the local install
        cmd = [VENV_BIN, "-i", src_path, "-o", temp_output]

        # Format mapping logic
        if format_choice == "nvfp4":
            cmd.append("--nvfp4")
        elif format_choice == "mxfp8":
            cmd.append("--mxfp8")

        # Add the specific WAN handling
        if use_wan_preset:
            cmd.append("--wan")

        # Add ComfyUI metadata tagging
        cmd.append("--comfy_quant")

        progress(0, desc="🚀 Starting FP Conversion (Streaming SSD -> RAM)...")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs = []
        for line in process.stdout:
            logs.append(line.strip())
            progress(0.5, desc="Quantizing weights...")

        process.wait()

        if process.returncode == 0:
            if not keep_in_ram:
                progress(0.9, desc="🚚 Moving output to SSD...")
                shutil.move(temp_output, final_dest)
                return f"✅ SUCCESS: Saved to {final_dest}"
            return f"✅ SUCCESS: Saved in RAMDisk: {temp_output}"
        else:
            return f"❌ Error:\n" + "\n".join(logs[-10:])

    except Exception as e:
        return f"❌ Script Error: {str(e)}"

# --- GUI ---
css = "#console_logs textarea { font-family: monospace; color: #00ff88; background: #111; }"

with gr.Blocks(title="DaSiWa WAN 2.2 Master") as demo:
    gr.Markdown("# 🌀 DaSiWa WAN 2.2 Master: Merger & Quantizer")

    with gr.Row():
        sys_info = gr.Textbox(label="Vitals", value=get_sys_info(), interactive=False)
        gr.Timer(2).tick(get_sys_info, outputs=sys_info)

    with gr.Tabs():
        # TAB 1: MODEL MERGER
        with gr.Tab("🧬 Model Merger"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(choices=get_model_list(), label="Base Model")
                    recipe_selector = gr.Dropdown(choices=get_recipe_list(), label="Select Recipe")
                    m_refresh = gr.Button("🔄 Refresh Lists")
                    save_recipe_btn = gr.Button("💾 Save Current Recipe", variant="secondary")

                with gr.Column(scale=3):
                    json_editor = gr.Code(label="Recipe JSON Configuration", language="json", lines=18)
                    run_merge_btn = gr.Button("🔥 START MERGE PIPELINE", variant="primary")
                    merge_output = gr.Textbox(label="Merger Output", lines=10, elem_id="console_logs")

        # TAB 2: GGUF QUANTIZER
        with gr.Tab("📦 GGUF Quantizer"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_model_selector = gr.Dropdown(choices=get_model_list(), label="Source Model")
                    q_refresh = gr.Button("🔄 Refresh List")
                    q_type = gr.Dropdown(
                        choices=["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"],
                        value="Q8_0",
                        label="Target Quantization"
                    )
                    keep_ram_toggle = gr.Checkbox(label="🚀 Keep in RAMDisk (Fast Upload)", value=True)
                    run_q_btn = gr.Button("🏗️ BEGIN CONVERSION", variant="primary")
                    gr.Markdown("---")
                    move_to_ssd_btn = gr.Button("🚚 Move RD Models to SSD", variant="secondary")
                    rd_clear_btn = gr.Button("🧹 Clear RAMDisk", variant="stop")

                with gr.Column(scale=2):
                    q_output = gr.Textbox(label="Quantizer Output", lines=20, elem_id="console_logs")

        # TAB 3: FP8 / NVFP4 QUANTIZER
        with gr.Tab("💎 FP Quants (FP8/NV4)"):
            with gr.Row():
                with gr.Column(scale=1):
                    fp_model_selector = gr.Dropdown(choices=get_model_list(), label="Source Model")
                    fp_refresh = gr.Button("🔄 Refresh List")
                    fp_format = gr.Radio(choices=["fp8", "nvfp4", "mxfp8"], value="fp8", label="Format")
                    wan_preset = gr.Checkbox(label="🛠️ Use WAN Video Preset", value=True)
                    keep_ram_toggle_fp = gr.Checkbox(label="🚀 Keep in RAMDisk (Fast Upload)", value=True)
                    run_fp_btn = gr.Button("🏗️ CONVERT TO FP", variant="primary")

                    gr.Markdown("---")
                    # Added these for convenience on this tab
                    move_to_ssd_fp = gr.Button("🚚 Move RD Models to SSD", variant="secondary")
                    rd_clear_fp = gr.Button("🧹 Clear RAMDisk", variant="stop")

                with gr.Column(scale=2):
                    fp_output = gr.Textbox(label="FP Quant Logs", lines=20, elem_id="console_logs")

    # --- Event Wiring ---

    # 1. Merger Events
    m_refresh.click(lambda: (gr.update(choices=get_model_list()), gr.update(choices=get_recipe_list())), outputs=[model_selector, recipe_selector])
    recipe_selector.change(load_selected_recipe, inputs=[recipe_selector], outputs=[json_editor])
    save_recipe_btn.click(save_active_recipe, inputs=[recipe_selector, json_editor], outputs=[merge_output])
    run_merge_btn.click(run_merge_pipeline, inputs=[json_editor, model_selector], outputs=[merge_output])

    # 2. GGUF Quantizer Events
    q_refresh.click(lambda: gr.update(choices=get_model_list()), outputs=q_model_selector)
    run_q_btn.click(run_unified_quantization, inputs=[q_model_selector, q_type, keep_ram_toggle], outputs=[q_output])

    # 3. FP Quantizer Events
    fp_refresh.click(lambda: gr.update(choices=get_model_list()), outputs=fp_model_selector)
    run_fp_btn.click(
        run_fp_quantization,
        inputs=[fp_model_selector, fp_format, wan_preset, keep_ram_toggle_fp],
        outputs=[fp_output]
    )

    # 4. Utilities
    move_to_ssd_btn.click(move_rd_to_ssd, outputs=[q_output])
    rd_clear_btn.click(clear_ramdisk, outputs=[q_output])
    move_to_ssd_fp.click(move_rd_to_ssd, outputs=[fp_output])
    rd_clear_fp.click(clear_ramdisk, outputs=[fp_output])

# 5. Launch (Gradio 6.0 style)
if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft(), css=css)

if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft(), css=css)
