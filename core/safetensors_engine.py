# core/safetensors_engine.py
import os, subprocess, sys
from core.metadata_manager import inject_metadata, get_current_meta
from config import CONVERT_PY 
from utils.file_ops import save_log

def run_safe_conversion(MODELS_DIR, source_path, formats, model_name, options, log_acc):
    FLAG_MAP = {
        "FP8": ["--comfy_quant"],
        "INT8 Block-wise": ["--int8", "--scaling_mode", "block", "--comfy_quant"],
        "NVFP4": ["--nvfp4", "--comfy_quant"],
        "Auto-Quality (Heur)": ["--heur"]
    }

    ULTRA_PARAMS = [
        "--save-quant-metadata", "--wan", "--optimizer", "adamw",
        "--num_iter", "9000", "--calib_samples", "45000", 
        "--lr_schedule", "plateau", "--lr_patience", "2", "--lr_factor", "0.96", 
        "--lr_min", "9e-9", "--lr_cooldown", "0", "--lr_threshold", "1e-11", 
        "--lr", "9.916700000002915715e-3", "--top_p", "0.05", 
        "--min_k", "64", "--max_k", "256", "--early-stop-stall", "20000", 
        "--early-stop-lr", "1e-8", "--early-stop-loss", "9e-8", "--lr-shape-influence", "3.5"
    ]

    for fmt in formats:
        suffix = fmt.lower().replace(" ", "_")
        final_path = source_path.replace(".safetensors", f"_{suffix}.safetensors")
        
        cmd = ["convert_to_quant", "-i", source_path, "-o", final_path]
        
        if fmt in FLAG_MAP:
            cmd.extend(FLAG_MAP[fmt])

        if options == "Ultra-Quality (Optimizer)":
            log_acc += f"💎 MODE: Ultra-Optimizer (9000 Iters) + {fmt}\n"
            cmd.extend(ULTRA_PARAMS)
        elif options == "Auto-Quality (Heur)":
            if "--heur" not in cmd:
                cmd.append("--heur")
        else:
            cmd.append("--simple")

        if "wan" in source_path.lower() and "--wan" not in cmd: 
            cmd.append("--wan")

        log_acc += f"\n▶️ Executing: {' '.join(cmd)}\n"
        yield log_acc, f"Processing {fmt}..."

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, universal_newlines=True
        )

        current_line = ""
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None: break
            
            if char == '\n' or char == '\r':
                clean_line = current_line.strip().lower()
                if clean_line and not clean_line.startswith("optimizing"):
                    log_acc += current_line + "\n"
                    yield log_acc, f"Quantizing {fmt}..."
                current_line = ""
            else:
                current_line += char

        process.wait()

        if process.returncode == 0 and os.path.exists(final_path):
            meta = get_current_meta(model_name, fmt)
            inject_metadata(final_path, meta)
            log_acc += f"📝 Meta Injected: {os.path.basename(final_path)}\n"
        else:
            log_acc += f"❌ Failed or file missing for {fmt}\n"

    save_log(model_name, log_acc)       
    yield log_acc, "Finished Safetensors"