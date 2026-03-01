import torch
import os, gc, re, json, datetime
from safetensors.torch import load_file, save_file
from config import MODELS_DIR, get_ram_threshold_met
from utils import verify_model_integrity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.recipe = recipe_data
        self.paths = self.recipe.get('paths', {})
        self.summary_data = [] 
        base_path = self.paths.get('base_model', "")
        if not base_path or not os.path.exists(base_path):
            raise FileNotFoundError(f"❌ Base model not found at {base_path}")
        base_lower = base_path.lower()
        self.is_motion_base = "high_noise" in base_lower 
        self.role_label = "MOTION (14B High Noise)" if self.is_motion_base else "REFINER (14B Low Noise)"
        self.base_dict = load_file(base_path)
        self.base_keys = list(self.base_dict.keys())
        self.router_regex = re.compile(r"(\.gate$|\.router$|\.wg$|layer_norm_moe)", re.IGNORECASE)

    def soft_normalize(self):
        """
        Re-calibrates variance with GPU acceleration and Surgical Router Shielding.
        Optimized for 64GB RAM / 14B MoE architecture.
        """
        count = 0
        total_reduction = 0.0
        
        yield "  🔍 Scanning 14B weights for variance drift (GPU-Accelerated)..."

        with torch.no_grad():
            for i, key in enumerate(self.base_keys):
                # 🛡️ SURGICAL ROUTER SHIELD (Regex)
                # Skips the model's brain to maintain MoE routing logic integrity
                if self.router_regex.search(key):
                    continue

                w = self.base_dict[key]
                # Targeting linear/expert weights that can cause visual static if they drift
                if "weight" in key and w.ndim >= 2:
                    # 🚀 GPU SPEED BOOST: Move variance check to CUDA
                    # Standard deviation on a 14B layer is ~50x faster on GPU Tensor Cores
                    w_cuda = w.to("cuda", dtype=torch.float32, non_blocking=True)
                    std = torch.std(w_cuda).item()
                    
                    # 1.15 threshold is calibrated specifically for WAN 2.2 14B Expert stability
                    if std > 1.15:
                        scale = 1.10 / std
                        reduction = (1.0 - scale) * 100
                        
                        # Apply scaling and cast back to original precision (bfloat16/float16)
                        self.base_dict[key] = (w_cuda * scale).to(w.dtype).to("cpu")
                        
                        total_reduction += reduction
                        count += 1
                        
                        # UI Keep-Alive: Yields every 100 tensors to prevent Gradio timeout
                        if count % 100 == 0:
                            yield f"  ✨ NORM: Stabilizing {key}... (Current Avg: -{total_reduction/count:.2f}%)"
                    
                    # Explicitly clear VRAM reference to keep the 64GB RAM overhead clean
                    del w_cuda

        if count > 0:
            avg_red = total_reduction / count
            final_msg = f"  ✅ NORM COMPLETE: Recalibrated {count} tensors (Avg: -{avg_red:.2f}%)"
        else:
            final_msg = "  ✨ NORM: All 14B experts are within stable range."
        
        print(final_msg)
        yield final_msg

    def get_compatibility_report(self):
        forbidden = "low" if self.is_motion_base else "high"
        mismatches = []
        pipeline = self.recipe.get('pipeline', [])
        for step in pipeline:
            for feature in step.get('features', []):
                if forbidden in feature['file'].lower():
                    mismatches.append(feature['file'])
        return mismatches

    def scan_lora_density(self, lora_sd):
        mags = [v.abs().mean().item() for k, v in lora_sd.items() if "weight" in k or ".diff" in k]
        return sum(mags) / len(mags) if mags else 0

    def find_lora_keys(self, lora_sd, target_key):
        import re
        # target: "blocks.0.ffn.0.weight"
        t_clean = target_key.replace("diffusion_model.", "").replace(".weight", "")
        # 🛡️ INDEX EXTRACTION: Finds all numbers (e.g., [0, 0])
        t_indices = re.findall(r'\d+', t_clean)
        t_norm = t_clean.replace(".", "_")

        # The pairs found in your report (B is usually the 'Up' matrix in these)
        pairs = [
            (".lora_B.weight", ".lora_A.weight"),
            (".lora_B", ".lora_A"),
            (".lora_up.weight", ".lora_down.weight"),
            (".lora_up", ".lora_down")
        ]

        for k in lora_sd.keys():
            # 🛡️ STRICT NUMBER CHECK:
            # If the LoRA key has different numbers than the target, skip it!
            if t_indices != re.findall(r'\d+', k):
                continue

            for up_suf, down_suf in pairs:
                if up_suf in k:
                    # Normalize LoRA stem (remove prefixes/separators)
                    l_stem = k.replace(up_suf, "").replace(".", "_").replace("__", "_")
                    if "lora_unet_" in l_stem: l_stem = l_stem.replace("lora_unet_", "")
                    
                    # TAIL MATCH: Allows "blocks_0_ffn_0" to match "diffusion_model_blocks_0_ffn_0"
                    if t_norm.endswith(l_stem) or l_stem.endswith(t_norm):
                        down_k = k.replace(up_suf, down_suf)
                        if down_k in lora_sd:
                            return k, down_k, "matrix"
                            
        return None, None, None

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult, target_key, method, recipe_density):
        with torch.no_grad():
            # Move to CUDA for the actual math
            up = lora_sd[up_k].to("cuda", dtype=torch.float32)
            down = lora_sd[down_k].to("cuda", dtype=torch.float32)
            
            # 🛡️ AUTOMATIC RANK ALIGNMENT
            # Some LoRAs are [5120, 32], others are [32, 5120]. 
            # We ensure (Up @ Down) results in the shape the Base Model expects.
            if up.shape[1] != down.shape[0]:
                if up.shape[1] == down.shape[1]:
                    down = down.T
                elif up.shape[0] == down.shape[0]:
                    up = up.T

            # Calculate Delta
            delta = (up @ down) * (float(weight) * global_mult)

            # DUAL MODE: Addition vs Injection
            if method.upper() == "INJECTION" and recipe_density < 1.0:
                flat = delta.flatten()
                k_drop = int(flat.numel() * (1.0 - recipe_density))
                if k_drop > 0:
                    thresh = torch.topk(flat.abs(), k_drop, largest=False).values[-1]
                    mask = delta.abs() > thresh
                    delta = torch.where(mask, delta / recipe_density, torch.zeros_like(delta))
                kept_pct = recipe_density * 100
            else:
                # Standard Addition (100% weight, ignores density)
                kept_pct = 100.0

            return delta.to("cpu"), kept_pct, True

    def apply_delta(self, target_key, delta):
        """
        Applies delta to base model with 5D MoE support and Norm-Preservation.
        """
        if delta is None: return False
        base = self.base_dict[target_key]
        
        with torch.no_grad():
            cpu_delta = delta.to(torch.float32)
            base_f32 = base.to(torch.float32)
            
            # 🛡️ 5D TENSOR RECONCILIATION (For MoE Experts)
            # Reshapes 2D LoRA [H, W] to match 5D Expert [1, 1, 8, H, W]
            if base_f32.ndim == 5 and cpu_delta.ndim <= 2:
                view_shape = [1] * (base_f32.ndim - 2) + list(cpu_delta.shape)
                cpu_delta = cpu_delta.view(view_shape)

            # --- THE "VIDEO KILL" CURE: NORM PRESERVATION ---
            # Measure original activation energy (vital for Routers/Gates)
            orig_norm = torch.norm(base_f32)
            
            # Apply the update
            updated_weight = base_f32 + cpu_delta
            
            # Re-scale back to Original Norm 
            # This keeps the MoE Routers perfectly calibrated
            new_norm = torch.norm(updated_weight)
            if new_norm > 0:
                updated_weight = updated_weight * (orig_norm / new_norm)
            
            # 5-Sigma Stability Guard
            std, mean = torch.std_mean(base_f32)
            limit = mean + (std * 5)
            updated_weight = torch.clamp(updated_weight, -limit, limit)
            
            self.base_dict[target_key] = updated_weight.to(base.dtype)
        return True

    def process_pass(self, step, global_mult):
        """
        Executes a 14B Merge Pass. 
        Unshielded: Now merges into Routers and Experts alike.
        Optimized for 64GB RAM / MoE Architecture.
        """
        method_label = str(step.get('method', 'ADDITION')).upper()
        pass_name = str(step.get('pass_name', 'Pass')).upper()
        recipe_density = float(step.get('density', 1.0))
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        
        # Calculate baseline for Global Shift tracking
        pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        pass_stats, total_inj, pass_peaks = {"matrix": 0, "vector": 0, "dense": 0}, [], 0
        
        for f in features:
            from config import get_ram_threshold_met # Ensure import is available
            if get_ram_threshold_met():
                yield f"  ⚠️ SYSTEM OVERHEAD CRITICAL: Low-RAM Mode (zSwap Active) for {f['file']}..."
            
            lora_path = os.path.join(lora_dir, f['file'])
            if not os.path.exists(lora_path): 
                yield f"  ⚠️ SKIP: {f['file']} (Not Found)"
                continue
            
            yield f"  🧬 Analyzing: {f['file']}..."
            
            # Load LoRA to CPU first
            lora_sd = load_file(lora_path)
            
            for target_key in self.base_keys:
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                dk = target_key if target_key in lora_sd else None
                
                # --- UNSHIELDED MERGE LOGIC ---
                # We removed the 'if not self.router_regex.search(target_key)' blocks
                match (k_type, dk):
                    case ("matrix", _):
                        # calculate_delta handles the Addition vs Injection logic internally
                        delta, inj_val, _ = self.calculate_delta(
                            lora_sd, k1, k2, f['weight'], global_mult, 
                            target_key, method_label, recipe_density
                        )
                        
                        # apply_delta handles 5D broadcasting and Norm-Preservation
                        if self.apply_delta(target_key, delta):
                            pass_stats["matrix"] += 1
                            total_inj.append(inj_val)
                            if inj_val > 0: pass_peaks += 1
                    
                    case ("vector", _):
                        delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        if self.apply_delta(target_key, delta):
                            pass_stats["vector"] += 1
                    
                    case (_, str(dense_key)):
                        delta = (lora_sd[dense_key].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        if self.apply_delta(target_key, delta):
                            pass_stats["dense"] += 1
                            total_inj.append(100.0)

            # Cleanup current LoRA
            del lora_sd
            self._cleanup()

            # 3. KNOWLEDGE HEALTH CALCULATION
            current_val = (sum(total_inj) / len(total_inj)) if total_inj else 100.0
            health_status = "STABLE" if current_val > 5.0 else "THIN"
            if pass_peaks > 15000: health_status = "VOLATILE" 

            yield f"    └─ [{method_label}] Knowledge Kept: {current_val:.1f}% | Health: {health_status}"

        # 4. INTERNAL NORMALIZATION
        if step.get('normalize', False):
            for norm_msg in self.soft_normalize():
                yield norm_msg

        # 5. FINAL PASS SUMMARY
        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        shift = abs(post_mean - pre_mean)
        
        # Count total successful applications
        total_layers = pass_stats["matrix"] + pass_stats["vector"] + pass_stats["dense"]
        
        self.summary_data.append({
            "pass": pass_name, 
            "method": method_label, 
            "layers": total_layers, 
            "kept": current_val, 
            "peaks": pass_peaks, 
            "delta": shift
        })
        
        # This is the line that shows you if the engine is actually working
        yield f"  ✅ {pass_name} Complete. Layers Matched: {total_layers} | Global Shift: {shift:.8f}"

    def get_final_summary_string(summary_list):
        if not summary_list: return "No data."
        header = f"{'PASS NAME':<15} | {'METHOD':<10} | {'LAYERS':<8} | {'KEPT %':>8} | {'PEAKS':<6} | {'SHIFT'}"
        lines = [header, "-" * len(header)]
        
        for s in summary_list:
            # Use 'kept' to match our refactored dictionary
            lines.append(f"{s['pass']:<15} | {s['method']:<10} | {s['layers']:<8} | {s['kept']:>8.1f}% | {s['peaks']:<6} | {s['delta']:.8f}")
        
        return "\n".join(lines)

    def get_metadata_string(self, quant="None"):
        from utils import get_final_summary_string
        summary_text = get_final_summary_string(self.summary_data, self.role_label)
        header = f"Title: {self.paths.get('title', 'Dasiwa Master')}\nExport: {quant}\n"
        return f"{header}\n{summary_text}\n🚀 Made with DaSiWa"

    def run_pre_save_check(self):
            # We pass the dictionary and keys to the utility
            for msg in verify_model_integrity(self.base_dict, self.base_keys, self.router_regex):
                yield msg

    def save_master(self, path):
        import os # Ensure os is available for sync
        metadata_str = self.get_metadata_string()
        custom_metadata = {"comment": metadata_str, "dasiwa_summary": metadata_str}
        
        self._cleanup() 
        print(f"📦 Pre-Processing 14B Tensors (Low-Pressure Alignment)...")
        
        try:
            with torch.no_grad():
                for k in self.base_keys:
                    # Fix: Move to CPU FIRST, then make contiguous to avoid doubling
                    # This uses zSwap/SSD-Cache instead of spiking physical RAM
                    temp_tensor = self.base_dict[k].to("cpu", non_blocking=True)
                    self.base_dict[k] = temp_tensor.contiguous()
                    del temp_tensor
                    
                    if "layers.10" in k or "layers.30" in k:
                        gc.collect()

            print(f"💾 SSD WRITE STARTING... (Forcing Hardware Sync)")
            # FORCE OS TO CLEAR CACHE BEFORE BIG WRITE
            try: os.sync() 
            except: pass
            
            save_file(self.base_dict, path, metadata=custom_metadata)
            
            # FORCE HARDWARE SYNC AFTER WRITE
            try: os.sync() 
            except: pass
            
            return path
            
        except Exception as e:
            print(f"❌ SSD WRITE CRASHED: {str(e)}")
            raise e
        finally:
            # Immediate wipe of the 26GB dictionary
            self.base_dict = None 
            self._cleanup()

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()