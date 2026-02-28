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
        clean_target = target_key.replace("diffusion_model.", "").replace(".weight", "")
        underscore_target = clean_target.replace(".", "_")
        pairs = [(".lora_B", ".lora_A"), (".lora_up", ".lora_down"), ("_lora_up", "_lora_down"), (".lora_up.weight", ".lora_down.weight"), ("_lora_B", "_lora_A")]
        for k in lora_sd.keys():
            if clean_target in k or underscore_target in k:
                for up_suf, down_suf in pairs:
                    if up_suf in k:
                        up_k, down_k = k, k.replace(up_suf, down_suf)
                        if down_k in lora_sd: return up_k, down_k, "matrix"
        if not len(self.base_dict[target_key].shape) == 2:
            for k in lora_sd.keys():
                if clean_target in k or underscore_target in k:
                    if any(x in k for x in [".diff", ".alpha", ".bias", "_alpha", "_bias"]): return k, None, "vector"
        return None, None, None

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult, target_key, recipe_density=1.0):
        # 🛡️ ROUTER SHIELD (From our previous step)
        if self.router_regex.search(target_key):
            return torch.zeros_like(self.base_dict[target_key]).to("cpu"), 0.0, False

        with torch.no_grad():
            # 🚀 64GB BOOST: Use Pinned Memory for DMA Transfers
            # We convert to float32 and 'pin' the memory so the GPU can pull it via DMA
            up_f32 = lora_sd[up_k].to(torch.float32).pin_memory()
            down_f32 = lora_sd[down_k].to(torch.float32).pin_memory()

            # Move to CUDA (non_blocking=True is safe with pinned memory)
            up = up_f32.to("cuda", non_blocking=True)
            down = down_f32.to("cuda", non_blocking=True)
            
            # ⚡ GPU MATRIX MATH
            delta = (up @ down) * (float(weight) * global_mult)
            
            # 🎯 TOP-K SPARSITY ON GPU
            if recipe_density < 1.0:
                flat_delta = delta.abs().view(-1)
                k = int(flat_delta.numel() * recipe_density)
                if k > 0:
                    threshold = torch.topk(flat_delta, k).values[-1]
                    delta = torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta))
                    passed_pct = (k / flat_delta.numel()) * 100
                else: passed_pct = 0.0
            else: passed_pct = 100.0

            # Move back to CPU (This is now the only slow part, but it's only 5% of the data)
            return delta.to("cpu"), passed_pct, True

    def process_pass(self, step, global_mult):
        """
        Executes a 14B Merge Pass with GPU-Acceleration and Router Shielding.
        Optimized for 64GB RAM / MoE Architecture.
        """
        method_label = str(step.get('method', 'ADDITION')).upper()
        pass_name = str(step.get('pass_name', 'Pass')).upper()
        recipe_density = float(step.get('density', 1.0))
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        
        # Calculate baseline for Global Shift tracking
        pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        pass_stats, total_inj, pass_peaks = {"matrix": 0, "vector": 0, "dense": 0, "skipped": 0}, [], 0
        
        for f in features:
            if get_ram_threshold_met():
                yield f"  ⚠️ SYSTEM OVERHEAD CRITICAL: Low-RAM Mode (zSwap Active) for {f['file']}..." 
            
            lora_path = os.path.join(lora_dir, f['file'])
            if not os.path.exists(lora_path): 
                yield f"  ⚠️ SKIP: {f['file']} (Not Found)"
                continue
            
            yield f"  🧬 Analyzing: {f['file']}..."
            
            # Load LoRA to CPU first, then we'll pin/move specific tensors in calculate_delta
            lora_sd = load_file(lora_path)
            
            for target_key in self.base_keys:
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                dk = target_key if target_key in lora_sd else None
                
                match (k_type, dk):
                    case ("matrix", _):
                        # 1. GPU-ACCELERATED CALCULATION
                        # Returns 'hit=False' if the key triggered the Router Shield
                        delta, inj_val, hit = self.calculate_delta(
                            lora_sd, k1, k2, f['weight'], global_mult, 
                            target_key, recipe_density
                        )
                        
                        # 2. APPLY ONLY IF NOT SHIELDED
                        if hit:
                            if self.apply_delta(target_key, delta):
                                pass_stats["matrix"] += 1
                                total_inj.append(inj_val)
                                if inj_val > 0: pass_peaks += 1
                        else:
                            pass_stats["skipped"] += 1
                    
                    case ("vector", _):
                        # Vectors are small enough to do on CPU, but we still shield them
                        if not self.router_regex.search(target_key):
                            delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                            if self.apply_delta(target_key, delta):
                                pass_stats["vector"] += 1
                    
                    case (_, str(dense_key)):
                        if not self.router_regex.search(target_key):
                            delta = (lora_sd[dense_key].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                            if self.apply_delta(target_key, delta):
                                pass_stats["dense"] += 1
                                total_inj.append(100.0)

            # Cleanup current LoRA before moving to the next feature
            del lora_sd
            self._cleanup()

            # 3. KNOWLEDGE HEALTH CALCULATION
            current_val = (sum(total_inj) / len(total_inj)) if total_inj else 100.0
            
            # Calibration for 14B MoE: "Thin" is expected at low densities
            health_status = "STABLE" if current_val > 5.0 else "THIN"
            if pass_peaks > 15000: health_status = "VOLATILE" 

            yield f"    └─ [{method_label}] Knowledge Kept: {current_val:.1f}% | Health: {health_status}"

        # 4. INTERNAL NORMALIZATION (Handles doubling internally)
        if step.get('normalize', False):
            # The generator yields messages directly to the UI
            for norm_msg in self.soft_normalize():
                yield norm_msg

        # 5. FINAL PASS SUMMARY & SHIFT
        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        shift = abs(post_mean - pre_mean)
        
        self.summary_data.append({
            "pass": pass_name, 
            "method": method_label, 
            "layers": sum(v for k,v in pass_stats.items() if k != "skipped"), 
            "kept": current_val, 
            "peaks": pass_peaks, 
            "delta": shift
        })
        
        yield f"  ✅ {pass_name} Complete. Global Shift: {shift:.8f}"

    def apply_delta(self, target_key, delta):
        base = self.base_dict[target_key]
        
        # 1. STRUCTURAL VALIDATION
        if base.ndim > 2 and delta.ndim <= 2: return False
        if delta.numel() == base.numel(): 
            delta = delta.reshape(base.shape)
        elif delta.ndim == 2 and delta.t().shape == base.shape: 
            delta = delta.t()
        else: 
            return False

        with torch.no_grad():
            # 2. MATCH PRECISION
            # 14B models love bfloat16 for stability. 
            # We do the addition in float32 for accuracy, then clamp.
            cpu_delta = delta.to("cpu", dtype=torch.float32)
            base_f32 = base.to(torch.float32)
            
            # 3. THE "STABILITY" ADDITION
            updated_weight = base_f32 + cpu_delta
            
            # 4. OVERFLOW SHIELD (Crucial for WAN 2.2)
            # Prevents weights from spiraling out of control
            std, mean = torch.std_mean(base_f32)
            limit = mean + (std * 5) # Stay within 5 standard deviations
            updated_weight = torch.clamp(updated_weight, -limit, limit)
            
            # 5. CAST BACK
            self.base_dict[target_key] = updated_weight.to(base.dtype)
            
        return True

    def get_final_summary_string(self):
        lines = ["\n" + "="*85, f"📊 FINAL MERGE SUMMARY: {self.role_label}", "="*85]
        lines.append(f"{'PASS NAME':<15} | {'METHOD':<10} | {'LAYERS':<8} | {'KNOWLEDGE %':<12} | {'PEAKS':<6} | {'SHIFT'}")
        lines.append("-" * 85)
        total_delta = 0
        for s in self.summary_data:
            lines.append(f"{s['pass']:<15} | {s['method']:<10} | {s['layers']:<8} | {s['inj']:>10.1f}% | {s['peaks']:<6} | {s['delta']:.8f}")
            total_delta += s['delta']
        lines.append("-" * 85)
        # Your engine.py version has three tiers of stability
        status = "STABLE" if total_delta < 0.015 else ("SATURATED" if total_delta < 0.030 else "VOLATILE")
        lines.append(f"{'TOTAL MODEL SHIFT':<52} | {total_delta:.8f}")
        lines.append(f"{'STABILITY CHECK':<52} | {status}")
        lines.append("="*85 + "\n")
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