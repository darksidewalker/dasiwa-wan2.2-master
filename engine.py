import torch
import os, gc, re, json, datetime
from safetensors.torch import load_file, save_file
from config import MODELS_DIR, get_ram_threshold_met

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

    def soft_normalize(self):
        """Re-calibrates tensor variance with real-time feedback."""
        count = 0
        total_reduction = 0.0
        
        with torch.no_grad():
            for key in self.base_keys:
                w = self.base_dict[key]
                if "weight" in key and w.ndim >= 2:
                    w_float = w.to(torch.float32)
                    std = torch.std(w_float)
                    
                    # If variance drifts > 5% from standard
                    if std > 1.05:
                        scale = 1.0 / std
                        reduction = (1.0 - scale.item()) * 100
                        self.base_dict[key] = (w_float * scale).to(w.dtype)
                        total_reduction += reduction
                        count += 1
        
        if count > 0:
            avg_red = total_reduction / count
            msg = f"  ✨ NORM: Recalibrated {count} tensors (Avg Reduction: -{avg_red:.2f}%)"
        else:
            msg = "  ✨ NORM: Tensors already stable. No adjustment needed."
            
        print(msg) # CLI Feedback
        return msg # GUI Feedback

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

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult, target_key, lora_path, smart_logic=True):
        # 1. TRIGGER LOW-RAM SAFEGUARD
        is_low_ram = get_ram_threshold_met() 
        
        with torch.no_grad():
            calc_device = DEVICE if not is_low_ram else "cpu"
            
            up_w = lora_sd[up_k].to(calc_device, dtype=torch.float32)
            down_w = lora_sd[down_k].to(calc_device, dtype=torch.float32)

            # Shape Correction
            if up_w.shape[1] != down_w.shape[0]:
                if up_w.shape[1] == down_w.shape[1]: down_w = down_w.t()
                elif up_w.shape[0] == down_w.shape[0]: up_w = up_w.t()
            
            rank = up_w.shape[1]
            alpha_key = up_k.split('.lora')[0] + ".alpha"
            alpha_val = lora_sd.get(alpha_key, torch.tensor(float(rank)))
            scale = alpha_val.item() / rank if rank != 0 else 1.0
            
            delta = torch.matmul(up_w, down_w) * scale
            
            if is_low_ram:
                up_w, down_w = up_w.to("cpu"), down_w.to("cpu")

            passed_pct, limit_hit = 100.0, False

            if smart_logic:
                # --- DYNAMIC MASS CALIBRATION (NEW) ---
                # Calculate size in MB to determine Rank-based aggression
                lora_size_mb = os.path.getsize(lora_path) / (1024 * 1024)
                
                if lora_size_mb < 350:      # RANK 32: Surgical / Small
                    g_base, l_base = 0.005, 0.20
                elif lora_size_mb < 750:    # RANK 64: Balanced
                    g_base, l_base = 0.015, 0.12
                elif lora_size_mb < 1100:   # RANK 128: Heavy
                    g_base, l_base = 0.040, 0.09
                else:                       # RANK 256+: Structural
                    g_base, l_base = 0.120, 0.06

                # --- DISPATCH LOGIC ---
                base_w = self.base_dict[target_key].to(calc_device, dtype=torch.float32)
                base_var = torch.var(base_w).item()
                is_new_info = torch.var(delta).item() > (base_var * 1.5)
                
                match is_new_info:
                    case True: # NSFW / MOTION / IMPACT
                        gate_mult, limit = g_base, l_base
                    case False: # REFINER / STYLE
                        gate_mult, limit = g_base * 8, l_base * 0.5
                
                # Apply Gating
                gate = delta.abs().mean() * gate_mult
                mask = delta.abs() > gate
                delta = torch.where(mask, delta, torch.zeros_like(delta))
                passed_pct = (mask.sum().item() / mask.numel()) * 100
                
                # Apply Safety Limit (Prevent "Deep-Frying")
                if delta.abs().max() > limit:
                    delta *= (limit / delta.abs().max())
                    limit_hit = True
                
                if calc_device != "cpu": del base_w

            # Final result return
            return (delta * float(weight) * global_mult).to("cpu"), passed_pct, limit_hit

    def process_pass(self, step, global_mult):
        method_label = str(step.get('method', 'ADDITION')).upper()
        pass_name = str(step.get('pass_name', 'Pass')).upper()
        
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        use_smart = (method_label == "INJECTION")
        
        pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        pass_stats, total_inj, pass_peaks = {"matrix": 0, "vector": 0, "dense": 0, "skipped": 0}, [], 0
        
        for f in features:
            if get_ram_threshold_met():
                yield f"  ⚠️ SYSTEM OVERHEAD CRITICAL: Entering Low-RAM Mode for {f['file']}..." 
            
            lora_path = os.path.join(lora_dir, f['file'])
            if not os.path.exists(lora_path): 
                yield f"  ⚠️ SKIP: {f['file']} (Not Found)"
                continue
            
            yield f"  🧬 Analyzing: {f['file']}..."
            lora_sd = load_file(lora_path)
            
            for target_key in self.base_keys:
                # find_lora_keys logic remains the same
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                dk = target_key if target_key in lora_sd else None
                
                match (k_type, dk):
                    case ("matrix", _):
                        # UPDATED: Now passing lora_path for the Smart Gate logic
                        delta, inj_val, hit = self.calculate_delta(
                            lora_sd, k1, k2, f['weight'], global_mult, target_key, lora_path, use_smart
                        )
                        if hit: pass_peaks += 1
                        if self.apply_delta(target_key, delta):
                            pass_stats["matrix"] += 1
                            total_inj.append(inj_val)
                    
                    case ("vector", _):
                        delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        if self.apply_delta(target_key, delta):
                            pass_stats["vector"] += 1
                    
                    case (_, str(dense_key)):
                        delta = (lora_sd[dense_key].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        if self.apply_delta(target_key, delta):
                            pass_stats["dense"] += 1
                            total_inj.append(100.0)

            del lora_sd
            self._cleanup()

            current_val = (sum(total_inj) / len(total_inj)) if total_inj else 100.0
            
            # Health Logic: For 14B, we want high Knowledge but low Peak Shift
            # A 'Volatile' tag here now correctly flags high variance in small LoRAs
            health_status = "OK" if current_val > 60 else "THIN"
            if pass_peaks > 5000: health_status = "VOLATILE" 

            yield f"    └─ [{method_label}] Knowledge Kept: {current_val:.1f}% | Health: {health_status}"

        if step.get('normalize', False):
            norm_msg = self.soft_normalize()
            yield norm_msg

        # Final pass summary calculation
        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        shift = abs(post_mean - pre_mean)
        
        lora_size_mb = os.path.getsize(lora_path) / (1024 * 1024)
        if lora_size_mb < 350: tier = "R32"
        elif lora_size_mb < 750: tier = "R64"
        elif lora_size_mb < 1100: tier = "R128"
        else: tier = "R256+"

        self.summary_data.append({
            "pass": pass_name, 
            "method": method_label, 
            "tier": tier, 
            "layers": sum(v for k,v in pass_stats.items() if k != "skipped"), 
            "kept": current_val, # MUST BE 'kept'
            "peaks": pass_peaks, 
            "delta": shift
        })
        
        yield f"  ✅ {pass_name} Complete. Global Shift: {shift:.8f}"

    def apply_delta(self, target_key, delta):
        base = self.base_dict[target_key]
        if base.ndim > 2 and delta.ndim <= 2: return False
        if delta.numel() == base.numel(): delta = delta.reshape(base.shape)
        elif delta.ndim == 2 and delta.t().shape == base.shape: delta = delta.t()
        else: return False
        with torch.no_grad():
            cpu_delta = delta.to("cpu", dtype=torch.float32, non_blocking=True).pin_memory()
            self.base_dict[target_key] = (base.to(torch.float32) + cpu_delta).to(base.dtype)
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

    def save_master(self, path):
        """Memory-optimized save to prevent system lock during 14B writes."""
        metadata_str = self.get_metadata_string()
        custom_metadata = {
            "comment": metadata_str, 
            "dasiwa_summary": metadata_str
        }
        
        # 1. PRE-SAVE PURGE: Clear every possible byte of VRAM/RAM
        self._cleanup() 
        
        try:
            # 2. CONTIGUOUS CHECK: Process one tensor at a time to avoid mass spikes
            with torch.no_grad():
                for k in self.base_keys:
                    if not self.base_dict[k].is_contiguous():
                        # Use .to() to move and clean in one step if needed
                        self.base_dict[k] = self.base_dict[k].contiguous()

            # 3. DIRECT SAVE: Write to SSD
            print(f"💾 SSD WRITE STARTING: {os.path.basename(path)}")
            save_file(self.base_dict, path, metadata=custom_metadata)
            return path
            
        except Exception as e:
            print(f"❌ SSD WRITE CRASHED: {str(e)}")
            raise e
        finally:
            # 4. IMMEDIATE RELEASE: Destroy the 26GB dictionary
            self.base_dict = None 
            self._cleanup()

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()