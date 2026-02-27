import torch
import os, gc, re, json, datetime
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.recipe = recipe_data
        self.paths = self.recipe.get('paths', {})
        self.summary_data = [] 
        
        base_path = self.paths.get('base_model', "")
        if not base_path or not os.path.exists(base_path):
            raise FileNotFoundError(f"❌ Base model not found at {base_path}")
            
        print(f"📥 Loading Base Model: {os.path.basename(base_path)}")
        self.base_dict = load_file(base_path)
        self.base_keys = list(self.base_dict.keys())
        
        self.is_motion_base = "high" in base_path.lower()
        self.role_label = "MOTION (14B High Noise)" if self.is_motion_base else "REFINER (14B Low Noise)"
        print(f"🛰️ Engine: {self.role_label} Active.")

    def scan_lora_density(self, lora_sd):
        mags = [v.abs().mean().item() for k, v in lora_sd.items() if "weight" in k or ".diff" in k]
        avg_mag = sum(mags) / len(mags) if mags else 0
        return avg_mag

    def find_lora_keys(self, lora_sd, target_key):
        # Standardize the target path (e.g., 'blocks.0.cross_attn.k')
        # This acts as our "anchor" that must exist in the LoRA key
        clean_target = target_key.replace("diffusion_model.", "").replace(".weight", "")
        
        # Map of Up/Down pairs for your 3 specific examples
        pairs = [
            (".lora_B", ".lora_A"),       # Example 1 & 2
            (".lora_up", ".lora_down"),   # Example 3
        ]

        for k in lora_sd.keys():
            # Check if the anchor path is present and strictly bounded by dots
            # This prevents 'blocks.1' from matching 'blocks.11'
            if f"{clean_target}." in k or k.startswith(f"{clean_target}."):
                for up_suf, down_suf in pairs:
                    if up_suf in k:
                        up_k = k
                        down_k = k.replace(up_suf, down_suf)
                        if down_k in lora_sd:
                            return up_k, down_k, "matrix"
                
                # Fallback for vectors/alphas
                if any(x in k for x in [".diff", ".alpha"]):
                    return k, None, "vector"
                    
        return None, None, None

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult, target_key, smart_logic=True):
        with torch.no_grad():
            up_w = lora_sd[up_k].to(DEVICE, dtype=torch.float32)
            down_w = lora_sd[down_k].to(DEVICE, dtype=torch.float32)
            
            # 1. Automatic Dimension Correction
            if up_w.shape[1] != down_w.shape[0]:
                if up_w.shape[1] == down_w.shape[1]: down_w = down_w.t()
                elif up_w.shape[0] == down_w.shape[0]: up_w = up_w.t()
            
            # 2. Rank and Alpha Scaling (Crucial for Style Preservation)
            rank = up_w.shape[1]
            # Look for alpha using the base name of the Up key
            alpha_key = up_k.split('.lora')[0] + ".alpha"
            alpha_val = lora_sd.get(alpha_key, torch.tensor(float(rank)))
            scale = alpha_val.item() / rank if rank != 0 else 1.0
            
            delta = torch.matmul(up_w, down_w) * scale

            # 3. Apply your Specific Merging Methods
            passed_pct, limit_hit = 100.0, False
            
            if smart_logic:  # This is your "Injection" method
                base_w = self.base_dict[target_key].to(DEVICE, dtype=torch.float32)
                # Thresholding based on base weight variance
                base_var = torch.var(base_w).item()
                is_new_info = torch.var(delta).item() > (base_var * 1.5)
                
                # Sensitivity gate
                sensitivity = 0.4 if is_new_info else 1.8 
                gate = delta.abs().mean() * sensitivity
                mask = delta.abs() > gate
                delta = torch.where(mask, delta, torch.zeros_like(delta))
                passed_pct = (mask.sum().item() / mask.numel()) * 100

                # Peak Limiter to prevent "Deep Fried" pixels
                max_val = delta.abs().max()
                limit = 0.08 if is_new_info else 0.05
                if max_val > limit:
                    delta *= (limit / max_val)
                    limit_hit = True

            final_mult = float(weight) * global_mult
            return (delta * final_mult).to("cpu"), passed_pct, limit_hit

    def process_pass(self, step, global_mult):
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        req_density = float(step.get('density', 0.9))
        method = str(step.get('method', 'addition')).lower().strip()
        use_smart = (method == "injection")
        
        pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        pass_stats = {"matrix": 0, "vector": 0, "dense": 0}
        total_inj, pass_peaks = [], 0
        
        for f in features:
            lora_path = os.path.join(lora_dir, f['file'])
            if not os.path.exists(lora_path): continue

            lora_sd = load_file(lora_path)
            avg_mag = self.scan_lora_density(lora_sd)
            asked_weight = float(f.get('weight', 1.0))
            
            print(f"🔍 SCAN [{f['file']}]")
            print(f"   ├─ Raw Signal: {avg_mag:.4f}")
            print(f"   ├─ Multiplier: x{asked_weight:.2f} (Target: {avg_mag * asked_weight:.4f})")
            print(f"   └─ Mode: {'INJECTION' if use_smart else 'ADDITION'}")

            for target_key in self.base_keys:
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                dk = target_key if target_key in lora_sd else (f"diffusion_model.{target_key}" if f"diffusion_model.{target_key}" in lora_sd else None)
                
                with torch.no_grad():
                    delta, inj_val, hit = None, 100.0, False
                    if k_type == "matrix":
                        delta, inj_val, hit = self.calculate_delta(lora_sd, k1, k2, f['weight'], global_mult, target_key, smart_logic=use_smart)
                    elif k_type == "vector":
                        delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                    elif dk:
                        k_type = "dense"
                        raw_delta = lora_sd[dk].to(torch.float32) * float(f['weight']) * global_mult
                        delta = raw_delta.cpu() # Simplified Dense for brevity
                    
                    if hit: pass_peaks += 1
                    if delta is not None and self.apply_delta(target_key, delta):
                        pass_stats[k_type] += 1
                        if k_type in ["matrix", "dense"]: total_inj.append(inj_val)
            
            if pass_peaks > 0 and use_smart:
                print(f"   ⚠️ PEAK ALERT: Limiter squashed {pass_peaks} layers. (Too hot!)")
            self._cleanup()

        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        avg_injection = sum(total_inj) / len(total_inj) if total_inj else 100.0
        
        self.summary_data.append({
            "pass": step.get('pass_name', 'Pass'),
            "method": method.upper(),
            "layers": sum(pass_stats.values()),
            "inj": avg_injection,
            "peaks": pass_peaks,
            "delta": abs(post_mean - pre_mean)
        })

    def apply_delta(self, target_key, delta):
        base = self.base_dict[target_key]
        if delta.numel() == base.numel(): delta = delta.reshape(base.shape)
        elif delta.t().shape == base.shape: delta = delta.t()
        else: return False
        self.base_dict[target_key] = (base.to(torch.float32) + delta.to(base.device)).to(base.dtype)
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
        status = "STABLE" if total_delta < 0.015 else ("SATURATED" if total_delta < 0.030 else "VOLATILE")
        lines.append(f"{'TOTAL MODEL SHIFT':<52} | {total_delta:.8f}")
        lines.append(f"{'STABILITY CHECK':<52} | {status}")
        lines.append("="*85 + "\n")
        return "\n".join(lines)

    def get_metadata_string(self, quant_label="None (Raw Weights)"):
        """Generates the branded summary with dynamic precision mapping."""
        sample_key = self.base_keys[0]
        dtype = self.base_dict[sample_key].dtype
        
        # Internal math precision
        internal_prec = "BF16" if dtype == torch.bfloat16 else ("FP16" if dtype == torch.float16 else "FP32")
        
        # Clean up the export label for the header
        export_prec = quant_label.replace("GGUF_", "") if "GGUF_" in quant_label else quant_label
        
        header = (
            f"Title: {self.paths.get('title', 'Dasiwa Master')}\n"
            f"Type: {self.paths.get('type', 'Wan 2.2 14B')}\n"
            f"Resolution: {self.paths.get('resolution', '960x960')}\n"
            f"Internal Math: {internal_prec}\n"
            f"Export Precision: {export_prec}\n"
        )
        
        branding = "\n🚀 Made with DaSiWa WAN 2.2 Master by Darksidewalker"
        return f"{header}\n{self.get_final_summary_string()}{branding}"

    def save_master(self, path):
        full_log = self.get_metadata_string()
        custom_metadata = {"comment": full_log, "dasiwa_summary": full_log}
        master_sd = {k: v.contiguous().cpu() for k, v in self.base_dict.items()}
        save_file(master_sd, path, metadata=custom_metadata)
        self._cleanup()
        return path

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()