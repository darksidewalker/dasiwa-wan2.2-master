import torch
import os, gc, re, json
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
        """Calculates weight magnitude to suggest optimal density/heat."""
        mags = [v.abs().mean().item() for k, v in lora_sd.items() if "weight" in k or ".diff" in k]
        avg_mag = sum(mags) / len(mags) if mags else 0
        if avg_mag > 0.08: sug = 0.60  
        elif avg_mag > 0.04: sug = 0.75 
        else: sug = 0.95               
        return avg_mag, sug

    def find_lora_keys(self, lora_sd, target_key):
        """Universal Mapping: Matches Base keys to LoRA keys (Dots vs Underscores)."""
        # Normalize the base key for flexible matching
        base_slug = target_key.replace("diffusion_model.", "").replace(".weight", "").replace(".", "").replace("_", "")
        
        for k in lora_sd.keys():
            lora_slug = k.replace("diffusion_model.", "").replace(".weight", "").replace(".", "").replace("_", "")
            
            if base_slug in lora_slug:
                # Matrix LoRA Check
                if "loraup" in lora_slug or "loraB" in lora_slug:
                    up = k
                    down = up.replace("lora_up", "lora_down").replace("lora_B", "lora_A")
                    if down in lora_sd: return up, down, "matrix"
                
                # Vector/Bias Check
                if "diff" in lora_slug or "alpha" in lora_slug:
                    return k, None, "vector"
                    
        return None, None, None

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult, target_key, smart_logic=True):
        """Calculates the change vector with Adaptive Injection and Noise Gating."""
        with torch.no_grad():
            up_w = lora_sd[up_k].to(DEVICE, dtype=torch.float32)
            down_w = lora_sd[down_k].to(DEVICE, dtype=torch.float32)
            
            # Matmul Alignment
            if up_w.shape[1] == down_w.shape[1]: down_w = down_w.t()
            elif up_w.shape[0] == down_w.shape[0]: up_w = up_w.t()
            
            try: 
                delta = torch.matmul(up_w, down_w)
            except: 
                return None, 0.0

            passed_pct = 100.0 

            if smart_logic:
                # --- ADAPTIVE KNOWLEDGE INJECTION ---
                base_w = self.base_dict[target_key].to(DEVICE, dtype=torch.float32)
                base_var = torch.var(base_w).item()
                lora_var = torch.var(delta).item()
                
                # Logic: If LoRA variance is much higher than base, it's a "New Concept"
                is_new = lora_var > (base_var * 1.5)
                sensitivity = 0.4 if is_new else 1.8 # Lower gate for new concepts
                
                gate_threshold = delta.abs().mean() * sensitivity
                mask = delta.abs() > gate_threshold
                delta = torch.where(mask, delta, torch.zeros_like(delta))
                passed_pct = (mask.sum().item() / mask.numel()) * 100

                # --- DYNAMIC LIMITER ---
                max_val = delta.abs().max()
                limit = 0.08 if is_new else 0.05
                if max_val > limit:
                    delta *= (limit / max_val)

            # --- APPLY ALPHA & RANK SCALING ---
            rank = up_w.shape[1]
            a_key = up_k.replace("lora_B", "alpha").replace("lora_up", "alpha")
            a_val = lora_sd.get(a_key, torch.tensor(float(rank)))
            a = a_val.float().flatten()[0].item() if a_val.numel() > 0 else float(rank)
            
            final_mult = (a / rank if rank != 0 else 1.0) * float(weight) * global_mult
            return (delta * final_mult).to("cpu"), passed_pct

    def process_pass(self, step, global_mult):
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        req_density = float(step.get('density', 0.9))
        method = str(step.get('method', 'addition')).lower().strip()
        use_smart = (method == "injection")
        
        # Pre-pass health check
        pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        pass_stats = {"matrix": 0, "vector": 0, "dense": 0}
        total_inj = []
        
        for f in features:
            lora_path = os.path.join(lora_dir, f['file'])
            if not os.path.exists(lora_path):
                print(f"⚠️ Missing: {f['file']}")
                continue

            lora_sd = load_file(lora_path)
            avg_mag, sug_density = self.scan_lora_density(lora_sd)
            print(f"🔍 SCAN [{f['file']}]: Mag: {avg_mag:.4f} | Mode: {'SMART' if use_smart else 'RAW'}")

            for target_key in self.base_keys:
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                
                # Check for Dense weights (Full models or extracted LoRAs)
                dk = target_key if target_key in lora_sd else (f"diffusion_model.{target_key}" if f"diffusion_model.{target_key}" in lora_sd else None)
                
                with torch.no_grad():
                    delta, inj_val = None, 100.0
                    
                    # PATH A: Standard LoRA Matrices (A/B)
                    if k_type == "matrix":
                        # Returns (delta, passed_pct) from your updated calculate_delta
                        delta, inj_val = self.calculate_delta(lora_sd, k1, k2, f['weight'], global_mult, target_key, smart_logic=use_smart)
                    
                    # PATH B: Vectors (Bias, Alphas, Scales)
                    elif k_type == "vector":
                        delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                    
                    # PATH C: Dense Tensors (DARE-TIES Logic)
                    elif dk:
                        k_type = "dense"
                        raw_delta = lora_sd[dk].to(torch.float32) * float(f['weight']) * global_mult
                        
                        if use_smart:
                            # Apply TIES/DARE sparsity
                            mask = torch.bernoulli(torch.full_like(raw_delta, req_density))
                            rescaled = (raw_delta * mask / req_density)
                            k = int(rescaled.numel() * (req_density * 0.8))
                            
                            if 0 < k < rescaled.numel():
                                threshold = torch.topk(rescaled.abs().flatten(), k).values[-1]
                                delta = torch.where(rescaled.abs() >= threshold, rescaled, torch.zeros_like(rescaled)).cpu()
                                inj_val = req_density * 100.0 # Injection % for dense is the density
                            else: 
                                delta = rescaled.cpu()
                        else:
                            delta = raw_delta.cpu()

                    # Commit the change to the base model
                    if delta is not None:
                        if self.apply_delta(target_key, delta):
                            pass_stats[k_type] += 1
                            # Only track matrix/dense injection strengths
                            if k_type in ["matrix", "dense"]:
                                total_inj.append(inj_val)
                                
            self._cleanup() # Free VRAM after each LoRA file

        # Post-pass health check
        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        
        # Calculate final injection average for this pass
        avg_injection = sum(total_inj) / len(total_inj) if total_inj else 100.0
        
        self.summary_data.append({
            "pass": step.get('pass_name', 'Unnamed Pass'),
            "method": method.upper(),
            "layers": sum(pass_stats.values()),
            "inj": avg_injection,
            "delta": abs(post_mean - pre_mean)
        })
        
        print(f"✅ Pass '{step.get('pass_name')}' Finished | Inj Strength: {avg_injection:.1f}%")

        # Finalize Summary Data
        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        self.summary_data.append({
            "pass": step.get('pass_name'),
            "method": method.upper(),
            "layers": sum(pass_stats.values()),
            "inj": sum(total_inj)/len(total_inj) if total_inj else 100.0,
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
        """Returns the summary table as a formatted string for the UI logs."""
        lines = ["\n" + "="*75, f"📊 FINAL MERGE SUMMARY: {self.role_label}", "="*75]
        lines.append(f"{'PASS NAME':<15} | {'METHOD':<10} | {'LAYERS':<8} | {'KNOWLEDGE %':<12} | {'SHIFT'}")
        lines.append("-" * 75)
        
        total_delta = 0
        for s in self.summary_data:
            lines.append(f"{s['pass']:<15} | {s['method']:<10} | {s['layers']:<8} | {s['inj']:>10.1f}% | {s['delta']:.8f}")
            total_delta += s['delta']
            
        lines.append("-" * 75)
        status = "STABLE"
        if total_delta > 0.015: status = "SATURATED"
        if total_delta > 0.030: status = "VOLATILE"
        
        lines.append(f"{'TOTAL MODEL SHIFT':<42} | {total_delta:.8f}")
        lines.append(f"{'STABILITY CHECK':<42} | {status}")
        lines.append("="*75 + "\n")
        return "\n".join(lines)

    def save_master(self, path):
        print(f"💾 Saving High-Precision Master to: {path}")
        master_sd = {k: v.contiguous().cpu() for k, v in self.base_dict.items()}
        save_file(master_sd, path, metadata={"modelspec.architecture": "wan_2.2_video"})
        del master_sd
        self._cleanup()
        return path

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()