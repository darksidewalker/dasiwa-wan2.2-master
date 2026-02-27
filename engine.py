import torch
import os, gc, re, json, datetime
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.recipe = recipe_data
        base_path = recipe.get('paths', {}).get('base_model', "").lower()
        self.is_motion_base = "high" in base_path or "i2v" in base_path
        self.role_label = "HIGH-NOISE (Motion)" if self.is_motion_base else "LOW-NOISE (Refiner)"
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

    def validate_recipe_compatibility(self):
        print(f"🛡️ VALIDATING RECIPE: Ensuring LoRA compatibility for {self.role_label}...")
        
        # Determine the "Forbidden" keyword based on the active model
        forbidden = "low" if self.is_motion_base else "high"
        required = "high" if self.is_motion_base else "low"
        
        issues = []
        for step in self.recipe.get('pipeline', []):
            for feature in step.get('features', []):
                filename = feature['file'].lower()
                
                # Check for explicit mismatches
                if forbidden in filename:
                    issues.append(f"❌ MISMATCH: LoRA '{feature['file']}' contains '{forbidden}' but engine is {self.role_label}.")
                
                # Check for missing expected labels (Optional Warning)
                if required not in filename:
                    # We don't block the merge, just notify
                    pass 

        if issues:
            print("\n".join(issues))
            confirm = input("\n⚠️ Critical compatibility issues found. Continue anyway? (y/n): ")
            if confirm.lower() != 'y':
                print("🛑 Operation aborted by user.")
                exit()
        else:
            print("✅ Recipe validation passed: No noise-level conflicts detected.")
    
    def get_compatibility_report(self):
        """Returns list of LoRAs that contain the 'forbidden' noise-level keyword."""
        forbidden = "low" if self.is_motion_base else "high"
        mismatches = []
        
        pipeline = self.recipe.get('pipeline', [])
        for step in pipeline:
            for feature in step.get('features', []):
                fname = feature['file'].lower()
                if forbidden in fname:
                    mismatches.append(feature['file'])
        return mismatches

    def scan_lora_density(self, lora_sd):
        mags = [v.abs().mean().item() for k, v in lora_sd.items() if "weight" in k or ".diff" in k]
        avg_mag = sum(mags) / len(mags) if mags else 0
        return avg_mag

    def find_lora_keys(self, lora_sd, target_key):
        clean_target = target_key.replace("diffusion_model.", "").replace(".weight", "")
        is_2d_target = len(self.base_dict[target_key].shape) == 2
        underscore_target = clean_target.replace(".", "_")

        # Expanded pairs to handle mixed dot/underscore separators
        pairs = [
            (".lora_B", ".lora_A"),
            (".lora_up", ".lora_down"),
            ("_lora_up", "_lora_down"),
            (".lora_up.weight", ".lora_down.weight"),
            # Safety for Kohya-style mixed naming
            ("_lora_B", "_lora_A"), 
        ]

        for k in lora_sd.keys():
            # Check for Dot-style OR Underscore-style match
            if clean_target in k or underscore_target in k:
                for up_suf, down_suf in pairs:
                    if up_suf in k:
                        up_k = k
                        # Dynamic suffix replacement to keep the prefix intact
                        down_k = k.replace(up_suf, down_suf)
                        if down_k in lora_sd:
                            return up_k, down_k, "matrix"
        
        # Fallback for vectors/alphas/biases
        if not is_2d_target:
            for k in lora_sd.keys():
                if clean_target in k or underscore_target in k:
                    # Added '_alpha' and '.bias' coverage
                    if any(x in k for x in [".diff", ".alpha", ".bias", "_alpha", "_bias"]):
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
        try:
            features = step.get('features', [])
            lora_dir = self.paths.get('lora_dir', 'loras')
            method = str(step.get('method', 'addition')).lower().strip()
            use_smart = (method == "injection")
            
            pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
            pass_stats = {"matrix": 0, "vector": 0, "dense": 0, "skipped": 0}
            total_inj, pass_peaks = [], 0
            
            for f in features:
                lora_path = os.path.join(lora_dir, f['file'])
                if not os.path.exists(lora_path): continue

                lora_sd = load_file(lora_path)
                avg_mag = self.scan_lora_density(lora_sd)
                asked_weight = float(f.get('weight', 1.0))
                
                # Visibility: Track which LoRA keys were successfully used
                lora_keys_in_file = [k for k in lora_sd.keys() if "weight" in k]
                used_lora_keys = set()

                print(f"🔍 SCAN [{f['file']}]")
                print(f"   ├─ Raw Signal: {avg_mag:.4f}")
                print(f"   ├─ Multiplier: x{asked_weight:.2f} (Target: {avg_mag * asked_weight:.4f})")
                print(f"   └─ Mode: {'INJECTION' if use_smart else 'ADDITION'}")

                for target_key in self.base_keys:
                    # Use our new strict Architecture-Aware key finder
                    k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                    
                    # Manual fallback for dense (full-weight) merges if exists
                    dk = None
                    if not k1:
                        dk = target_key if target_key in lora_sd else (f"diffusion_model.{target_key}" if f"diffusion_model.{target_key}" in lora_sd else None)

                    with torch.no_grad():
                        delta, inj_val, hit = None, 100.0, False
                        
                        if k_type == "matrix":
                            used_lora_keys.add(k1)
                            if k2: used_lora_keys.add(k2)
                            delta, inj_val, hit = self.calculate_delta(lora_sd, k1, k2, f['weight'], global_mult, target_key, smart_logic=use_smart)
                        
                        elif k_type == "vector":
                            used_lora_keys.add(k1)
                            delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        
                        elif dk:
                            k_type = "dense"
                            used_lora_keys.add(dk)
                            delta = (lora_sd[dk].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        
                        if hit: pass_peaks += 1
                        
                        # Apply the delta using the CPU-handshake method
                        if delta is not None:
                            if self.apply_delta(target_key, delta):
                                pass_stats[k_type] += 1
                                if k_type in ["matrix", "dense"]: total_inj.append(inj_val)
                            else:
                                pass_stats["skipped"] += 1

                # Check for orphaned keys (LoRA weights that found no home in the base model)
                orphans = len([k for k in lora_keys_in_file if k not in used_lora_keys])
                if orphans > 0:
                    print(f"   ℹ️  INFO: {orphans} keys in LoRA had no matching layers in Base (Structural Mismatch).")
                    pass_stats["skipped"] += orphans
                
                if pass_peaks > 0 and use_smart:
                    print(f"   ⚠️ PEAK ALERT: Limiter squashed {pass_peaks} layers. (Too hot!)")
                self._cleanup()

            post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
            avg_injection = sum(total_inj) / len(total_inj) if total_inj else 100.0
            
            self.summary_data.append({
                "pass": step.get('pass_name', 'Pass'),
                "method": method.upper(),
                "layers": sum(v for k,v in pass_stats.items() if k != "skipped"),
                "inj": avg_injection,
                "peaks": pass_peaks,
                "delta": abs(post_mean - pre_mean)
            })
        except KeyboardInterrupt:
            print(f"\n🛑 ENGINE INTERRUPT: Pass '{step.get('pass_name')}' halted.")
            # We don't 'exit' here, we let the exception bubble up to app.py 
            # after we clean up our local mess.
            raise 
        finally:
            # This runs NO MATTER WHAT (success or Ctrl+C)
            self._cleanup()
            torch.cuda.empty_cache()

    def apply_delta(self, target_key, delta):
        base = self.base_dict[target_key]
        
        # 1. THE SILENT GUARD: Filter out specialized layers (Convnd) 
        # that LoRAs often flatten. This stops the log spam.
        if base.ndim > 2 and delta.ndim <= 2:
            return False # Exit silently, no log entry created

        # 2. SHAPE CORRECTION (Standard 2D Weights)
        if delta.numel() == base.numel():
            delta = delta.reshape(base.shape)
        elif delta.ndim == 2 and delta.t().shape == base.shape:
            delta = delta.t()
        else:
            # ONLY log if it's a real anomaly (e.g. two 2D layers that don't match)
            print(f"⚠️ SHAPE MISMATCH at {target_key}: {delta.shape} vs {base.shape}")
            return False

        # 3. HIGH-PRECISION ADDITION (CPU Handshake)
        with torch.no_grad():
            # Move delta to CPU to meet the 14B base
            updated_weight = base.to(torch.float32) + delta.to("cpu", dtype=torch.float32)
            
            # Cast back to BF16/FP16 and store
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
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        print(f"💾 EXPORT: Finalizing {os.path.basename(path)}...")
        full_log = self.get_metadata_string()
        custom_metadata = {"comment": full_log, "dasiwa_summary": full_log}
        
        # We process weights one by one to CPU to avoid RAM spikes on 14B models
        master_sd = {}
        for k in self.base_keys:
            master_sd[k] = self.base_dict[k].contiguous().cpu()
        
        try:
            save_file(master_sd, path, metadata=custom_metadata)
            print(f"✅ SUCCESS: High-Precision Master saved to {path}")
        except Exception as e:
            print(f"❌ EXPORT FAILED: {str(e)}")
            
        self._cleanup()
        return path

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()