import torch
import os, gc, re, json, datetime
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from config import MODELS_DIR
from utils import verify_model_integrity, get_final_summary_string

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.recipe = recipe_data
        self.paths = self.recipe.get('paths', {})
        base_path = self.paths.get('base_model', "")
        
        if not base_path or not os.path.exists(base_path):
            raise FileNotFoundError(f"❌ Base model not found at {base_path}")

        # --- STREAMING SETUP ---
        self.handle = safe_open(base_path, framework="pt", device="cpu")
        self.base_keys = list(self.handle.keys())
        
        # FP32 Buffer to prevent bit-drift across 8 sections
        self.modified_tensors = {}
        self.summary_data = []
        
        # Metadata for UI
        base_lower = base_path.lower()
        self.is_motion_base = "high_noise" in base_lower
        self.role_label = "MOTION (14B High)" if self.is_motion_base else "REFINER (14B Low)"
        self.router_regex = re.compile(r"(\.gate$|\.router$|\.wg$|layer_norm_moe)", re.IGNORECASE)

    def get_compatibility_report(self):
        forbidden = "low" if self.is_motion_base else "high"
        mismatches = []
        pipeline = self.recipe.get('pipeline', [])
        for step in pipeline:
            for feature in step.get('features', []):
                if forbidden in feature['file'].lower():
                    mismatches.append(feature['file'])
        return mismatches

    def find_lora_keys(self, lora_sd, target_key):
        """Silveroxides Token-Based Matcher with Alpha Recovery"""
        base_shape = self.handle.get_slice(target_key).get_shape()
        clean_target = target_key.replace("diffusion_model.", "").replace(".weight", "")
        t_tokens = set(re.findall(r'[a-zA-Z]+|\d+', clean_target))
        
        pairs = [(".lora_B", ".lora_A"), (".lora_up", ".lora_down"), (".lora_B.weight", ".lora_A.weight")]

        for k in lora_sd.keys():
            l_tokens = set(re.findall(r'[a-zA-Z]+|\d+', k))
            if t_tokens.issubset(l_tokens):
                for up_suf, down_suf in pairs:
                    if up_suf in k:
                        down_k = k.replace(up_suf, down_suf)
                        if down_k in lora_sd:
                            up_s, dn_s = lora_sd[k].shape, lora_sd[down_k].shape
                            if (up_s[0] == base_shape[-2] and dn_s[1] == base_shape[-1]) or \
                               (dn_s[0] == base_shape[-2] and up_s[1] == base_shape[-1]) or \
                               (up_s[1] == base_shape[-2] and dn_s[0] == base_shape[-1]):
                                
                                # Find Alpha for Silveroxides scaling
                                alpha_key = k.replace(up_suf, ".alpha")
                                alpha = lora_sd.get(alpha_key, torch.tensor(up_s[1])).item()
                                return k, down_k, alpha
        return None, None, None

    def calculate_delta(self, lora_sd, k1, k2, alpha, weight, global_mult):
        """Precise Silveroxides Math: (Up @ Down) * (Alpha / Rank)"""
        t1 = lora_sd[k1].to("cuda", dtype=torch.float32)
        t2 = lora_sd[k2].to("cuda", dtype=torch.float32)
        
        if t1.shape[1] == t2.shape[0]: res = t1 @ t2
        elif t2.shape[1] == t1.shape[0]: res = t2 @ t1
        elif t1.shape[0] == t2.shape[0]: res = t1.T @ t2
        elif t1.shape[1] == t2.shape[1]: res = t1 @ t2.T
        else: return None
        
        rank = t1.shape[1] if t1.shape[1] < t1.shape[0] else t1.shape[0]
        scaling = alpha / rank if rank != 0 else 1.0
        
        return (res * (float(weight) * float(global_mult) * scaling)).to("cpu")

    def process_pass(self, pass_data, global_mult):
        """INDENTED CORRECTLY: Now recognizes 'self'"""
        pass_name = pass_data.get('pass_name', 'Unknown')
        features = pass_data.get('features', [])
        lora_dir = self.paths.get('lora_dir', '')
        
        loaded_loras = []
        for f in features:
            l_path = os.path.join(lora_dir, f['file'])
            if os.path.exists(l_path):
                loaded_loras.append({'sd': load_file(l_path), 'weight': f['weight']})
        
        if not loaded_loras:
            yield f"  ⚠️ No LoRAs found for section: {pass_name}"
            return

        yield f"  🚀 Section: {pass_name} ({len(loaded_loras)} LoRAs)"
        merged_count, total_shift = 0, 0.0

        for i, target_key in enumerate(self.base_keys):
            if i % 200 == 0:
                yield f"    ... Progress: {i}/{len(self.base_keys)} layers"

            layer_deltas = []
            for lora in loaded_loras:
                k1, k2, alpha = self.find_lora_keys(lora['sd'], target_key)
                if k1 and k2:
                    delta = self.calculate_delta(lora['sd'], k1, k2, alpha, lora['weight'], global_mult)
                    if delta is not None:
                        layer_deltas.append(delta)

            if layer_deltas:
                # Use FP32 throughout the merge to prevent rounding errors
                if target_key in self.modified_tensors:
                    base_t = self.modified_tensors[target_key].to("cuda", dtype=torch.float32)
                else:
                    base_t = self.handle.get_tensor(target_key).to("cuda", dtype=torch.float32)
                
                combined_delta = torch.stack(layer_deltas).sum(dim=0).to("cuda")
                
                if base_t.ndim == 5 and combined_delta.ndim == 2:
                    combined_delta = combined_delta.view(1, 1, 1, *combined_delta.shape)
                elif base_t.ndim == 3 and combined_delta.ndim == 2:
                     combined_delta = combined_delta.unsqueeze(0)
                
                new_w = base_t + combined_delta
                total_shift += combined_delta.abs().mean().item()
                # Store in FP32 until the final Save call
                self.modified_tensors[target_key] = new_w.to("cpu", dtype=torch.float32)
                merged_count += 1
                del base_t, combined_delta, layer_deltas, new_w

        for l in loaded_loras: del l['sd']
        gc.collect()
        torch.cuda.empty_cache()

        shift_avg = total_shift / merged_count if merged_count > 0 else 0
        self.summary_data.append({"pass": pass_name, "layers": merged_count, "delta": shift_avg})
        yield f"  ✅ {pass_name} Complete."

    def run_pre_save_check(self):
        yield "  🛡️ FINAL INTEGRITY CHECK..."
        fail_count = 0
        for key, tensor in self.modified_tensors.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                yield f"  ❌ CRITICAL: Numerical Error in {key}"
                fail_count += 1
        if fail_count == 0:
            yield "  ✅ All modified tensors are stable."

    def get_final_summary(self, quant):
        summary_text = get_final_summary_string(self.summary_data, self.role_label)
        header = f"Title: {self.paths.get('title', 'ActionMaster')}\nExport: {quant}\n"
        return f"{header}\n{summary_text}\n🚀 Powered by ActionMaster Engine"

    def save_master(self, path):
        print(f"💾 SSD STREAM-WRITE: {os.path.basename(path)}")
        final_dict = {}
        for k in self.base_keys:
            if k in self.modified_tensors:
                # Final Downcast to BF16 only during disk-write
                final_dict[k] = self.modified_tensors[k].to(torch.bfloat16)
                del self.modified_tensors[k] 
            else:
                final_dict[k] = self.handle.get_tensor(k).to(torch.bfloat16)

        save_file(final_dict, path)
        self.handle = None 
        gc.collect()
        return path