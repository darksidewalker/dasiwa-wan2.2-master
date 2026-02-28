import torch
import os, gc, re, json, datetime
from safetensors.torch import load_file, save_file
from config import MODELS_DIR

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

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult, target_key, smart_logic=True):
        with torch.no_grad():
            up_w = lora_sd[up_k].to(DEVICE, dtype=torch.float32)
            down_w = lora_sd[down_k].to(DEVICE, dtype=torch.float32)
            if up_w.shape[1] != down_w.shape[0]:
                if up_w.shape[1] == down_w.shape[1]: down_w = down_w.t()
                elif up_w.shape[0] == down_w.shape[0]: up_w = up_w.t()
            rank = up_w.shape[1]
            alpha_key = up_k.split('.lora')[0] + ".alpha"
            alpha_val = lora_sd.get(alpha_key, torch.tensor(float(rank)))
            scale = alpha_val.item() / rank if rank != 0 else 1.0
            delta = torch.matmul(up_w, down_w) * scale
            passed_pct, limit_hit = 100.0, False
            if smart_logic:
                base_w = self.base_dict[target_key].to(DEVICE, dtype=torch.float32)
                base_var = torch.var(base_w).item()
                is_new_info = torch.var(delta).item() > (base_var * 1.5)
                gate = delta.abs().mean() * (0.4 if is_new_info else 1.8)
                mask = delta.abs() > gate
                delta = torch.where(mask, delta, torch.zeros_like(delta))
                passed_pct = (mask.sum().item() / mask.numel()) * 100
                limit = 0.08 if is_new_info else 0.05
                if delta.abs().max() > limit:
                    delta *= (limit / delta.abs().max())
                    limit_hit = True
            return (delta * float(weight) * global_mult).to("cpu"), passed_pct, limit_hit

    def process_pass(self, step, global_mult):
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        method = str(step.get('method', 'addition')).lower().strip()
        use_smart = (method == "injection")
        pre_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        pass_stats, total_inj, pass_peaks = {"matrix": 0, "vector": 0, "dense": 0, "skipped": 0}, [], 0
        for f in features:
            lora_path = os.path.join(lora_dir, f['file'])
            if not os.path.exists(lora_path): continue
            lora_sd = load_file(lora_path)
            used_lora_keys = set()
            for target_key in self.base_keys:
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                dk = target_key if target_key in lora_sd else None
                delta, inj_val, hit = None, 100.0, False
                if k_type == "matrix":
                    used_lora_keys.update([k1, k2])
                    delta, inj_val, hit = self.calculate_delta(lora_sd, k1, k2, f['weight'], global_mult, target_key, use_smart)
                elif k_type == "vector":
                    used_lora_keys.add(k1)
                    delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                elif dk:
                    used_lora_keys.add(dk)
                    delta, k_type = (lora_sd[dk].to(torch.float32) * float(f['weight']) * global_mult).cpu(), "dense"
                if hit: pass_peaks += 1
                if delta is not None and self.apply_delta(target_key, delta):
                    pass_stats[k_type] += 1
                    if k_type in ["matrix", "dense"]: total_inj.append(inj_val)
                else: pass_stats["skipped"] += 1
            self._cleanup()
        post_mean = sum(v.abs().mean().item() for v in self.base_dict.values()) / len(self.base_dict)
        self.summary_data.append({"pass": step.get('pass_name', 'Pass'), "method": method.upper(), "layers": sum(v for k,v in pass_stats.items() if k != "skipped"), "inj": (sum(total_inj)/len(total_inj) if total_inj else 100.0), "peaks": pass_peaks, "delta": abs(post_mean - pre_mean)})

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
        header = f"Title: {self.paths.get('title', 'Dasiwa Master')}\nExport: {quant}\n"
        return f"{header}\n{self.get_final_summary_string()}\n🚀 Made with DaSiWa"

    def save_master(self, path):
        custom_metadata = {"comment": self.get_metadata_string(), "dasiwa_summary": self.get_metadata_string()}
        try:
            serialized_dict = {k: v.contiguous() for k, v in self.base_dict.items()}
            save_file(serialized_dict, path, metadata=custom_metadata)
            del serialized_dict
        finally:
            self.base_dict = None
            self._cleanup()
        return path

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()