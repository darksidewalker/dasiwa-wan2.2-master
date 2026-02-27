import torch
import os, gc, re, json
from safetensors.torch import load_file, save_file

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.recipe = recipe_data
        self.paths = self.recipe.get('paths', {})
        
        # Load base model to CPU to keep VRAM open for MatMul
        base_path = self.paths.get('base_model', "")
        if not base_path or not os.path.exists(base_path):
            raise FileNotFoundError(f"❌ Base model not found at {base_path}")
            
        print(f"📥 Loading Base Model: {os.path.basename(base_path)}")
        self.base_dict = load_file(base_path)
        self.base_keys = list(self.base_dict.keys())
        
        # Identity Logic for Wan 2.2
        self.is_motion_base = "high" in base_path.lower()
        self.role_label = "MOTION (14B High Noise)" if self.is_motion_base else "REFINER (14B Low Noise)"
        print(f"🛰️ Engine: {self.role_label} Active.")

    def find_lora_keys(self, lora_sd, target_key):
        """Maps Wan 2.2 Base keys to LoRA keys (Standard & VBVR)."""
        possible_bases = [target_key, f"diffusion_model.{target_key}"]
        for base in possible_bases:
            clean = base.replace(".weight", "").replace("_", ".")
            # Matrix check
            up = next((k for k in lora_sd.keys() if clean in k.replace("_", ".") and (".lora_up" in k or ".lora_B" in k)), None)
            if up:
                down = up.replace(".lora_up", ".lora_down") if ".lora_up" in up else up.replace(".lora_B", ".lora_A")
                if down in lora_sd: return up, down, "matrix"
            # Vector (VBVR) check
            diff = next((k for k in lora_sd.keys() if clean in k.replace("_", ".") and (".diff" in k or ".diff_b" in k)), None)
            if diff: return diff, None, "vector"
        return None, None, None

    def process_pass(self, step, global_mult):
        """Merges LoRA features and prints stats to the terminal."""
        features = step.get('features', [])
        lora_dir = self.paths.get('lora_dir', 'loras')
        
        for f in features:
            stats = {"matrix": 0, "vector": 0, "fail": 0}
            lora_path = os.path.join(lora_dir, f['file'])
            
            if not os.path.exists(lora_path):
                print(f"⚠️ LoRA Not Found: {f['file']}")
                continue

            print(f"💉 Processing: {f['file']}")
            lora_sd = load_file(lora_path)
            
            for target_key in self.base_keys:
                k1, k2, k_type = self.find_lora_keys(lora_sd, target_key)
                if not k_type: continue
                
                try:
                    with torch.no_grad():
                        if k_type == "matrix":
                            delta = self.calculate_delta(lora_sd, k1, k2, f['weight'], global_mult)
                        else: # vector
                            delta = (lora_sd[k1].to(torch.float32) * float(f['weight']) * global_mult).cpu()
                        
                        if delta is not None:
                            if self.apply_delta(target_key, delta):
                                stats[k_type] += 1
                            else: stats["fail"] += 1
                except:
                    stats["fail"] += 1
            
            print(f"✅ {f['file']} | Matrices: {stats['matrix']} | Vectors: {stats['vector']} | Fails: {stats['fail']}")
            del lora_sd
            self._cleanup()

    def calculate_delta(self, lora_sd, up_k, down_k, weight, global_mult):
        """Performs Rank-Decomposition with Alpha Scaling."""
        with torch.no_grad():
            up_w = lora_sd[up_k].to(DEVICE, dtype=torch.float32)
            down_w = lora_sd[down_k].to(DEVICE, dtype=torch.float32)
            
            if up_w.shape[1] == down_w.shape[1]: down_w = down_w.t()
            elif up_w.shape[0] == down_w.shape[0]: up_w = up_w.t()
            
            try: delta = torch.matmul(up_w, down_w)
            except: return None
            
            rank = up_w.shape[1]
            a_key = up_k.replace("lora_B", "alpha").replace("lora_up", "alpha")
            a_val = lora_sd.get(a_key, torch.tensor(float(rank)))
            a = a_val.float().flatten()[0].item() if a_val.numel() > 0 else float(rank)
            
            return (delta * (a / rank if rank != 0 else 1.0) * float(weight) * global_mult).to("cpu")

    def apply_delta(self, target_key, delta):
        """Applies delta on CPU to preserve VRAM."""
        base = self.base_dict[target_key]
        if delta.numel() == base.numel(): delta = delta.reshape(base.shape)
        elif delta.t().shape == base.shape: delta = delta.t()
        else: return False
        
        self.base_dict[target_key] = (base.to(torch.float32) + delta).to(base.dtype)
        return True

    def save_master(self, path):
        """The final high-precision save method."""
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