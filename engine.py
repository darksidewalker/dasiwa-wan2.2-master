import torch
import os, gc, re, json, subprocess
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gpu_mm(B, A):
    """Offloads matrix math to GPU using float32 for Wan 2.2 precision."""
    with torch.no_grad():
        # Wan 2.2 requires high precision for video latents; avoid FP16 here if possible
        B_g = B.to(DEVICE, dtype=torch.float32)
        A_g = A.to(DEVICE, dtype=torch.float32)
        res = torch.mm(B_g, A_g).to("cpu")
        del B_g, A_g
        return res

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.abort_requested = False
        self.bridge_cache = {}
        self.recipe = recipe_data
        self.paths = self.recipe['paths']
        
        print(f"📦 Loading Base Model: {self.paths['base_model']}...")
        self.base_dict = load_file(self.paths['base_model'])
        self.base_keys = list(self.base_dict.keys())
        self.is_high_res = "high" in self.paths['base_model'].lower()

    def get_dynamic_target(self, concept):
        """Maps LoRA keys (e.g. transformer.blocks.1.attn) to Base keys."""
        if concept in self.bridge_cache: return self.bridge_cache[concept]
        
        parts = re.split(r'[._]', concept)
        block_num = next((p for p in parts if p.isdigit()), None)
        components = [p for p in parts if not p.isdigit() and p != 'blocks']

        for bk in self.base_keys:
            if not bk.endswith(".weight"): continue
            bk_norm = bk.replace("_", ".")
            if block_num and f"blocks.{block_num}." in bk_norm:
                if all(comp in bk_norm for comp in components):
                    self.bridge_cache[concept] = bk
                    return bk
        return None

    def process_pass(self, step, global_mult):
        """THE CORE MERGING LOGIC: Iterates through styles and applies weights."""
        conflict_log = []
        styles = step.get('styles', [])
        use_shield = step.get('block_shield', False)
        # Wan 2.2 Low-Res variant needs a dampener to prevent over-saturation
        noise_dampener = 1.0 if self.is_high_res else 0.85
        
        # Load LoRAs for this pass
        style_dicts = []
        for s in styles:
            path = os.path.join(self.paths['lora_dir'], s['file'])
            if os.path.exists(path):
                style_dicts.append(load_file(path))
            else:
                print(f"⚠️ LoRA missing: {path}")

        # Extract all unique concepts across all LoRAs in this pass
        unique_concepts = set()
        for sd in style_dicts:
            for k in sd.keys():
                c = re.sub(r'\.(lora_up|lora_down|alpha|lora_A|lora_B|default).*', '', k)
                c = c.replace("lora_unet.", "").replace("transformer.", "").replace("diffusion_model.", "")
                unique_concepts.add(c)

        # Merge each concept
        for concept in unique_concepts:
            if self.abort_requested: return "ABORTED"
            
            target_key = self.get_dynamic_target(concept)
            if not target_key: continue

            # Skeletal Shielding (Blocks 0-8 handle core physics/structure)
            if use_shield and self.is_skeletal(target_key):
                continue 

            layer_deltas = []
            for j, sd in enumerate(style_dicts):
                # Flexible key matching for different LoRA naming conventions
                up_k = next((k for k in sd.keys() if concept in k.replace("_", ".") and (".up" in k or "_B" in k or ".lora_B" in k)), None)
                down_k = next((k for k in sd.keys() if concept in k.replace("_", ".") and (".down" in k or "_A" in k or ".lora_A" in k)), None)

                if up_k and down_k:
                    w = styles[j]['weight'] * global_mult * noise_dampener
                    delta = gpu_mm(sd[up_k], sd[down_k]) * w
                    layer_deltas.append(delta)

            # WEIGHT-SIGN ALIGNMENT (The 'Secret Sauce' for Video Models)
            if len(layer_deltas) > 1:
                stacked = torch.stack(layer_deltas)
                signs = torch.sign(stacked)
                dom_sign = torch.sign(signs.sum(dim=0))
                
                # Check for conflicts where LoRAs fight each other
                if not torch.all(signs[0] == signs):
                    conflict_log.append(target_key)
                
                # Aligned average: contributor must match the dominant sign to be included
                mask = (signs == dom_sign).float()
                aligned = (stacked * mask).sum(dim=0) / (stacked != 0).sum(dim=0).clamp(min=1)
                self.apply_delta(target_key, aligned)
            elif len(layer_deltas) == 1:
                self.apply_delta(target_key, layer_deltas[0])

        del style_dicts
        gc.collect()
        torch.cuda.empty_cache()
        return conflict_log

    def is_skeletal(self, key):
        match = re.search(r'blocks?\.(\d+)\.', key.replace('_', '.'))
        return match and int(match.group(1)) <= 8

    def apply_delta(self, target_key, delta):
        if self.base_dict[target_key].shape != delta.shape:
            delta = delta.t()
        self.base_dict[target_key] = (self.base_dict[target_key].to(torch.float32) + delta).to(self.base_dict[target_key].dtype)

    def save_and_patch(self, use_ramdisk=True):
        """Live 5D Reshaping & ComfyUI Metadata Injection."""
        out_dir = "/mnt/ramdisk" if (use_ramdisk and os.path.exists("/mnt/ramdisk")) else "models"
        path = os.path.join(out_dir, f"{self.paths['output_prefix']}_PATCHED.safetensors")
        
        patched, meta = {}, {}
        for k, v in self.base_dict.items():
            if len(v.shape) == 5:
                meta[f"comfy.gguf.orig_shape.{k}"] = json.dumps(list(v.shape))
                patched[k] = v.reshape(-1, *v.shape[-3:])
            else: patched[k] = v

        save_file(patched, path, metadata=meta)
        del self.base_dict
        gc.collect()
        torch.cuda.empty_cache()
        return path
    
    def save_to_ramdisk(self):
        """Saves the merged FP16 model to RAMDisk with 5D flattening for GGUF/Quant."""
        ram_path = os.path.join("/mnt/ramdisk", f"{self.paths['output_prefix']}_TMP.safetensors")
        
        # 5D Flattening & Metadata Injection for GGUF compatibility
        patched_dict = {}
        metadata = {}
        for k, v in self.base_dict.items():
            if len(v.shape) == 5:
                metadata[f"comfy.gguf.orig_shape.{k}"] = json.dumps(list(v.shape))
                patched_dict[k] = v.reshape(-1, *v.shape[-3:])
            else:
                patched_dict[k] = v

        save_file(patched_dict, ram_path, metadata=metadata)
        
        # Crucial: Free RAM before the heavy Quantization/GGUF subprocess starts
        del self.base_dict
        gc.collect()
        torch.cuda.empty_cache()
        return ram_path

    def run_cli_task(self, cmd):
        """Generic subprocess runner for llama.cpp/GGUF or Quant tools."""
        import subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return process