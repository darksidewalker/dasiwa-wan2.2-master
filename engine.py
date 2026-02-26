import torch
import os, gc, re, json
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gpu_mm(B, A):
    """Offloads matrix math to GPU using float32 for Wan 2.2 precision."""
    with torch.no_grad():
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
        
        print(f"📦 Loading Wan 2.2 MoE Component: {self.paths['base_model']}...")
        self.base_dict = load_file(self.paths['base_model'])
        self.base_keys = list(self.base_dict.keys())

        # --- WAN 2.2 MoE ROLES ---
        # HIGH = Motion Model (Temporal Dynamics)
        # LOW  = Refiner Model (Spatial Details)
        self.is_motion_base = "high" in self.paths['base_model'].lower()
        self.is_spatial_refiner = "low" in self.paths['base_model'].lower()
        self.role_label = "MOTION (High)" if self.is_motion_base else "REFINER (Low)"
            
        print(f"🛰️ Engine initialized for {self.role_label}.")

    def get_dynamic_target(self, concept):
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
        conflict_log = []
        styles = step.get('styles', [])
        use_shield = step.get('block_shield', False)
        
        style_dicts = []
        for s in styles:
            path = os.path.join(self.paths['lora_dir'], s['file'])
            if os.path.exists(path):
                style_dicts.append(load_file(path))
            else:
                print(f"⚠️ LoRA missing: {path}")

        unique_concepts = set()
        for sd in style_dicts:
            for k in sd.keys():
                c = re.sub(r'\.(lora_up|lora_down|alpha|lora_A|lora_B|default).*', '', k)
                c = c.replace("lora_unet.", "").replace("transformer.", "").replace("diffusion_model.", "")
                unique_concepts.add(c)

        for concept in unique_concepts:
            if self.abort_requested: return "ABORTED"
            target_key = self.get_dynamic_target(concept)
            if not target_key: continue

            if use_shield and self.is_skeletal(target_key):
                continue 

            layer_deltas = []
            for j, sd in enumerate(style_dicts):
                up_k = next((k for k in sd.keys() if concept in k.replace("_", ".") and (".up" in k or "_B" in k or ".lora_B" in k)), None)
                down_k = next((k for k in sd.keys() if concept in k.replace("_", ".") and (".down" in k or "_A" in k or ".lora_A" in k)), None)

                if up_k and down_k:
                    # Pure JSON Weight logic
                    w = styles[j]['weight'] * global_mult
                    delta = gpu_mm(sd[up_k], sd[down_k]) * w
                    layer_deltas.append(delta)

            if len(layer_deltas) > 1:
                stacked = torch.stack(layer_deltas)
                signs = torch.sign(stacked)
                dom_sign = torch.sign(signs.sum(dim=0))
                if not torch.all(signs[0] == signs):
                    conflict_log.append(target_key)
                mask = (signs == dom_sign).float()
                aligned = (stacked * mask).sum(dim=0) / (stacked != 0).sum(dim=0).clamp(min=1)
                self.apply_delta(target_key, aligned)
            elif len(layer_deltas) == 1:
                self.apply_delta(target_key, layer_deltas[0])

        del style_dicts
        self._cleanup()
        return conflict_log

    def is_skeletal(self, key):
        match = re.search(r'blocks?\.(\d+)\.', key.replace('_', '.'))
        if not match: return False
        block_idx = int(match.group(1))
        # Protecting motion physics (High) vs textures (Low)
        if self.is_motion_base:
            return block_idx <= 10 
        else:
            return block_idx <= 4  

    def apply_delta(self, target_key, delta):
        if self.base_dict[target_key].shape != delta.shape:
            delta = delta.t()
        self.base_dict[target_key] = (self.base_dict[target_key].to(torch.float32) + delta).to(self.base_dict[target_key].dtype)

    def save_and_patch(self, use_ramdisk=True):
        """FLATTENED 4D: Memory-Safe Pop Logic for 64GB Systems."""
        out_dir = "/mnt/ramdisk" if (use_ramdisk and os.path.exists("/mnt/ramdisk")) else "models"
        path = os.path.join(out_dir, "wan22_intermediate_flattened.safetensors")
        
        meta = {
            "modelspec.architecture": "wan_2.2_video",
            "diffusion_model.tensor_layout": "5d_flattened_to_4d",
            "modelspec.variant": "motion_base" if self.is_motion_base else "spatial_refiner"
        }

        patched = {}
        # We use list(keys) so we can modify the dict size while iterating
        for k in list(self.base_dict.keys()):
            v = self.base_dict.pop(k) # REMOVE FROM RAM IMMEDIATELY
            if len(v.shape) == 5:
                meta[f"comfy.gguf.orig_shape.{k}"] = json.dumps(list(v.shape))
                patched[k] = v.reshape(-1, *v.shape[-3:]).contiguous()
            else: 
                patched[k] = v.contiguous()

        save_file(patched, path, metadata=meta)
        del patched
        self._cleanup()
        return path

    def save_pure_5d(self, use_ramdisk=True):
        """NATIVE 5D: Memory-Safe Pop Logic for 64GB Systems."""
        out_dir = "/mnt/ramdisk" if (use_ramdisk and os.path.exists("/mnt/ramdisk")) else "models"
        path = os.path.join(out_dir, "wan22_intermediate_native.safetensors")
        
        meta = {
            "modelspec.architecture": "wan_2.2_video",
            "diffusion_model.tensor_layout": "5d_native",
            "modelspec.variant": "motion_base" if self.is_motion_base else "spatial_refiner"
        }

        processed = {}
        for k in list(self.base_dict.keys()):
            v = self.base_dict.pop(k) # REMOVE FROM RAM IMMEDIATELY
            processed[k] = v.contiguous()

        save_file(processed, path, metadata=meta)
        del processed
        self._cleanup()
        return path

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()