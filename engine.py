import torch
import os, gc, re, json
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gpu_mm(B, A):
    """Offloads matrix math to GPU using float32 for Wan 2.2 precision."""
    with torch.no_grad():
        B_gpu = B.to(DEVICE, dtype=torch.float32)
        A_gpu = A.to(DEVICE, dtype=torch.float32)
        res = torch.mm(B_gpu, A_gpu).to("cpu")
        del B_gpu, A_gpu
        return res

class ActionMasterEngine:
    def __init__(self, recipe_data):
        self.abort_requested = False
        self.bridge_cache = {}
        
        # 1. Handle Recipe (if string, parse it; if dict, use it)
        if isinstance(recipe_data, str):
            try:
                self.recipe = json.loads(re.sub(r'//.*', '', recipe_data))
            except: # If it's a file path
                with open(recipe_data, 'r') as f:
                    self.recipe = json.loads(re.sub(r'//.*', f.read()))
        else:
            self.recipe = recipe_data
        
        self.paths = self.recipe['paths']
        
        # 2. Load Base Model
        print(f"📦 Loading Base Model: {self.paths['base_model']}...")
        self.base_dict = load_file(self.paths['base_model'])
        self.base_keys = list(self.base_dict.keys())

        # 3. Wan 2.2 Integrity Check
        self.is_high_res = "high" in self.paths['base_model'].lower()
        print(f"🧬 Wan 2.2 Mode: {'HIGH' if self.is_high_res else 'LOW'} Noise Variant detected.")

    def get_dynamic_target(self, concept):
        """Learns the bridge between LoRA and Base keys on the fly."""
        if concept in self.bridge_cache:
            return self.bridge_cache[concept]

        # Extract block number and components
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
        density = step.get('density', 1.0)
        noise_dampener = 1.0 if self.is_high_res else 0.85
        
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
                # Clean LoRA key to find the core concept
                concept = re.sub(r'\.(lora_up|lora_down|alpha|lora_A|lora_B|default).*', '', k)
                concept = concept.replace("lora_unet.", "").replace("transformer.", "").replace("diffusion_model.", "")
                unique_concepts.add(concept)

        for concept in unique_concepts:
            if self.abort_requested: return "ABORTED"
            
            target_key = self.get_dynamic_target(concept)
            if not target_key: continue

            is_skeletal = self.is_skeletal(target_key)
            layer_deltas = []

            for j, sd in enumerate(style_dicts):
                # Apply shielding for skeletal blocks (0-8)
                if use_shield and is_skeletal: 
                    # Only skip if this specific LoRA isn't allowed to modify skeleton
                    if not styles[j].get("modify_skeleton", False): continue

                # Look for the weight pairs
                # FIX: More robust matching for 'A/B' vs 'up/down' naming
                up_k = next((k for k in sd.keys() if concept in k.replace("_", ".") and (".up" in k or "_B" in k or ".lora_B" in k)), None)
                down_k = next((k for k in sd.keys() if concept in k.replace("_", ".") and (".down" in k or "_A" in k or ".lora_A" in k)), None)

                if up_k and down_k:
                    # Weight calculation including your 0.85 low-res dampener
                    calc_weight = styles[j]['weight'] * global_mult * noise_dampener
                    d = gpu_mm(sd[up_k], sd[down_k]) * calc_weight
                    
                    if density < 1.0:
                        mask = (torch.rand_like(d) < density).float()
                        d = (d * mask) / density
                    
                    layer_deltas.append(d)

            # Weight-Sign Alignment Logic
            if len(layer_deltas) > 1:
                stacked = torch.stack(layer_deltas)
                signs = torch.sign(stacked)
                dom_sign = torch.sign(signs.sum(dim=0))
                
                # Check for sign conflict
                if not torch.all(signs[0] == signs):
                    conflict_log.append(target_key)
                
                # Aligned average: only use contributors that match the dominant sign
                aligned = (stacked * (signs == dom_sign).float()).sum(dim=0) / (stacked != 0).sum(dim=0).clamp(min=1)
                self.apply_delta(target_key, aligned)
            elif len(layer_deltas) == 1:
                self.apply_delta(target_key, layer_deltas[0])

        del style_dicts
        gc.collect()
        torch.cuda.empty_cache()
        return conflict_log
    
    def is_skeletal(self, key):
        match = re.search(r'blocks?\.(\d+)\.', key.replace('_', '.'))
        if match:
            return int(match.group(1)) <= 8
        return False

    def apply_delta(self, target_key, delta):
        if self.base_dict[target_key].shape != delta.shape:
            delta = delta.t()
        dtype = self.base_dict[target_key].dtype
        self.base_dict[target_key] = (self.base_dict[target_key].to(torch.float32) + delta).to(dtype)

    def get_heatmap_stats(self, conflicts):
        early, mid, late = 0, 0, 0
        for c in conflicts:
            match = re.search(r'blocks?\.(\d+)\.', c.replace('_', '.'))
            if match:
                blk = int(match.group(1))
                if blk <= 8: early += 1
                elif 9 <= blk <= 30: mid += 1
                else: late += 1
        return f"📊 PASS REPORT\n  🔹 [Structure] B0-8: {early} | 🟢 [Motion] B9-30: {mid} | 🟡 [Aesthetic] B31+: {late}"

    def save(self):
        final_path = f"{self.paths['output_prefix']}_FINAL.safetensors"
        save_file(self.base_dict, final_path)
        return final_path