import torch
import os, gc, re, json
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gpu_mm(B, A):
    """Offloads matrix math to GPU using half-precision for Wan 2.2 speed."""
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda"):
            res = torch.mm(B.to(DEVICE), A.to(DEVICE)).to("cpu")
        torch.cuda.empty_cache()
        return res

class ActionMasterEngine:
    def __init__(self, recipe_path):
        with open(recipe_path, 'r') as f:
            content = f.read()
            clean_content = re.sub(r'//.*', '', content)
            self.recipe = json.loads(clean_content)

        self.paths = self.recipe['paths']
        self.base_dict = load_file(self.paths['base_model'])
        self.base_keys = list(self.base_dict.keys())

        # Wan 2.2 Integrity Check
        self.is_high_res = "high" in self.paths['base_model'].lower()
        print(f"🧬 Wan 2.2 Mode: {'HIGH' if self.is_high_res else 'LOW'} Noise Variant detected.")

    def process_pass(self, step, global_mult):
        conflict_log = []
        use_shield = step.get('block_shield', False)

        # Adaptive Weighting for Wan 2.2 Low
        # If running on 'low' model, we slightly dampen weights to prevent over-saturation
        noise_dampener = 1.0 if self.is_high_res else 0.85

        styles = step.get('styles', [])
        if styles:
            density = step.get('density', 0.6)
            style_dicts = [load_file(os.path.join(self.paths['lora_dir'], s['file'])) for s in styles]
            all_keys = set().union(*(set(self.clean_k(k) for k in sd.keys()) for sd in style_dicts))

            for core in all_keys:
                target_key = next((bk for bk in self.base_keys if core in bk and bk.endswith(".weight")), None)
                if not target_key: continue

                is_skeletal = self.is_skeletal(target_key)
                layer_deltas = []
                for j, sd in enumerate(style_dicts):
                    if use_shield and is_skeletal: continue
                    if is_skeletal and not styles[j].get("modify_skeleton", False): continue

                    down_k = next((k for k in sd.keys() if core in self.clean_k(k) and ("down" in k or "_A" in k)), None)
                    up_k = next((k for k in sd.keys() if core in self.clean_k(k) and ("up" in k or "_B" in k)), None)
                    if down_k and up_k:
                        # Applying Noise Dampener for Wan 2.2 Low variant
                        calc_weight = styles[j]['weight'] * global_mult * noise_dampener
                        d = gpu_mm(sd[up_k].to(torch.float32), sd[down_k].to(torch.float32)) * calc_weight
                        mask = (torch.rand_like(d) < density).float()
                        layer_deltas.append((d * mask) / density)

                if len(layer_deltas) > 1:
                    stacked = torch.stack(layer_deltas)
                    signs = torch.sign(stacked)
                    dom_sign = torch.sign(signs.sum(dim=0))
                    if not torch.all(signs[0] == signs): conflict_log.append(core)
                    aligned = (stacked * (signs == dom_sign).float()).sum(dim=0) / (stacked != 0).sum(dim=0).clamp(min=1)
                    self.apply_delta(target_key, aligned)
                elif len(layer_deltas) == 1:
                    self.apply_delta(target_key, layer_deltas[0])
            del style_dicts
        gc.collect()
        return conflict_log

    def clean_k(self, k):
        return re.sub(r'.lora_(down|up|A|B|default).*', '', k).replace("__", ".").replace("lora_unet.", "").replace("blocks_", "blocks.")

    def is_skeletal(self, key):
        # Wan 2.2 maintains the 40-block structure; early blocks still handle structure
        match = re.search(r'blocks\.(\d+)\.', key)
        return match and int(match.group(1)) <= 8

    def apply_delta(self, target_key, delta):
        if self.base_dict[target_key].shape != delta.shape: delta = delta.t()
        dtype = self.base_dict[target_key].dtype
        self.base_dict[target_key] = self.base_dict[target_key].to(torch.float32).add_(delta).to(dtype)

    def save(self):
        final_path = f"{self.paths['output_prefix']}_FINAL.safetensors"
        save_file(self.base_dict, final_path)
        return final_path

def log_heatmap(self, conflicts, pass_name):
    early, mid, late = 0, 0, 0
    for c in conflicts:
        match = re.search(r'blocks\.(\d+)\.', c)
        if match:
            blk = int(match.group(1))
            if blk <= 8: early += 1
            elif 9 <= blk <= 30: mid += 1
            else: late += 1

    # Printing to Konsole with ANSI colors
    print(f"\n\033[1m📊 PASS REPORT: {pass_name}\033[0m")
    print(f"  \033[94m[Structure] Blocks 0-8:   {early:3} clashes\033[0m")
    print(f"  \033[92m[Motion]    Blocks 9-30:  {mid:3} clashes\033[0m")
    print(f"  \033[93m[Aesthetic] Blocks 31-40: {late:3} clashes\033[0m")

    if early > 25:
        print("  \033[91m⚠️  CRITICAL: Skeletal instability detected!\033[0m")
    print("-" * 40)
