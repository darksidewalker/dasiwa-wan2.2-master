import torch
import os, gc, re, json
from safetensors.torch import load_file, save_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gpu_mm(B, A):
    """Offloads matrix math to GPU using half-precision for Wan 2.2 speed."""
    with torch.no_grad():
        # Ensure input is float32 for the math, then autocast handles the speed
        B_gpu = B.to(DEVICE, dtype=torch.float32)
        A_gpu = A.to(DEVICE, dtype=torch.float32)
        res = torch.mm(B_gpu, A_gpu).to("cpu")
        del B_gpu, A_gpu
        torch.cuda.empty_cache()
        return res

class ActionMasterEngine:
    def __init__(self, recipe_path):
        with open(recipe_path, 'r') as f:
            content = f.read()
            clean_content = re.sub(r'//.*', '', content)
            self.recipe = json.loads(clean_content)

        self.paths = self.recipe['paths']
        print(f"📂 Loading Base Model: {self.paths['base_model']}...")
        self.base_dict = load_file(self.paths['base_model'])
        self.base_keys = list(self.base_dict.keys())

        # Wan 2.2 Integrity Check
        self.is_high_res = "high" in self.paths['base_model'].lower()
        print(f"🧬 Wan 2.2 Mode: {'HIGH' if self.is_high_res else 'LOW'} Noise Variant detected.")

    def clean_k(self, k):
            """Standardizes Wan 2.2 keys to match base model architecture."""
            # 1. Strip LoRA suffixes
            k = re.sub(r'\.(lora_up|lora_down|alpha|lora_A|lora_B|default).*', '', k)
            
            # 2. Normalize common training prefixes
            k = k.replace("lora_unet_", "")
            k = k.replace("diffusion_model.", "")
            k = k.replace("transformer.", "")
            
            # 3. CRITICAL: Convert underscores to dots ONLY for block structures
            # This turns 'blocks_4_cross_attn_o' -> 'blocks.4.cross_attn.o'
            if "blocks_" in k:
                k = k.replace("blocks_", "blocks.")
                # Replace remaining underscores with dots
                k = k.replace("_", ".")
                
            # 4. Clean up any double dots created by the replacement
            k = k.replace("__", ".").strip(".")
            
            return k

    def is_skeletal(self, key):
        """Identifies if a key belongs to the structural core (Blocks 0-8)."""
        match = re.search(r'blocks?\.(\d+)\.', key)
        return match and int(match.group(1)) <= 8

    def process_pass(self, step, global_mult):
        conflict_log = []
        use_shield = step.get('block_shield', False)
        noise_dampener = 1.0 if self.is_high_res else 0.85
        density = step.get('density', 0.6)
        
        styles = step.get('styles', [])
        if not styles:
            return []

        # Load all LoRAs for this pass
        style_dicts = []
        for s in styles:
            l_path = os.path.join(self.paths['lora_dir'], s['file'])
            if os.path.exists(l_path):
                style_dicts.append(load_file(l_path))
            else:
                print(f"⚠️ LoRA not found: {s['file']}")

        # Aggregate all unique keys across these LoRAs
        all_cores = set().union(*(set(self.clean_k(k) for k in sd.keys()) for sd in style_dicts))

        for core in all_cores:
            # Try 1: Direct Match (most efficient)
            target_key = next((bk for bk in self.base_keys if core in bk and bk.endswith(".weight")), None)
            
            # Try 2: Deep Match (if core is a partial path)
            if not target_key:
                # Remove prefixes from base keys to see if they match core
                target_key = next((bk for bk in self.base_keys if bk.endswith(f"{core}.weight")), None)

            if not target_key:
                # Still logging missed keys so we can refine further if needed
                if "blocks" in core:
                    print(f"❓ Still missed: {core}")
                continue

            is_skeletal = self.is_skeletal(target_key)
            layer_deltas = []

            for j, sd in enumerate(style_dicts):
                # Apply shielding logic
                if use_shield and is_skeletal: continue
                if is_skeletal and not styles[j].get("modify_skeleton", False): continue

                # Locate the Up and Down pairs
                down_k = next((k for k in sd.keys() if core in self.clean_k(k) and ("down" in k or "_A" in k)), None)
                up_k = next((k for k in sd.keys() if core in self.clean_k(k) and ("up" in k or "_B" in k)), None)

                if down_k and up_k:
                    calc_weight = styles[j]['weight'] * global_mult * noise_dampener
                    
                    # Compute weight delta: (Up * Down) * weight
                    d = gpu_mm(sd[up_k], sd[down_k]) * calc_weight
                    
                    # Apply density mask
                    if density < 1.0:
                        mask = (torch.rand_like(d) < density).float()
                        d = (d * mask) / density
                    
                    layer_deltas.append(d)

            # Resolve conflicts if multiple LoRAs hit the same key
            if len(layer_deltas) > 1:
                stacked = torch.stack(layer_deltas)
                signs = torch.sign(stacked)
                sum_signs = signs.sum(dim=0)
                dom_sign = torch.sign(sum_signs)
                
                # Check for sign conflict (clash)
                if not torch.all(signs[0] == signs):
                    conflict_log.append(target_key)
                
                # Align weights to the dominant sign to prevent "gray sludge" cancellation
                aligned = (stacked * (signs == dom_sign).float()).sum(dim=0) / (stacked != 0).sum(dim=0).clamp(min=1)
                self.apply_delta(target_key, aligned)
                
            elif len(layer_deltas) == 1:
                self.apply_delta(target_key, layer_deltas[0])

        # Cleanup RAM
        del style_dicts
        gc.collect()
        torch.cuda.empty_cache()
        return conflict_log

    def apply_delta(self, target_key, delta):
        """Safely applies the calculated delta to the base model weights."""
        if self.base_dict[target_key].shape != delta.shape:
            delta = delta.t()
            
        dtype = self.base_dict[target_key].dtype
        # Perform addition in float32 for precision, then revert to original dtype
        self.base_dict[target_key] = (self.base_dict[target_key].to(torch.float32) + delta).to(dtype)

    def get_heatmap_stats(self, conflicts):
        early, mid, late = 0, 0, 0
        for c in conflicts:
            match = re.search(r'blocks?\.(\d+)\.', c)
            if match:
                blk = int(match.group(1))
                if blk <= 8: early += 1
                elif 9 <= blk <= 30: mid += 1
                else: late += 1

        report = [
            f"📊 PASS REPORT",
            f"  🔹 [Structure] Blocks 0-8:   {early:3} clashes",
            f"  🟢 [Motion]    Blocks 9-30:  {mid:3} clashes",
            f"  🟡 [Aesthetic] Blocks 31-40: {late:3} clashes"
        ]
        if early > 25:
            report.append("  ⚠️ CRITICAL: Skeletal instability detected!")
        return "\n".join(report)

    def save(self):
        final_path = f"{self.paths['output_prefix']}_FINAL.safetensors"
        print(f"💾 Saving merged model to {final_path}...")
        save_file(self.base_dict, final_path)
        return final_path