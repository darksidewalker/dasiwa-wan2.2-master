# utils/scanner_5d.py
import os
import torch
import gguf
from safetensors.torch import load_file

def scan_5d_tensors(file_path):
    if not os.path.exists(file_path):
        return f"❌ Error: File not found at {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".safetensors":
            return _scan_safetensors(file_path)
        elif ext == ".gguf":
            return _scan_gguf(file_path)
        else:
            return f"❌ Unsupported file extension: {ext}"
    except Exception as e:
        return f"🔥 Scanning Error: {str(e)}"

def _scan_safetensors(path):
    state_dict = load_file(path)
    output = [f"🔍 [Safetensors] Scanning 5D+ Tensors: {os.path.basename(path)}", "-" * 50]
    found_any = False
    total_bytes = 0

    for key, tensor in state_dict.items():
        if len(tensor.shape) > 4:
            size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
            total_bytes += (tensor.numel() * tensor.element_size())
            output.append(f"🎯 {key} | {list(tensor.shape)} | {tensor.dtype} | {size_mb:.2f} MB")
            found_any = True

    if not found_any: output.append("✅ No 5D tensors found.")
    else: output.append(f"\n📊 TOTAL 5D STORAGE: {total_bytes / (1024 * 1024):.2f} MB")
    return "\n".join(output)

def _scan_gguf(path):
    reader = gguf.GGUFReader(path)
    output = [f"🔍 [GGUF] Scanning Tensors: {os.path.basename(path)}", "-" * 50]
    found_any = False
    
    for tensor in reader.tensors:
        shape = tensor.shape.tolist()
        if len(shape) > 4:
            output.append(f"🎯 {tensor.name} | Shape: {shape} | Type: {tensor.tensor_type.name}")
            found_any = True

    if not found_any:
        output.append("✅ No >4D tensors found in this GGUF.")
        output.append("💡 Note: GGUF usually flattens 5D tensors into 2D during conversion.")
    
    return "\n".join(output)