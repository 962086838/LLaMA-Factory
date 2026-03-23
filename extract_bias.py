"""
从 safetensor 文件中提取并打印所有 e_score_correction_bias 参数
"""
import glob
import os
from safetensors import safe_open

checkpoint_dir = "/gemini-1/space/space/private/liuxz/telechat3-saved/105B_sft1_1223_step7000/"
pattern = os.path.join(checkpoint_dir, "*.safetensors")
files = sorted(glob.glob(pattern))

print(f"Found {len(files)} safetensor files in {checkpoint_dir}\n")

for filepath in files:
    filename = os.path.basename(filepath)
    print(filepath)
    try:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "e_score_correction_bias" in key:
                    tensor = f.get_tensor(key)
                    print(f"[{filename}] {key}")
                    print(f"  shape: {tensor.shape}  dtype: {tensor.dtype}")
                    print(f"  min: {tensor.min().item():.6f}  max: {tensor.max().item():.6f}  mean: {tensor.mean().item():.6f}")
                    print(f"  values: {tensor}")
                    print()
    except Exception as e:
        print(f"  ⚠️ SKIP: {e}\n")
