import safetensors
import glob

# 获取所有 safetensors 文件
safetensor_files = glob.glob("/gfs/space/public/public_models/GLM-5-FP8/*.safetensors")
print(f"找到 {len(safetensor_files)} 个 safetensors 文件")

# 检查每个文件中的 tensor 名称
for file_path in sorted(safetensor_files):
    print(f"\n检查文件: {file_path}")
    
    # 只加载元数据，不加载实际权重
    with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        print(f"包含 {len(keys)} 个 tensors")
        
        # 查找我们关心的权重
        for key in keys:
            if "embed_tokens.weight" in key or "lm_head.weight" in key:
                print(f"  发现: {key}")
                # 获取 tensor 的元信息
                metadata = f.get_tensor(key)
                print(f"    形状: {metadata.shape}")
                print(f"    数据类型: {metadata.dtype}")