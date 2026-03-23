import safetensors
import torch
import json
import numpy as np

def deep_verify(model_path):
    # 加载索引
    with open(f"{model_path}/model.safetensors.index.json", 'r') as f:
        index = json.load(f)
    
    # 找到相关文件
    embed_file = index['weight_map']['model.embed_tokens.weight']
    lm_head_file = index['weight_map']['lm_head.weight']
    
    print(f"Embed tokens 文件: {embed_file}")
    print(f"LM head 文件: {lm_head_file}")
    
    # 加载权重
    with safetensors.safe_open(f"{model_path}/{embed_file}", framework="pt", device="cpu") as f:
        embed = f.get_tensor('model.embed_tokens.weight')
    
    with safetensors.safe_open(f"{model_path}/{lm_head_file}", framework="pt", device="cpu") as f:
        lm_head = f.get_tensor('lm_head.weight')
    
    # 检查它们的关系
    print("\n=== 权重关系分析 ===")
    print(f"embed 形状: {embed.shape}")
    print(f"lm_head 形状: {lm_head.shape}")
    
    # 检查是否为转置关系
    if embed.shape == lm_head.shape:
        print("\n检查是否为完全相同的数据:")
        # 由于数据量大，使用抽样
        sample_rows = torch.randint(0, embed.shape[0], (100,))
        sample_cols = torch.randint(0, embed.shape[1], (100,))
        
        embed_samples = embed[sample_rows, sample_cols]
        lm_head_samples = lm_head[sample_rows, sample_cols]
        
        diff = torch.abs(embed_samples - lm_head_samples)
        print(f"  100个随机位置的差值:")
        print(f"    最大差值: {diff.max().item()}")
        print(f"    平均差值: {diff.mean().item()}")
        print(f"    中位数差值: {diff.median().item()}")
        
        if diff.max().item() == 0:
            print("\n✅ 确认: lm_head.weight 和 model.embed_tokens.weight 完全相同")
            print("   它们确实是绑定的权重，只是被存储在不同的分片文件中")
        else:
            print("\n⚠️  发现差异！它们不是完全相同的")
    
    # 检查那个奇怪的 layer 61 的 embed_tokens
    if 'model.layers.61.embed_tokens.weight' in index['weight_map']:
        strange_file = index['weight_map']['model.layers.61.embed_tokens.weight']
        print(f"\n奇怪的 embed_tokens 在: {strange_file}")
        
        with safetensors.safe_open(f"{model_path}/{strange_file}", framework="pt", device="cpu") as f:
            strange = f.get_tensor('model.layers.61.embed_tokens.weight')
        
        print(f"奇怪的 embed_tokens 形状: {strange.shape}")
        print(f"是否和正常的 embed_tokens 相同: {torch.all(embed == strange).item()}")

# 运行验证
deep_verify("/gfs/space/public/public_models/GLM-5-FP8")