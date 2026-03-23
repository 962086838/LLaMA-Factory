"""
调试脚本：检查预训练数据的 labels 构成 和 模型 logits 分布。

用法（在训练机上运行）：
    python debug_loss.py --model_path saves/glm5-small/pt/fineweb_edu_10b  # 已训练的 checkpoint
  或者
    python debug_loss.py --model_path GLM5-small --from_scratch            # 未训练，从零初始化

如果只想检查数据（不加载模型），加 --data_only：
    python debug_loss.py --data_only
"""

import argparse
import sys
import os

# 把 src 加入路径，方便导入 llamafactory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling


def check_data(tokenizer, data_collator, num_samples=5):
    """检查 labels 中 padding token 的分布情况"""
    print("=" * 70)
    print("【1】检查数据和 labels 构造")
    print("=" * 70)

    print(f"\ntokenizer.pad_token_id = {tokenizer.pad_token_id}")
    print(f"tokenizer.eos_token_id = {tokenizer.eos_token_id}")
    print(f"tokenizer.pad_token = {repr(tokenizer.pad_token)}")
    print(f"tokenizer.eos_token = {repr(tokenizer.eos_token)}")

    # 模拟一些文本数据
    sample_texts = [
        "Hello, this is a test sentence for debugging.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    # 模拟 pretrain pipeline: text + eos_token
    text_examples = [t + tokenizer.eos_token for t in sample_texts]
    tokenized = tokenizer(text_examples, add_special_tokens=False, padding=True, return_tensors="pt")

    print(f"\n--- Tokenized input_ids shape: {tokenized['input_ids'].shape} ---")

    for i in range(min(num_samples, len(sample_texts))):
        ids = tokenized["input_ids"][i]
        print(f"\nSample {i}:")
        print(f"  input_ids: {ids.tolist()}")
        print(f"  decoded:   {repr(tokenizer.decode(ids))}")

        # 统计 pad token
        pad_count = (ids == tokenizer.pad_token_id).sum().item()
        total = len(ids)
        print(f"  pad_token_id ({tokenizer.pad_token_id}) 出现次数: {pad_count}/{total} ({pad_count / total * 100:.1f}%)")

        # 统计 eos token(s)
        if isinstance(tokenizer.eos_token_id, list):
            for eos_id in tokenizer.eos_token_id:
                eos_count = (ids == eos_id).sum().item()
                print(f"  eos_token_id ({eos_id}) 出现次数: {eos_count}")
        else:
            eos_count = (ids == tokenizer.eos_token_id).sum().item()
            print(f"  eos_token_id ({tokenizer.eos_token_id}) 出现次数: {eos_count}")

    # 用 data_collator 处理
    batch_input = [{"input_ids": tokenized["input_ids"][i]} for i in range(len(sample_texts))]
    batch = data_collator(batch_input)

    print(f"\n--- DataCollatorForLanguageModeling 输出 ---")
    print(f"  labels shape: {batch['labels'].shape}")

    for i in range(min(num_samples, len(sample_texts))):
        labels = batch["labels"][i]
        ignore_count = (labels == -100).sum().item()
        total = len(labels)
        pad_in_labels = (labels == tokenizer.pad_token_id).sum().item()

        print(f"\nSample {i} labels:")
        print(f"  labels:           {labels.tolist()}")
        print(f"  -100 (IGNORE) 数: {ignore_count}/{total} ({ignore_count / total * 100:.1f}%)")
        print(f"  pad_token_id 在 labels 中出现: {pad_in_labels} 次")

        # ⚠️ 关键检查：如果 pad_token_id 大量出现在 labels 中（非 -100），则 loss 会虚低
        if pad_in_labels > 0:
            print(f"  ⚠️ 警告：pad_token_id 在 labels 中出现了 {pad_in_labels} 次！")
            print(f"     如果模型学会预测 pad，loss 会虚低！")


def check_logits(model, tokenizer, device="cuda"):
    """检查已训练模型的 logits 分布"""
    print("\n" + "=" * 70)
    print("【2】检查模型 logits 分布")
    print("=" * 70)

    model.eval()

    # 一句正常文本
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # 最后一个 token 的 logits
    last_logits = logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)

    print(f"\n输入: {repr(test_text)}")
    print(f"logits shape: {logits.shape}")

    # Top-10 预测
    top_k = 10
    top_probs, top_ids = torch.topk(probs, top_k)
    print(f"\nTop-{top_k} 预测（最后一个位置）：")
    for i in range(top_k):
        token_id = top_ids[i].item()
        prob = top_probs[i].item()
        token_str = tokenizer.decode([token_id])
        print(f"  #{i+1}: token_id={token_id}, prob={prob:.6f}, token={repr(token_str)}")

    # 检查 pad_token 和 eos_token 的概率
    print(f"\n--- 特殊 token 概率 ---")
    if tokenizer.pad_token_id is not None:
        pad_prob = probs[tokenizer.pad_token_id].item()
        print(f"  pad_token_id ({tokenizer.pad_token_id}) 概率: {pad_prob:.8f}")

    if isinstance(tokenizer.eos_token_id, list):
        for eos_id in tokenizer.eos_token_id:
            eos_prob = probs[eos_id].item()
            print(f"  eos_token_id ({eos_id}) 概率: {eos_prob:.8f}")
    elif tokenizer.eos_token_id is not None:
        eos_prob = probs[tokenizer.eos_token_id].item()
        print(f"  eos_token_id ({tokenizer.eos_token_id}) 概率: {eos_prob:.8f}")

    # 全序列每个位置的 max prob 和 entropy
    all_probs = torch.softmax(logits[0], dim=-1)  # [seq_len, vocab_size]
    max_probs = all_probs.max(dim=-1).values
    entropy = -(all_probs * (all_probs + 1e-10).log()).sum(dim=-1)

    print(f"\n--- 全序列统计 ---")
    print(f"  每个位置的 max prob: {[f'{p:.4f}' for p in max_probs.tolist()]}")
    print(f"  每个位置的 entropy:  {[f'{e:.2f}' for e in entropy.tolist()]}")
    print(f"  平均 max prob: {max_probs.mean().item():.4f}")
    print(f"  平均 entropy:  {entropy.mean().item():.2f}")

    if max_probs.mean().item() > 0.9:
        print(f"\n  🚨 模型对大部分预测都极度自信 (avg max_prob > 0.9)，可能过拟合！")

    # 检查 pad_token 在所有位置上是否有异常高概率
    if tokenizer.pad_token_id is not None:
        pad_probs_all = all_probs[:, tokenizer.pad_token_id]
        print(f"\n--- pad_token 在各位置的概率 ---")
        print(f"  {[f'{p:.6f}' for p in pad_probs_all.tolist()]}")
        if pad_probs_all.mean().item() > 0.01:
            print(f"  ⚠️ pad_token 平均概率 {pad_probs_all.mean().item():.4f} 异常偏高！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GLM5-small",
                        help="模型路径（checkpoint 或原始模型目录）")
    parser.add_argument("--from_scratch", action="store_true",
                        help="从零初始化模型（而非加载已训练权重）")
    parser.add_argument("--data_only", action="store_true",
                        help="只检查数据，不加载模型")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    args = parser.parse_args()

    print(f"加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    check_data(tokenizer, data_collator)

    if not args.data_only:
        print(f"\n加载模型: {args.model_path}")
        if args.from_scratch:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        device = args.device if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        check_logits(model, tokenizer, device=device)

    print("\n✅ 调试完成")


if __name__ == "__main__":
    main()
