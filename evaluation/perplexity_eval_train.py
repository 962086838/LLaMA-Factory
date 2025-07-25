import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate.test_utils.testing import get_backend
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perplexity for a model on a dataset")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2",
                        help="Path to pretrained model or model name from huggingface.co")
    parser.add_argument("--expert_limit", type=int, default=8,
                        help="")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get device
    device, _, _ = get_backend()

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(device)
    model = model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load dataset
    # print(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name})")
    # data = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    training_dataset = load_from_disk('/gemini/space/thu/hehaowei/LLaMA-Factory/data/test_data_ad/')
    print("dataset length", len(training_dataset))

    # Tokenize entire dataset
    # text = "\n\n".join(data["text"])
    # encodings = tokenizer(text, return_tensors="pt").to(device)

    # seq_len = encodings.input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    # prev_end_loc = 0

    model.eval()


    with torch.no_grad():
        # 初始化一个12层×16专家的统计矩阵
        expert_activation_counts = torch.zeros(12, 16, device=device)
        for i in tqdm(range(0, 10000)):

            input_ids = torch.Tensor(training_dataset[i]["input_ids"]).to(device).view(1, -1).long()
            target_ids = input_ids.clone()
            # target_ids[:, :-trg_len] = -100  # ignore padding labels

            outputs = model.single_forward_with_expert_limit(input_ids,
                                                             labels=target_ids,
                                                             output_router_logits=True,
                                                             return_dict=True,
                                                             expert_limit=args.expert_limit,
                                                             output_router_bucket_status=True,
                                                             )
            neg_log_likelihood = outputs.lm_loss
            router_bucket_status = outputs.router_bucket_status
            # print("neg_log_likelihood", neg_log_likelihood, outputs.lm_loss)

            # Count valid tokens
            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # because of internal label shift

            nll_sum += neg_log_likelihood.item() * num_loss_tokens
            n_tokens += num_loss_tokens

            # 统计专家使用情况
            if router_bucket_status is not None:
                # 更新专家激活统计
                for layer_idx, layer_status in enumerate(router_bucket_status):
                    expert_activation_counts[layer_idx] += layer_status['expert_counts']

    # 计算平均激活频率（每个token平均激活次数）
    # 因为每个token会激活2个专家，所以总token数为n_tokens
    expert_activation_freq = expert_activation_counts / (n_tokens * 2)
    # 将结果转为numpy数组
    expert_activation_freq_np = expert_activation_freq.cpu().numpy()

    # 确保输出目录存在
    os.makedirs("eval_expert_activation_freq", exist_ok=True)
    base_filename = f"eval_expert_activation_freq/eval_expert_activation_freq{'-'.join(args.model_name_or_path.split('/'))}"

    # 打印统计结果
    print(f"\nExpert Activation Frequency (12 layers × {args.expert_limit} experts):")
    print(expert_activation_freq_np)

    # 保存为.npy文件（最简洁的二进制格式）
    np.save(f"{base_filename}.npy", expert_activation_freq_np)
    print(f"\nSaved activation frequency matrix to {base_filename}.npy")

    # 可视化热力图
    plt.figure(figsize=(16, 8))
    plt.imshow(expert_activation_freq.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Activation Frequency')
    plt.xlabel('Expert Index')
    plt.ylabel('Layer Index')
    plt.title('Expert Activation Frequency Across Layers')
    plt.xticks(range(args.expert_limit))
    plt.yticks(range(12))
    os.makedirs("eval_expert_activation_freq", exist_ok=True)
    plt.savefig(f"{base_filename}.png")

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(torch.tensor(avg_nll))

    print(f"Average Negative Log-Likelihood: {avg_nll:.4f}")
    print(f"Perplexity: {ppl.item():.4f}")


if __name__ == "__main__":
    main()