import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a pretrained model")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2",
                        help="Path to pretrained model or model name from huggingface.co")
    parser.add_argument("--expert_limit", type=int, default=8,
                        help="Maximum number of experts to use per layer")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Number of tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(device)
    # print(model.config)
    # assert 1==0
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model.eval()

    with torch.no_grad():
        input_text = "Long long ago, there "
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        # input_ids = torch.cat([torch.Tensor([1]).unsqueeze(0).to(input_ids), input_ids], dim=1)
        # print(input_ids)
        # assert 1==0
        # input_ids = torch.tensor().to(device).to(torch.long).unsqueeze(0)

        print(f"Start generating {args.max_new_tokens} tokens...")

        for _ in range(args.max_new_tokens):
            outputs = model.single_forward_with_expert_limit(
                input_ids,
                expert_limit=args.expert_limit,
                output_router_logits=False,
                return_dict=True
            )

            # 获取最后一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]

            # 取概率最大的 token (贪婪解码)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # 或者使用 top-k 采样增加多样性（可选）
            # topk_indices = torch.topk(next_token_logits, k=50).indices
            # probs = torch.softmax(next_token_logits[:, topk_indices], dim=-1)
            # next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print("\n生成的回答：")
        print(generated_text)

if __name__ == "__main__":
    main()