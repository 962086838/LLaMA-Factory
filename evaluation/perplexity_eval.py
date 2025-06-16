import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate.test_utils.testing import get_backend
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perplexity for a model on a dataset")
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2",
                        help="Path to pretrained model or model name from huggingface.co")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name from the datasets library")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration name")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to use: train, validation or test")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Maximum sequence length of the model")
    parser.add_argument("--stride", type=int, default=8192,
                        help="Stride for sliding window over tokens")
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
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name})")
    data = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    # training_dataset = load_from_disk('/gemini/space/thu/receieve_wangzihan/Megatron-LM-core_r0.6.0_dropout_HWMoE/converted_dataset')

    # Tokenize entire dataset
    text = "\n\n".join(data["text"])
    print(text[0:100], text[-100:])
    encodings = tokenizer(text, return_tensors="pt").to(device)

    seq_len = encodings.input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    model.eval()
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, args.stride), total=(seq_len // args.stride) + 1):
            end_loc = min(begin_loc + args.max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            # input_ids = torch.Tensor(training_dataset[0]["input_ids"]).to(device).view(1, -1).long()
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # ignore padding labels

            outputs = model(input_ids, labels=target_ids, output_router_logits=False, return_dict=True)
            neg_log_likelihood = outputs.loss
            print("neg_log_likelihood", neg_log_likelihood, outputs.lm_loss, target_ids.size(0))

            # Count valid tokens
            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # because of internal label shift

            nll_sum += neg_log_likelihood.item() * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            # assert 1==0

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(torch.tensor(avg_nll))

    print(f"Average Negative Log-Likelihood: {avg_nll:.4f}")
    print(f"Perplexity: {ppl.item():.4f}")


if __name__ == "__main__":
    main()