from datasets import load_from_disk
from transformers import AutoTokenizer


# 假设这些文件在同一个目录下（比如 './my_dataset'）
dataset = load_from_disk('/gemini/space/thu/hehaowei/LLaMA-Factory/saves/TeleChat-1B/dataset/sft/train')  # 传入包含这些文件的目录路径
# dataset = load_from_disk('/gemini/space/thu/receieve_wangzihan/Megatron-LM-core_r0.6.0_dropout_HWMoE/converted_hf_dataset')  # 传入包含这些文件的目录路径

tokenizer = AutoTokenizer.from_pretrained('/gemini/space/thu/TeleChat-1B', trust_remote_code=True)

print(f"dataset size: {len(dataset)}")

print(dataset[0])  # 查看第一条数据c
print(dataset[0].keys())  # 查看第一条数据c
print(type(dataset[0]["input_ids"][0]), type(dataset[0]["attention_mask"][0]))
# print(len(dataset[0]["input_ids"]))

# print(tokenizer.decode(dataset[-1]["input_ids"]))