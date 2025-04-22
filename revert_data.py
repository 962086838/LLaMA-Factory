import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def convert_bin_to_json(bin_file, idx_file, output_json, tokenizer_path):
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 加载索引文件
    doc_indices = np.fromfile(idx_file, dtype=np.int64)
    print(doc_indices.shape)
    assert 1==0

    # 加载二进制数据
    with open(bin_file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)

    print(data[0:100])

    print("len(doc_indices)", len(doc_indices))

    documents = []
    # 遍历文档索引
    for i in tqdm(range(len(doc_indices) - 1)):
        start_idx = doc_indices[i]
        end_idx = doc_indices[i + 1]

        # 提取文档的token IDs
        token_ids = data[start_idx:end_idx].tolist()
        print(start_idx, end_idx, len(token_ids))

        # 使用tokenizer解码
        # try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        documents.append({"text": text})
        print(text)
        assert 1==0
        # except Exception as e:
        #     print(f"Error decoding document {i}: {str(e)}")
        #     continue

    # 写入JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-file", type=str, required=True, help="Path to the .bin file")
    parser.add_argument("--idx-file", type=str, required=True, help="Path to the .idx file")
    parser.add_argument("--output-json", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--tokenizer-path", type=str, default="/gemini/space/thu/TeleChat-1B-moe",
                        help="Path to the tokenizer directory")
    args = parser.parse_args()

    convert_bin_to_json(args.bin_file, args.idx_file, args.output_json, args.tokenizer_path)
    print(f"Conversion complete. JSON file saved to {args.output_json}")

    """
    python revert_data.py \
      --bin-file /gemini/space/thu/receieve_wangzihan/data_part1_aa_text_document.bin \
      --idx-file /gemini/space/thu/receieve_wangzihan/data_part1_aa_text_document.idx \
      --output-json output.json \
      --tokenizer-path /gemini/space/thu/TeleChat-1B-moe
    """