from megatron.core.datasets.indexed_dataset import IndexedDataset
from datasets import Dataset, Features, Value, Array2D, Sequence
import numpy as np
import os
import pyarrow as pa
from tqdm import tqdm
import argparse
import math
from multiprocessing import Process

os.environ["HF_DATASETS_OFFLINE"] = "1"  # 完全离线模式
os.environ["DISABLE_WEBSITE_CHECK"] = "true"  # 禁用网站检查

def process_chunk(megatron_paths, output_dir, chunk_idx, total_chunks, batch_size=1000):
    """处理数据的一个分片"""
    features = Features({
        'input_ids': Sequence(Value('int32')),
        'attention_mask': Sequence(Value('int32'))
    })

    # 每个进程有自己的临时目录
    temp_dir = os.path.join(output_dir, f"temp_{chunk_idx}")
    os.makedirs(temp_dir, exist_ok=True)

    all_samples = []
    batch_count = 0

    for data_path in megatron_paths:
        dataset = IndexedDataset(data_path)
        dataset_length = len(dataset)

        # 计算当前进程应该处理的范围
        chunk_size = math.ceil(dataset_length / total_chunks)
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, dataset_length)

        print(f"Process {chunk_idx} processing {data_path} documents {start}-{end}...")

        for i in tqdm(range(start, end), position=chunk_idx):
            tokens = dataset[i]
            input_ids = np.array(tokens, dtype=np.int32)
            attention_mask = np.ones_like(input_ids, dtype=np.int32)

            all_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            if len(all_samples) >= batch_size:
                save_batch(all_samples, temp_dir, batch_count, features)
                batch_count += 1
                all_samples = []

    if all_samples:
        save_batch(all_samples, temp_dir, batch_count, features)


def save_batch(samples, temp_dir, batch_num, features):
    batch_dataset = Dataset.from_list(samples, features=features)
    batch_dataset.to_parquet(os.path.join(temp_dir, f"batch_{batch_num}.parquet"))


def merge_results(output_dir, total_chunks):
    """合并所有分片的结果"""
    temp_dirs = [os.path.join(output_dir, f"temp_{i}") for i in range(total_chunks)]

    all_parquet_files = []
    for temp_dir in temp_dirs:
        all_parquet_files.extend([
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith("batch_") and f.endswith(".parquet")
        ])

    # 设置自定义缓存路径
    cache_dir = "/gemini/space/thu/receieve_wangzihan/Megatron-LM-core_r0.6.0_dropout_HWMoE/cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 添加进度条和优化加载参数
    print(f"开始合并 {len(all_parquet_files)} 个parquet文件...")

    # 使用更高效的加载方式
    final_dataset = Dataset.from_parquet(
        all_parquet_files,
        cache_dir=cache_dir,
        keep_in_memory=False,  # 对于大数据集设为False
        num_proc=32,  # 使用多进程加载
    )

    print("合并完成，正在保存最终数据集...")
    final_dataset.save_to_disk(
        output_dir,
        max_shard_size="10GB",  # 控制每个分片的最大大小
        num_proc=16,  # 使用多进程加速（根据CPU核心数调整）
        # writer_batch_size=10000  # 每个进程的批量写入大小
    )
    print(f"最终数据集已保存到 {output_dir}")

    # 清理临时文件
    print("清理临时文件...")
    for temp_dir in temp_dirs:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    print("所有临时文件已清理")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-paths", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 启动多个工作进程
    # processes = []
    # for i in range(args.num_workers):
    #     p = Process(target=process_chunk, args=(
    #         args.megatron_paths,
    #         args.output_dir,
    #         i,
    #         args.num_workers
    #     ))
    #     p.start()
    #     processes.append(p)
    #
    # # 等待所有进程完成
    # for p in processes:
    #     p.join()

    # 合并结果
    merge_results(args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()

    """
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    python convert_megatron_dataset_hf_datasets.py --megatron-paths /gemini/space/thu/hehaowei/LLaMA-Factory/data/data_2000_4000_shard0_ad_text_document --output-dir /gemini/space/thu//hehaowei/LLaMA-Factory/data/tmp
    """