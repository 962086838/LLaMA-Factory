import os
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

# 要检查的目录
data_dir = "/gemini-3/space/thu/hehaowei/modelscope_cache/datasets/fineweb_hhw_transformed/sample/100BT"
# data_dir = "/gemini-3/space/thu/hehaowei/modelscope_cache/datasets/ultrafineweb_hhw_transformed/data/ultrafineweb_en"
# data_dir = "/gemini-3/space/thu/hehaowei/modelscope_cache/datasets/ultrafineweb_hhw_transformed/data/ultrafineweb_zh"

# 获取所有 .parquet 文件路径
parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
file_paths = [os.path.join(data_dir, f) for f in parquet_files]

# 单个文件的验证函数
def check_parquet_file(file_path):
    try:
        table = pq.read_table(file_path)
        return f"{os.path.basename(file_path)}: 读取成功，行数={table.num_rows}"
    except Exception as e:
        return f"{os.path.basename(file_path)}: 读取失败 - {e}"

# 使用 32 个进程并行处理
def parallel_check(file_paths, max_workers=32):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_parquet_file, fp) for fp in file_paths]
        for future in as_completed(futures):
            if "失败" in future.result():
                print(future.result())

# 执行
parallel_check(file_paths)