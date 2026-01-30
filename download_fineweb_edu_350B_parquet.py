import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置参数 ---
# 基础 URL 模板
BASE_URL = "https://modelscope.cn/datasets/AI-ModelScope/fineweb-edu/resolve/master/sample/350BT/"
# 保存目录
SAVE_DIR = "fineweb_data_350B"
# 最大并发线程数 (根据带宽调整，建议 4-10)
MAX_WORKERS = 10

# 定义需要下载的文件范围
# 格式: (前缀, 结束编号)
# 例如 000_00000 到 000_00029
file_ranges = [
    ("000", 30), ("001", 30), ("002", 30), ("003", 30), ("004", 30),
    ("005", 30), ("006", 30), ("007", 30), ("008", 30), ("009", 30),
    ("010", 30), ("011", 30), ("012", 30), ("013", 30), ("014", 30),
    ("015", 20), # 015 只有 20 个文件 (00-19)
    ("016", 2)   # 016 只有 2 个文件 (00-01)
]

def get_file_list():
    """根据规律生成完整的文件名列表"""
    urls = []
    for prefix, count in file_ranges:
        for i in range(count):
            filename = f"{prefix}_{i:05d}.parquet"
            urls.append(filename)
    return urls

def download_file(filename):
    """单个文件下载函数，支持跳过已存在文件"""
    url = BASE_URL + filename
    filepath = os.path.join(SAVE_DIR, filename)
    
    # 如果文件已存在且大小不为0，跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return f"Skipped: {filename}"

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunk
                if chunk:
                    f.write(chunk)
        return f"Done: {filename}"
    except Exception as e:
        return f"Error: {filename} - {str(e)}"

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    files = get_file_list()
    print(f"准备下载 {len(files)} 个文件到 '{SAVE_DIR}' 目录...")

    # 使用进度条
    with tqdm(total=len(files), unit='file') as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_file = {executor.submit(download_file, f): f for f in files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                pbar.update(1)
                # 如果报错则打印，正常完成则静默
                if "Error" in result:
                    print(f"\n{result}")

if __name__ == "__main__":
    main()