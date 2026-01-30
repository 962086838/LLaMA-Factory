import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置参数 ---
BASE_URL = "https://modelscope.cn/datasets/AI-ModelScope/fineweb-edu/resolve/master/sample/350BT/"
SAVE_DIR = "fineweb_data_350B"
MAX_WORKERS = 3  # 降低并发数以减少超时概率

# 指定需要重试的文件列表
target_files = [
    "000_00016.parquet",
    "002_00002.parquet",
    "006_00010.parquet"
]

def download_file(filename):
    """单个文件下载函数"""
    url = BASE_URL + filename
    filepath = os.path.join(SAVE_DIR, filename)
    
    # 如果文件已存在且大小不为0，可以选择删除重下或者跳过
    # 这里为了确保修复，我们打印一下并覆盖（或者你可以手动删除后再运行）
    if os.path.exists(filepath):
        print(f"Warning: {filename} exists, overwriting...")
    
    try:
        print(f"Start downloading: {filename} from {url}")
        # 增加超时时间到 60 秒
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): 
                if chunk:
                    f.write(chunk)
        return f"Successfully downloaded: {filename}"
    except Exception as e:
        return f"Error downloading {filename}: {str(e)}"

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"准备重试下载 {len(target_files)} 个文件到 '{SAVE_DIR}'...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(download_file, f): f for f in target_files}
        
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main()
