import os
import json
import shutil
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 源目录和目标目录
source_dir = '/gemini-3/space/thu/hehaowei/modelscope_cache/datasets/downloads'
target_base_dir = '/gemini-3/space/thu/hehaowei/modelscope_cache/datasets/ultrafineweb_hhw_transformed'

# 单个文件的处理函数
def process_file(filename):
    file_path = os.path.join(source_dir, filename)

    if not filename.endswith('.json'):
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            url = data.get('url', '')

        if not url:
            print(f"JSON 文件中没有 url 字段: {filename}")
            return

        # 提取 FilePath 参数
        query = urllib.parse.urlparse(url).query
        params = urllib.parse.parse_qs(query)
        file_path_in_url = params.get('FilePath', [''])[0]

        if not file_path_in_url:
            print(f"URL 中没有 FilePath 参数: {url}")
            return

        decoded_file_path = urllib.parse.unquote(file_path_in_url)

        # 原始数据文件路径
        data_file = file_path.replace('.json', '')

        if not os.path.exists(data_file):
            print(f"找不到对应的数据文件: {data_file}")
            return

        # 构建目标路径
        target_file_path = os.path.join(target_base_dir, decoded_file_path)
        target_dir = os.path.dirname(target_file_path)
        os.makedirs(target_dir, exist_ok=True)

        # 拷贝文件
        shutil.copy2(data_file, target_file_path)
        print(f"已复制: {data_file} -> {target_file_path}")

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")

# 主函数
def main():
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_file, filename) for filename in json_files]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"线程执行出错: {e}")

if __name__ == '__main__':
    main()