import os
import hashlib
from tqdm import tqdm


# 判断是否为需要仅比较大小的文件类型
def is_large_model_file(filename):
    return filename.endswith(('.safetensors', 'optim_states.pt', 'model_states.pt'))


def get_file_hash(filepath):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"无法读取文件 {filepath}: {e}")
        return None


def get_file_size(filepath):
    """获取文件大小（字节）"""
    try:
        return os.path.getsize(filepath)
    except Exception as e:
        print(f"无法获取文件大小 {filepath}: {e}")
        return None


def find_files(root_dir):
    """递归查找目录下所有文件及其相对路径"""
    files_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir)
            files_dict[rel_path] = full_path
    return files_dict


def compare_directories(path1, path2):
    """比较两个目录中的文件，并在 tqdm 中显示当前文件"""
    print(f"正在扫描目录：{path1}")
    files1 = find_files(path1)

    print(f"开始比较与目录：{path2}")

    pbar = tqdm(files1.items(), desc="Comparing", unit="file")

    for rel_path, file1 in pbar:
        display_name = os.path.basename(rel_path)
        pbar.set_postfix(file=rel_path)  # 显示当前文件名（最多30字符）

        file2 = os.path.join(path2, rel_path)

        if not os.path.exists(file2):
            print(f"\n[缺失] 路径 {file2} 不存在")
            continue

        filename = os.path.basename(rel_path)

        # 判断文件类型，选择比较方式
        if is_large_model_file(filename):
            size1 = get_file_size(file1)
            size2 = get_file_size(file2)
            if size1 != size2:
                print(f"\n[修改] 文件大小不一致: {file1} ({size1}B) vs {file2} ({size2}B)")
        else:
            hash1 = get_file_hash(file1)
            hash2 = get_file_hash(file2)
            if hash1 != hash2:
                print(f"\n[修改] 文件内容不一致: {file1} 与 {file2}")

    # # 查找 path2 中是否有额外的文件（即 path1 中没有的）
    # print("开始检查新增文件...")
    # files2 = find_files(path2)
    # added_files = set(files2.keys()) - set(files1.keys())
    # for rel_path in added_files:
    #     print(f"[新增] 在 {path2} 中新增了文件: {os.path.join(path2, rel_path)}")


if __name__ == "__main__":
    path1 = "/gemini/space/thu/hehaowei/LLaMA-Factory/saves/TeleChat-2B-A05B-hwmoe/"
    path2 = "/gemini-3/space/thu/hehaowei/LLaMA-Factory/saves/TeleChat-2B-A05B-hwmoe/"

    compare_directories(path1, path2)