import os
import glob
import shutil

# ================= 配置区域 =================

# 1. 原始路径（包含 arrow 和 json 的那个杂乱文件夹）
# 根据你的报错日志，推测路径如下，请务必核实：
source_dir = "/gfs/space/private/hehaowei/LLaMA-Factory/data/fineweb_edu_10b/AI-ModelScope___fineweb-edu/sample-10BT-fb3652c99ebd84a2/0.0.0/master"

# 2. 目标路径（你希望创建的新文件夹，只需包含 arrow）
# 建议放在 LLaMA-Factory/data 下，方便管理
target_dir = "/gfs/space/private/hehaowei/LLaMA-Factory/data/fineweb_edu_10b_clean"

# ===========================================

def create_symlinks():
    # 1. 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 错误：源目录不存在 -> {source_dir}")
        return

    # 2. 创建目标目录（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"✅ 已创建目标目录 -> {target_dir}")
    else:
        print(f"ℹ️ 目标目录已存在 -> {target_dir}")

    # 3. 查找所有的 .arrow 文件
    # 使用 glob 匹配所有 arrow 文件
    arrow_files = glob.glob(os.path.join(source_dir, "*.arrow"))

    if not arrow_files:
        print("❌ 警告：在源目录中没有找到任何 .arrow 文件！")
        return

    print(f"🔎 发现 {len(arrow_files)} 个 arrow 文件，准备建立软连接...")

    count = 0
    for src_file in arrow_files:
        filename = os.path.basename(src_file)
        dst_file = os.path.join(target_dir, filename)

        # 如果目标文件/链接已经存在，先跳过或删除（这里选择跳过提示）
        if os.path.exists(dst_file) or os.path.islink(dst_file):
            # print(f"  [跳过] 已存在: {filename}")
            continue

        try:
            # 创建软连接：os.symlink(源文件, 目标链接)
            os.symlink(src_file, dst_file)
            count += 1
        except OSError as e:
            print(f"❌ 创建连接失败 {filename}: {e}")

    print("-" * 30)
    print(f"🎉 处理完成！")
    print(f"共创建了 {count} 个软连接。")
    print(f"新的数据集路径为: {target_dir}")

if __name__ == "__main__":
    create_symlinks()