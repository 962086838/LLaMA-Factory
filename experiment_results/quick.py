import wandb
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# ==========================================
# 1. 实验配置区
# ==========================================
RUN_DICT = {
    "baseline": "hallway/llamafactory/x2wri8jb",
    "mhc": "hallway/llamafactory/ls0k9m2s",
    "jitter": "hallway/llamafactory/myhiq6bn",
}
METRIC = "train/loss"

target_exp = "jitter"
base_exp = "baseline"

# ==========================================
# 2. 极速数据拉取与对齐核心逻辑
# ==========================================
api = wandb.Api()
all_dataframes = []

print("正在从 WandB 拉取数据 (极速降采样模式)...")
for exp_name, run_path in RUN_DICT.items():
    print(f" -> 正在获取 {exp_name}")
    try:
        run = api.run(run_path)
        
        # 核心优化点：
        # 使用 history() 并指定 samples=1000 (代表均匀抽取 1000 个点)
        # pandas=True 直接返回 DataFrame，极大地提升处理速度
        df = run.history(keys=["_step", METRIC], pandas=True, samples=500)
        
        if METRIC in df.columns:
            # 只保留需要的列，并将 _step 重命名为 step，指标列重命名为实验名
            df = df[["_step", METRIC]].rename(columns={"_step": "step", METRIC: exp_name})
            all_dataframes.append(df)
        else:
            print(f"    [警告] {exp_name} 中没有找到 {METRIC} 数据！")
    except Exception as e:
        print(f"    [报错] 获取 {exp_name} 失败: {e}")

# ==========================================
# 3. 数据合并与绘图
# ==========================================
if all_dataframes:
    # 依然使用外连接保证数据对齐
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="step", how="outer"), all_dataframes)
    df_merged = df_merged.sort_values("step").reset_index(drop=True)
    df_merged = df_merged.interpolate(method='linear')

    plt.figure(figsize=(8, 5))
    
    # [画各自的 Loss 曲线]
    # for exp_name in RUN_DICT.keys():
    #     if exp_name in df_merged.columns:
    #         plt.plot(df_merged["step"], df_merged[exp_name], label=f"{exp_name} Loss", alpha=0.7)
    
    # [画差值曲线]    
    if target_exp in df_merged.columns and base_exp in df_merged.columns:
        diff_col_name = f"Diff ({target_exp} - {base_exp})"
        df_merged[diff_col_name] = df_merged[target_exp] - df_merged[base_exp]
        
        plt.plot(df_merged["step"], df_merged[diff_col_name], 
                 label=diff_col_name, color='black', linestyle='--')
        plt.axhline(0, color='red', linestyle=':', alpha=0.6)

    plt.xlabel("Step")
    plt.ylabel(METRIC)
    plt.title("Train Loss Comparison & Difference (Sampled)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("没有拉取到任何有效数据。请检查网络或 Run 路径是否正确。")