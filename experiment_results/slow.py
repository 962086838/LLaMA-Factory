import wandb
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# ==========================================
# 1. 实验配置区 (未来更新代码只需要改这里)
# ==========================================
# 格式: {"你给实验起的简称": "用户名/项目名/run_id"}
RUN_DICT = {
    "baseline": "hallway/llamafactory/x2wri8jb",
    "mhc": "hallway/llamafactory/ls0k9m2s",
    # 未来有新实验，按这个格式取消注释并填入即可：
    # "experiment_v2": "hallway/llamafactory/xxxxxx",
}

# 想要对比和做差的指标
METRIC = "train/loss"

# ==========================================
# 2. 数据拉取与对齐核心逻辑 (通常不需要改)
# ==========================================
api = wandb.Api()
all_dataframes = []

print("正在从 WandB 拉取数据...")
for exp_name, run_path in RUN_DICT.items():
    print(f" -> 正在获取 {exp_name} ({run_path})")
    try:
        run = api.run(run_path)
        # scan_history 获取全量数据（如果不加 scan 默认只取 500 个点）
        history = run.scan_history(keys=["_step", METRIC])
        
        # 转换为 Pandas DataFrame，重命名指标列为实验名称
        df = pd.DataFrame([{
            "step": row["_step"], 
            exp_name: row[METRIC]
        } for row in history if METRIC in row])
        
        if not df.empty:
            all_dataframes.append(df)
        else:
            print(f"    [警告] {exp_name} 中没有找到 {METRIC} 数据！")
    except Exception as e:
        print(f"    [报错] 获取 {exp_name} 失败: {e}")

# 将所有 DataFrame 按照 step 严格对齐 (Outer Join 保证不丢失任何一方的记录点)
if all_dataframes:
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="step", how="outer"), all_dataframes)
    df_merged = df_merged.sort_values("step").reset_index(drop=True)
    
    # 因为外连接会产生 NaN (比如某一步 baseline 有记录但 mhc 没有)
    # 使用线性插值填补空缺，这样后续两列相减时就不会算出 NaN
    df_merged = df_merged.interpolate(method='linear')

    # ==========================================
    # 3. 结果计算与可视化绘制
    # ==========================================
    plt.figure(figsize=(12, 7))
    
    # [3.1 画各自的 Loss 曲线]
    for exp_name in RUN_DICT.keys():
        if exp_name in df_merged.columns:
            plt.plot(df_merged["step"], df_merged[exp_name], label=f"{exp_name} Loss", alpha=0.7)
    
    # [3.2 选定两个特定实验做差并画图]
    # 这里我们专门算出 mhc 和 baseline 的差值
    target_exp = "mhc"
    base_exp = "baseline"
    
    if target_exp in df_merged.columns and base_exp in df_merged.columns:
        diff_col_name = f"Diff ({target_exp} - {base_exp})"
        df_merged[diff_col_name] = df_merged[target_exp] - df_merged[base_exp]
        
        # 用不同样式（虚线+黑色）画出差值曲线
        plt.plot(df_merged["step"], df_merged[diff_col_name], 
                 label=diff_col_name, color='black', linestyle='--')
        
        # 画一条 y=0 的红色基准线，方便看清楚什么时候 mhc 的 loss 更低 (负数)
        plt.axhline(0, color='red', linestyle=':', alpha=0.6)

    # 完善图表信息
    plt.xlabel("Step")
    plt.ylabel(METRIC)
    plt.title("Train Loss Comparison & Difference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 如果需要把数据存下来供以后慢慢看，可以取消下面这行的注释导出为 Excel/CSV：
    # df_merged.to_csv("loss_comparison.csv", index=False)

else:
    print("没有拉取到任何有效数据。请检查网络或 Run 路径是否正确。")