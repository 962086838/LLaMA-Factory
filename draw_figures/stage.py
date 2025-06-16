import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from main import parse_log_file


import pandas as pd
def smooth_data(y, window_size=5):
    if window_size < 2:
        return y
    s = pd.Series(y)
    return s.rolling(window_size, center=True, min_periods=1).mean().values


# 文件路径和编码尝试顺序
file_paths = '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp16_ckpt24000_continue_distill.log'
encodings = ['utf-8', 'latin-1', 'gbk', 'utf-16']

# 平滑开关和参数
SMOOTH = True  # 设置为False关闭平滑
WINDOW_SIZE = 20  # 平滑窗口大小

plt.figure(figsize=(12, 8))


def prepare_plot_data(data_dict, key):
    steps = sorted(data_dict.keys())
    values = [data_dict[step].get(key, np.nan) for step in steps]

    # 处理NaN值
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
    valid_steps = [steps[i] for i in valid_indices]
    valid_values = [values[i] for i in valid_indices]

    if SMOOTH and len(valid_values) > WINDOW_SIZE:
        valid_values = smooth_data(valid_values, WINDOW_SIZE)

    return valid_steps, valid_values


plain_training_data = parse_log_file("/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp16_16machine_z2.log", encodings)

plain_steps, plain_lm_loss = prepare_plot_data(plain_training_data, 'lm_loss')
plt.plot(plain_steps, plain_lm_loss, label='Plain LM Loss (16 Exps)', linestyle='-', linewidth=1)


# 解析所有文件
with open(file_paths, "r") as f:
    lines = f.readlines()

data = defaultdict(dict)

for line in lines:
    """
    10.244.135.25: {''student_2_kl_loss': 1.1181, 'student_3_loss': 9.6925, 'student_3_hidden_loss': 3.2491, 'student_3_kl_loss': 1.3977, 'student_4_loss': 7.4853, 'student_4_hidden_loss': 2.7255, 'student_4_kl_loss': 1.9997, 'student_5_loss': 6.389, 'student_5_hidden_loss': 2.4784, 'student_5_kl_loss': 2.3508, 'student_6_loss': 4.9232, 'student_6_hidden_loss': 1.9541, 'student_6_kl_loss': 2.5965, 'student_7_loss': 4.0984, 'student_7_hidden_loss': 1.4026, 'student_7_kl_loss': 1.9686, 'student_8_loss': 3.6656, 'student_8_hidden_loss': 1.1682, 'student_8_kl_loss': 1.5273, 'student_9_loss': 3.268, 'student_9_hidden_loss': 0.9787, 'student_9_kl_loss': 1.4539, 'student_10_loss': 3.0548, 'student_10_hidden_loss': 0.8427, 'student_10_kl_loss': 1.3036, 'student_11_loss': 2.8433, 'student_11_hidden_loss': 0.6965, 'student_11_kl_loss': 1.2454, 'student_12_loss': 2.6016, 'student_12_hidden_loss': 0.5692, 'student_12_kl_loss': 1.2381, 'student_13_loss': 2.4518, 'student_13_hidden_loss': 0.444, 'student_13_kl_loss': 1.217, 'student_14_loss': 2.3461, 'student_14_hidden_loss': 0.3049, 'student_14_kl_loss': 1.3298, 'student_15_loss': 2.2619, 'student_15_hidden_loss': 0.1954, 'student_15_kl_loss': 1.3594, 'grad_norm': 0.31523433327674866, 'learning_rate': 2.401920331049412e-05, 'epoch': 0.82}
    """
    if "time" in line and "loss" in line and "student" in line:
        match = re.search(r"\{'time': '[^']+', 'step': (\d+), 'loss': [\d.]+,"
                          r" 'lm_loss': ([\d.]+), 'aux_loss': ([\d.]+), "
                          r"'student_2_loss': ([\d.]+), 'student_2_hidden_loss': [\d.]+, "
                          r"'student_2_kl_loss': [\d.]+, 'student_3_loss': ([\d.]+), "
                          r"'student_3_hidden_loss': [\d.]+, 'student_3_kl_loss': [\d.]+, "
                          r"'student_4_loss': ([\d.]+), 'student_4_hidden_loss': [\d.]+, 'student_4_kl_loss': [\d.]+, "
                          r"'student_5_loss': ([\d.]+), 'student_5_hidden_loss': [\d.]+, 'student_5_kl_loss': [\d.]+, "
                          r"'student_6_loss': ([\d.]+), 'student_6_hidden_loss': [\d.]+, 'student_6_kl_loss': [\d.]+, "
                          r"'student_7_loss': ([\d.]+), 'student_7_hidden_loss': [\d.]+, 'student_7_kl_loss': [\d.]+, "
                          r"'student_8_loss': ([\d.]+), 'student_8_hidden_loss': [\d.]+, 'student_8_kl_loss': [\d.]+, "
                          r"'student_9_loss': ([\d.]+), 'student_9_hidden_loss': [\d.]+, 'student_9_kl_loss': [\d.]+, "
                          r"'student_10_loss': ([\d.]+), 'student_10_hidden_loss': [\d.]+, 'student_10_kl_loss': [\d.]+, "
                          r"'student_11_loss': ([\d.]+), 'student_11_hidden_loss': [\d.]+, 'student_11_kl_loss': [\d.]+, "
                          r"'student_12_loss': ([\d.]+), 'student_12_hidden_loss': [\d.]+, 'student_12_kl_loss': [\d.]+, "
                          r"'student_13_loss': ([\d.]+), 'student_13_hidden_loss': [\d.]+, 'student_13_kl_loss': [\d.]+, "
                          r"'student_14_loss': ([\d.]+), 'student_14_hidden_loss': [\d.]+, 'student_14_kl_loss': [\d.]+, "
                          r"'student_15_loss': ([\d.]+), 'student_15_hidden_loss': [\d.]+, 'student_15_kl_loss': [\d.]+, "
                          , line)
        if match:
            step = int(match.group(1))
            lm_loss = float(match.group(2))
            aux_loss = float(match.group(3))
            student_2_loss = float(match.group(4))
            student_3_loss = float(match.group(5))
            student_4_loss = float(match.group(6))
            student_5_loss = float(match.group(7))
            student_6_loss = float(match.group(8))
            student_7_loss = float(match.group(9))
            student_8_loss = float(match.group(10))
            student_9_loss = float(match.group(11))
            student_10_loss = float(match.group(12))
            student_11_loss = float(match.group(13))
            student_12_loss = float(match.group(14))
            student_13_loss = float(match.group(15))
            student_14_loss = float(match.group(16))
            student_15_loss = float(match.group(17))

            data[step]['lm_loss'] = lm_loss
            data[step]['aux_loss'] = aux_loss
            data[step]['student_2_loss'] = student_2_loss
            data[step]['student_3_loss'] = student_3_loss
            data[step]['student_4_loss'] = student_4_loss
            data[step]['student_5_loss'] = student_5_loss
            data[step]['student_6_loss'] = student_6_loss
            data[step]['student_7_loss'] = student_7_loss
            data[step]['student_8_loss'] = student_8_loss
            data[step]['student_9_loss'] = student_9_loss
            data[step]['student_10_loss'] = student_10_loss
            data[step]['student_11_loss'] = student_11_loss
            data[step]['student_12_loss'] = student_12_loss
            data[step]['student_13_loss'] = student_13_loss
            data[step]['student_14_loss'] = student_14_loss
            data[step]['student_15_loss'] = student_15_loss


# 合并hw_data和hw_continue_data，优先使用continue中的数据


# 准备绘图数据


# 绘制图形

# 绘制原始算法数据
steps, lm_loss = prepare_plot_data(data, 'lm_loss')
plt.plot(steps, lm_loss, label='Stage HWMoE Loss (16 Exps)', linestyle='-', linewidth=1)

for stu_num in range(2, 16):
    steps, stu_loss = prepare_plot_data(data, f'student_{stu_num}_loss')
    print(f"{stu_num}, {stu_loss[0:20]}")
    plt.plot(steps, stu_loss, label=f'Stage LM Loss ({stu_num} Exps)', linestyle='-', linewidth=1)
print(lm_loss[0:20])




plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Loss (log scale)')
plt.title(f'Loss vs Step Comparison{" (Smoothed)" if SMOOTH else ""}')
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="-", alpha=0.5)



offset = 710
last_step = 710
ax = plt.gca()

"""
"""
# ax = plt.gca()
# 插图：放大区域
axins = zoomed_inset_axes(ax, zoom=2.5, loc='upper center')  # zoom倍放大
axins.plot(plain_steps[24000+offset-last_step:24000+offset], plain_lm_loss[24000+offset-last_step:24000+offset], label='Stage HWMoE Loss (16 Exps)', linestyle='-', linewidth=1)
axins.plot(steps[offset-last_step:offset], lm_loss[offset-last_step:offset], label='Stage HWMoE Loss (16 Exps)', linestyle='-', linewidth=1)
for stu_num in range(2, 16):
    steps, stu_loss = prepare_plot_data(data, f'student_{stu_num}_loss')
    axins.plot(steps[offset - last_step:offset], stu_loss[offset - last_step:offset], label=f'Stage LM Loss ({stu_num} Exps)',
               linestyle='-', linewidth=1)

# 设置插图的显示区域
x1, x2 = min(steps[offset-last_step:offset]), max(steps[offset-last_step:offset])
y1, y2 = 2.1, 3.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray")
"""
"""

# 显示图形
plt.tight_layout()
plt.savefig("stage.png")
# plt.show()