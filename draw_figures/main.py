import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def parse_log_file(file_path, encodings):
    """解析日志文件并提取所需数据"""
    data = defaultdict(dict)
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    match = re.search(r"\{'time': '[^']+', 'step': (\d+), 'loss': [\d.]+,"
                                      r" 'lm_loss': ([\d.]+), 'aux_loss': ([\d.]+)", line)
                    if match and "student_2_loss" not in line:
                        step = int(match.group(1))
                        lm_loss = float(match.group(2))
                        aux_loss = float(match.group(3))
                        data[step]['lm_loss'] = lm_loss
                        data[step]['aux_loss'] = aux_loss
                    else:
                        # 尝试匹配第二个算法的日志格式
                        match = re.search(r"\{'time': '[^']+', 'step': (\d+), 'loss': [\d.]+,"
                                          r" 'lm_loss': ([\d.]+), 'aux_loss': ([\d.]+), "
                                          r"'student_2_loss': ([\d.]+), 'student_2_hidden_loss': [\d.]+, "
                                          r"'student_2_kl_loss': [\d.]+, 'student_4_loss': ([\d.]+), "
                                          r"'student_4_hidden_loss': [\d.]+, 'student_4_kl_loss': [\d.]+, "
                                          r"'student_8_loss': ([\d.]+)"
                                          , line)
                        if match:
                            step = int(match.group(1))
                            lm_loss = float(match.group(2))
                            aux_loss = float(match.group(3))
                            student_2_loss = float(match.group(4))
                            student_4_loss = float(match.group(5))
                            student_8_loss = float(match.group(6))
                            # print(step, lm_loss, aux_loss, student_2_loss, student_4_loss, student_8_loss)
                            data[step]['lm_loss'] = lm_loss
                            data[step]['aux_loss'] = aux_loss
                            data[step]['student_2_loss'] = student_2_loss
                            data[step]['student_4_loss'] = student_4_loss
                            data[step]['student_8_loss'] = student_8_loss
            break  # 如果成功读取，则跳出循环
        except UnicodeDecodeError:
            continue
    return data

import pandas as pd
def smooth_data(y, window_size=5):
    if window_size < 2:
        return y
    s = pd.Series(y)
    return s.rolling(window_size, center=True, min_periods=1).mean().values


# 文件路径和编码尝试顺序
file_paths = [
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp16_16machine_z2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_hw_16machine_z2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_hw_16machine_z2_continue0.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_hw_16machine_z2_continue1.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_hw_16machine_z2_continue2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_hw_16machine_z2_continue3.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp8_16machine_z2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp8_16machine_z2_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp8_16machine_z2_continue_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp8_16machine_z2_continue_continue_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp4_16machine_z2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp4_16machine_z2_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp4_16machine_z2_continue_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp4_16machine_z2_continue_continue_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp4_16machine_z2_continue_continue_continue_continue.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp2_16machine_z2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/moe_exp2_16machine_z2_continue.log',
]
encodings = ['utf-8', 'latin-1', 'gbk', 'utf-16']

# 平滑开关和参数
SMOOTH = True  # 设置为False关闭平滑
WINDOW_SIZE = 20  # 平滑窗口大小

# 解析所有文件
origin_data = parse_log_file(file_paths[0], encodings)
hw_data = parse_log_file(file_paths[1], encodings)
hw_continue_data = parse_log_file(file_paths[2], encodings)
hw_continue_data1 = parse_log_file(file_paths[3], encodings)
hw_continue_data2 = parse_log_file(file_paths[4], encodings)
hw_continue_data3 = parse_log_file(file_paths[5], encodings)
hw_exp_8 = parse_log_file(file_paths[6], encodings)
hw_exp_8_continue = parse_log_file(file_paths[7], encodings)
hw_exp_8_continue_continue = parse_log_file(file_paths[8], encodings)
hw_exp_8_continue_continue_continue = parse_log_file(file_paths[9], encodings)
hw_exp_4 = parse_log_file(file_paths[10], encodings)
hw_exp_4_continue = parse_log_file(file_paths[11], encodings)
hw_exp_4_continue_continue = parse_log_file(file_paths[12], encodings)
hw_exp_4_continue_continue_continue = parse_log_file(file_paths[13], encodings)
hw_exp_4_continue_continue_continue_continue = parse_log_file(file_paths[14], encodings)
hw_exp_2 = parse_log_file(file_paths[15], encodings)
hw_exp_2_continue = parse_log_file(file_paths[16], encodings)

# 合并hw_data和hw_continue_data，优先使用continue中的数据
merged_hw_data = {**hw_data, **hw_continue_data, **hw_continue_data1, **hw_continue_data2, **hw_continue_data3}
hw_exp_8 = {**hw_exp_8, **hw_exp_8_continue, **hw_exp_8_continue_continue, **hw_exp_8_continue_continue_continue}
hw_exp_4 = {**hw_exp_4, **hw_exp_4_continue, **hw_exp_4_continue_continue, **hw_exp_4_continue_continue_continue, **hw_exp_4_continue_continue_continue_continue}
hw_exp_2 = {**hw_exp_2, **hw_exp_2_continue}


# 准备绘图数据
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


# 绘制图形
plt.figure(figsize=(12, 8))

# 绘制原始算法数据
steps, lm_loss = prepare_plot_data(origin_data, 'lm_loss')
plt.plot(steps, lm_loss, label='Origin LM Loss (16 Exps)', linestyle='-', linewidth=2)

# steps, aux_loss = prepare_plot_data(origin_data, 'aux_loss')
# plt.plot(steps, aux_loss, label='Origin Aux Loss', linestyle='--', linewidth=2)

# 绘制新算法数据
steps, lm_loss_hw = prepare_plot_data(merged_hw_data, 'lm_loss')
plt.plot(steps, lm_loss_hw, label='HW LM Loss', linestyle='-', linewidth=2)

# 绘制新算法数据
steps, lm_loss_exp8 = prepare_plot_data(hw_exp_8, 'lm_loss')
plt.plot(steps, lm_loss_exp8, label='Origin LM Loss (8 Exps)', linestyle='-', linewidth=2)

steps, lm_loss_exp4 = prepare_plot_data(hw_exp_4, 'lm_loss')
plt.plot(steps, lm_loss_exp4, label='Origin LM Loss (4 Exps)', linestyle='-', linewidth=2)

steps, lm_loss_exp2 = prepare_plot_data(hw_exp_2, 'lm_loss')
plt.plot(steps, lm_loss_exp4, label='Origin LM Loss (2 Exps)', linestyle='-', linewidth=2)

# steps, aux_loss = prepare_plot_data(merged_hw_data, 'aux_loss')
# plt.plot(steps, aux_loss, label='HW Aux Loss', linestyle='--', linewidth=2)

steps, student_2_loss = prepare_plot_data(merged_hw_data, 'student_2_loss')
plt.plot(steps, student_2_loss, label='HW Student 2 Loss', linestyle='-', linewidth=2)

steps, student_4_loss = prepare_plot_data(merged_hw_data, 'student_4_loss')
plt.plot(steps, student_4_loss, label='HW Student 4 Loss', linestyle='-', linewidth=2)

steps, student_8_loss = prepare_plot_data(merged_hw_data, 'student_8_loss')
plt.plot(steps, student_8_loss, label='HW Student 8 Loss', linestyle='-', linewidth=2)

# 设置对数坐标轴
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Loss (log scale)')
plt.title(f'Loss vs Step Comparison{" (Smoothed)" if SMOOTH else ""}')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)


# print(lm_loss_hw[-20:], lm_loss_exp4[-20:], lm_loss_exp8[-20:], student_4_loss[-20:], student_8_loss[-20:])

"""
"""


offset = 10000
last_step = 2000


ax = plt.gca()
# 插图：放大区域
axins = zoomed_inset_axes(ax, zoom=4, loc='upper center')  # zoom倍放大
axins.plot(steps[offset-last_step:offset], lm_loss[offset-last_step:offset], label='Origin LM Loss (16 Exps)', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_hw[offset-last_step:offset], label='HW LM Loss', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_exp8[offset-last_step:offset], label='Origin LM Loss (8 Exps)', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_exp4[offset-last_step:offset], label='Origin LM Loss (4 Exps)', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_exp2[offset-last_step:offset], label='Origin LM Loss (2 Exps)', linestyle='-', linewidth=2)

axins.plot(steps[offset-last_step:offset], student_2_loss[offset-last_step:offset], label='HW Student 2 Loss', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], student_4_loss[offset-last_step:offset], label='HW Student 4 Loss', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], student_8_loss[offset-last_step:offset], label='HW Student 8 Loss', linestyle='-', linewidth=2)
# 设置插图的显示区域
x1, x2 = min(steps[offset-last_step:offset]), max(steps[offset-last_step:offset])
y1, y2 = 2.2, 2.55
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray")


"""
"""
offset = 29297
last_step = 2000


# ax = plt.gca()
# 插图：放大区域
axins = zoomed_inset_axes(ax, zoom=4, loc='center right')  # zoom倍放大
axins.plot(steps[offset-last_step:offset], lm_loss[offset-last_step:offset], label='Origin LM Loss (16 Exps)', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_hw[offset-last_step:offset], label='HW LM Loss', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_exp8[offset-last_step:offset], label='Origin LM Loss (8 Exps)', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_exp4[offset-last_step:offset], label='Origin LM Loss (4 Exps)', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], lm_loss_exp2[offset-last_step:offset], label='Origin LM Loss (2 Exps)', linestyle='-', linewidth=2)

axins.plot(steps[offset-last_step:offset], student_2_loss[offset-last_step:offset], label='HW Student 2 Loss', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], student_4_loss[offset-last_step:offset], label='HW Student 4 Loss', linestyle='-', linewidth=2)
axins.plot(steps[offset-last_step:offset], student_8_loss[offset-last_step:offset], label='HW Student 8 Loss', linestyle='-', linewidth=2)
# 设置插图的显示区域
x1, x2 = min(steps[offset-last_step:offset]), max(steps[offset-last_step:offset])
y1, y2 = 2.1, 2.35
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray")

"""
"""

# 显示图形
plt.tight_layout()
plt.savefig("loss.png")
# plt.show()