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
                                      r" 'lm_loss': ([\d.]+), 'aux_loss': ([\d.]+), 'grad_norm': ([\d.]+)", line)
                    if match and "student_2_loss" not in line:
                        step = int(match.group(1))
                        lm_loss = float(match.group(2))
                        aux_loss = float(match.group(3))
                        grad_norm = float(match.group(4))
                        data[step]['lm_loss'] = lm_loss
                        data[step]['aux_loss'] = aux_loss
                        data[step]['grad_norm'] = grad_norm
                    else:
                        # 尝试匹配第二个算法的日志格式
                        match = re.search(r"\{'time': '[^']+', 'step': (\d+), 'loss': [\d.]+,"
                                          r" 'lm_loss': ([\d.]+), 'aux_loss': ([\d.]+), "
                                          r"'student_2_loss': ([\d.]+), 'student_2_hidden_loss': ([\d.]+), "
                                          r"'student_2_kl_loss': ([\d.]+), 'student_4_loss': ([\d.]+), "
                                          r"'student_4_hidden_loss': ([\d.]+), 'student_4_kl_loss': ([\d.]+), "
                                          r"'student_8_loss': ([\d.]+), 'student_8_hidden_loss': ([\d.]+), "
                                          r"'student_8_kl_loss': ([\d.]+), 'grad_norm': ([\d.]+)"
                                          , line)
                        if match:
                            step = int(match.group(1))
                            lm_loss = float(match.group(2))
                            aux_loss = float(match.group(3))
                            student_2_loss = float(match.group(4))
                            student_2_hidden_loss = float(match.group(5))
                            student_2_kl_loss = float(match.group(6))
                            student_4_loss = float(match.group(7))
                            student_4_hidden_loss = float(match.group(8))
                            student_4_kl_loss = float(match.group(9))
                            student_8_loss = float(match.group(10))
                            student_8_hidden_loss = float(match.group(11))
                            student_8_kl_loss = float(match.group(12))
                            grad_norm = float(match.group(13))
                            # print(step, lm_loss, aux_loss, student_2_loss, student_4_loss, student_8_loss, student_2_kl_loss, student_4_kl_loss, student_8_kl_loss, student_2_hidden_loss, student_4_hidden_loss, student_8_hidden_loss)
                            # assert 1==0
                            data[step]['lm_loss'] = lm_loss
                            data[step]['aux_loss'] = aux_loss
                            data[step]['student_2_loss'] = student_2_loss
                            data[step]['student_4_loss'] = student_4_loss
                            data[step]['student_8_loss'] = student_8_loss
                            data[step]['student_2_kl_loss'] = student_2_kl_loss
                            data[step]['student_4_kl_loss'] = student_4_kl_loss
                            data[step]['student_8_kl_loss'] = student_8_kl_loss
                            data[step]['student_2_hidden_loss'] = student_2_hidden_loss
                            data[step]['student_4_hidden_loss'] = student_4_hidden_loss
                            data[step]['student_8_hidden_loss'] = student_8_hidden_loss
                            data[step]['grad_norm'] = grad_norm
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
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-hwmoe.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-hwmoe-p2.log',
    # '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-hwmoe.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-exp16-finewebedu100bt.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-exp8.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-exp4.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-exp2.log',
    '/gemini/space/thu/hehaowei/LLaMA-Factory/TeleChat-2B-A05B-hwmoe-param2.log',
]
encodings = ['utf-8', 'latin-1', 'gbk', 'utf-16']

# 平滑开关和参数
SMOOTH = True  # 设置为False关闭平滑
WINDOW_SIZE = 20  # 平滑窗口大小

# 解析所有文件
hw_data = parse_log_file(file_paths[0], encodings)
hw_data_p2 = parse_log_file(file_paths[1], encodings)
ori_data_exp16 = parse_log_file(file_paths[2], encodings)
ori_data_exp8 = parse_log_file(file_paths[3], encodings)
ori_data_exp4 = parse_log_file(file_paths[4], encodings)
ori_data_exp2 = parse_log_file(file_paths[5], encodings)
hw_data_param2 = parse_log_file(file_paths[6], encodings)


# 合并hw_data和hw_continue_data，优先使用continue中的数据
merged_hw_data = {**hw_data}
merged_hw_data_p2 = {**hw_data_p2}
ori_data_exp16 = {**ori_data_exp16}
ori_data_exp8 = {**ori_data_exp8}
ori_data_exp4 = {**ori_data_exp4}
ori_data_exp2 = {**ori_data_exp2}
merged_hw_data_param2 = {**hw_data_param2}

# 准备绘图数据
def prepare_plot_data(data_dict, key):
    steps = sorted(data_dict.keys())
    values = [data_dict[step].get(key, np.nan) for step in steps]

    # 处理NaN值
    valid_indices = [i for i, v in enumerate(values) if (not np.isnan(v) and v > 0)]
    valid_steps = [steps[i] for i in valid_indices]
    valid_values = [values[i] for i in valid_indices]

    if SMOOTH and len(valid_values) > WINDOW_SIZE:
        valid_values = smooth_data(valid_values, WINDOW_SIZE)

    return valid_steps, valid_values


# 绘制图形
plt.figure(figsize=(12, 8))

# 辅助loss
steps, hidden_loss_student_8 = prepare_plot_data(merged_hw_data, 'student_8_hidden_loss')
steps, hidden_loss_student_4 = prepare_plot_data(merged_hw_data, 'student_4_hidden_loss')
steps, hidden_loss_student_2 = prepare_plot_data(merged_hw_data, 'student_2_hidden_loss')
steps, kl_loss_student_8 = prepare_plot_data(merged_hw_data, 'student_8_kl_loss')
steps, kl_loss_student_4 = prepare_plot_data(merged_hw_data, 'student_4_kl_loss')
steps, kl_loss_student_2 = prepare_plot_data(merged_hw_data, 'student_2_kl_loss')
# plt.plot(steps, hidden_loss_student_8, label='HWMoE Hidden Loss (8 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, hidden_loss_student_4, label='HWMoE Hidden Loss (4 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, hidden_loss_student_2, label='HWMoE Hidden Loss (2 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, kl_loss_student_8, label='HWMoE KL Loss (8 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, kl_loss_student_4, label='HWMoE KL Loss (4 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, kl_loss_student_2, label='HWMoE KL Loss (2 Exps)', linestyle='-', linewidth=0.5)

# 辅助loss
steps, hidden_loss_student_8_param2 = prepare_plot_data(merged_hw_data_param2, 'student_8_hidden_loss')
steps, hidden_loss_student_4_param2 = prepare_plot_data(merged_hw_data_param2, 'student_4_hidden_loss')
steps, hidden_loss_student_2_param2 = prepare_plot_data(merged_hw_data_param2, 'student_2_hidden_loss')
steps, kl_loss_student_8_param2 = prepare_plot_data(merged_hw_data_param2, 'student_8_kl_loss')
steps, kl_loss_student_4_param2 = prepare_plot_data(merged_hw_data_param2, 'student_4_kl_loss')
steps, kl_loss_student_2_param2 = prepare_plot_data(merged_hw_data_param2, 'student_2_kl_loss')
# plt.plot(steps, hidden_loss_student_8_param2, label='HWMoE Hidden Loss (8 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, hidden_loss_student_4_param2, label='HWMoE Hidden Loss (4 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, hidden_loss_student_2_param2, label='HWMoE Hidden Loss (2 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, kl_loss_student_8_param2, label='HWMoE KL Loss (8 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, kl_loss_student_4_param2, label='HWMoE KL Loss (4 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, kl_loss_student_2_param2, label='HWMoE KL Loss (2 Exps)', linestyle='-', linewidth=0.5)



# # 绘制原始算法数据
coeff_hidden = 1e-5
coeff_kl = 1e-2
steps, lm_loss_hw = prepare_plot_data(merged_hw_data, 'lm_loss')
steps, lm_loss_student_8 = prepare_plot_data(merged_hw_data, 'student_8_loss')
steps, lm_loss_student_4 = prepare_plot_data(merged_hw_data, 'student_4_loss')
steps, lm_loss_student_2 = prepare_plot_data(merged_hw_data, 'student_2_loss')
# plt.plot(steps, lm_loss_hw, label='HWMoE LM Loss (16 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, lm_loss_student_8 - coeff_hidden * hidden_loss_student_8 - coeff_kl * kl_loss_student_8, label='HWMoE LM Loss (8 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, lm_loss_student_4 - coeff_hidden * hidden_loss_student_4 - coeff_kl * kl_loss_student_4, label='HWMoE LM Loss (4 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(steps, lm_loss_student_2 - coeff_hidden * hidden_loss_student_2 - coeff_kl * kl_loss_student_2, label='HWMoE LM Loss (2 Exps)', linestyle='-', linewidth=0.5)


# param2
coeff_hidden = 5e-2
coeff_kl = 1
steps_param2, lm_loss_hw_param2 = prepare_plot_data(merged_hw_data_param2, 'lm_loss')
steps_param2, lm_loss_student_8_param2 = prepare_plot_data(merged_hw_data_param2, 'student_8_loss')
steps_param2, lm_loss_student_4_param2 = prepare_plot_data(merged_hw_data_param2, 'student_4_loss')
steps_param2, lm_loss_student_2_param2 = prepare_plot_data(merged_hw_data_param2, 'student_2_loss')
# plt.plot(np.array(steps_param2)/2, lm_loss_hw_param2, label='HWMoE LM Loss (16 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(np.array(steps_param2)/2, lm_loss_student_8_param2 - coeff_hidden * hidden_loss_student_8_param2 - coeff_kl * kl_loss_student_8_param2, label='HWMoE LM Loss (8 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(np.array(steps_param2)/2, lm_loss_student_4_param2 - coeff_hidden * hidden_loss_student_4_param2 - coeff_kl * kl_loss_student_4_param2, label='HWMoE LM Loss (4 Exps)', linestyle='-', linewidth=0.5)
# plt.plot(np.array(steps_param2)/2, lm_loss_student_2_param2 - coeff_hidden * hidden_loss_student_2_param2 - coeff_kl * kl_loss_student_2_param2, label='HWMoE LM Loss (2 Exps)', linestyle='-', linewidth=0.5)
# print(f"HWMoE param2. step 10000: {lm_loss_hw_param2[20000]}. step 14000: {lm_loss_hw_param2[28000]}. step 14500: {lm_loss_hw_param2[29000]}. step 15000: {lm_loss_hw_param2[30000]}.")



# steps_p2, lm_loss_hw_p2 = prepare_plot_data(merged_hw_data_p2, 'lm_loss')
# plt.plot(steps_p2, lm_loss_hw_p2, label='HWMoE-P2 LM Loss (16 Exps)', linestyle='-', linewidth=0.5)
# steps_p2, lm_loss_student_8_p2 = prepare_plot_data(merged_hw_data_p2, 'student_8_loss')
# plt.plot(steps_p2, lm_loss_student_8_p2, label='HWMoE-P2 LM Loss (8 Exps)', linestyle='-', linewidth=0.5)
# steps_p2, lm_loss_student_4_p2 = prepare_plot_data(merged_hw_data_p2, 'student_4_loss')
# plt.plot(steps_p2, lm_loss_student_4_p2, label='HWMoE-P2 LM Loss (4 Exps)', linestyle='-', linewidth=0.5)
# steps_p2, lm_loss_student_2_p2 = prepare_plot_data(merged_hw_data_p2, 'student_2_loss')
# plt.plot(steps_p2, lm_loss_student_2_p2, label='HWMoE-P2 LM Loss (2 Exps)', linestyle='-', linewidth=0.5)

steps_ori_exp16, lm_loss_ori_exp16 = prepare_plot_data(ori_data_exp16, 'lm_loss')
steps_ori, lm_loss_ori_exp8 = prepare_plot_data(ori_data_exp8, 'lm_loss')
steps_ori, lm_loss_ori_exp4 = prepare_plot_data(ori_data_exp4, 'lm_loss')
steps_ori, lm_loss_ori_exp2 = prepare_plot_data(ori_data_exp2, 'lm_loss')
SMOOTH = False
steps_ori_exp16, grad_norm_ori_exp16 = prepare_plot_data(ori_data_exp16, 'grad_norm')
SMOOTH = True
plt.plot(steps_ori_exp16, lm_loss_ori_exp16, label='Original MoE LM Loss (16 Exps)', linestyle='-', linewidth=0.5)
plt.plot(steps_ori, lm_loss_ori_exp8, label='Original MoE LM Loss (8 Exps)', linestyle='-', linewidth=0.5)
plt.plot(steps_ori, lm_loss_ori_exp4, label='Original MoE LM Loss (4 Exps)', linestyle='-', linewidth=0.5)
plt.plot(steps_ori, lm_loss_ori_exp2, label='Original MoE LM Loss (2 Exps)', linestyle='-', linewidth=0.5)
plt.plot(steps_ori_exp16, grad_norm_ori_exp16, label='Original MoE Grad Norm (16 Exps)', linestyle='-', linewidth=0.5)

print(f"Original MoE LM Loss at last step. Exp 16: {lm_loss_ori_exp16[-1]}, Exp 8: {lm_loss_ori_exp8[-1]}, Exp 4: {lm_loss_ori_exp4[-1]}, Exp 2: {lm_loss_ori_exp2[-1]}")

# print(lm_loss_hw_p2)

# 设置对数坐标轴
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel('Loss (log scale)')
plt.title(f'Loss vs Step Comparison{" (Smoothed)" if SMOOTH else ""}')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)


# # print(lm_loss_hw[-20:], lm_loss_exp4[-20:], lm_loss_exp8[-20:], student_4_loss[-20:], student_8_loss[-20:])
#
# """
# """
#
#
# offset = 10000
# last_step = 2000
#
#
ax = plt.gca()
# # 插图：放大区域
# axins = zoomed_inset_axes(ax, zoom=4, loc='upper center')  # zoom倍放大
#
# # 设置插图的显示区域
# x1, x2 = min(steps[offset-last_step:offset]), max(steps[offset-last_step:offset])
# y1, y2 = 2.2, 2.55
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticks([])
# axins.set_yticks([])
#
# mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray")
#
#
"""
"""
offset = 19297
last_step = 2000


# ax = plt.gca()
# # 插图：放大区域
# axins = zoomed_inset_axes(ax, zoom=4, loc='center right')  # zoom倍放大
# axins.plot(steps[offset-last_step:offset], lm_loss_hw[offset-last_step:offset], label='HWMoE LM Loss (16 Exps)', linestyle='-', linewidth=0.5)
# axins.plot(steps[offset-last_step:offset], lm_loss_student_8[offset-last_step:offset], label='HWMoE LM Loss (8 Exps)', linestyle='-', linewidth=0.5)
# axins.plot(steps[offset-last_step:offset], lm_loss_student_4[offset-last_step:offset], label='HWMoE LM Loss (4 Exps)', linestyle='-', linewidth=0.5)
# axins.plot(steps[offset-last_step:offset], lm_loss_student_2[offset-last_step:offset], label='HWMoE LM Loss (2 Exps)', linestyle='-', linewidth=0.5)
#
# axins.plot(steps_ori_exp16, lm_loss_ori_exp16, label='Ori LM Loss (16 Exps)', linestyle='-', linewidth=0.5)
# axins.plot(steps[offset-last_step:offset], lm_loss_ori_exp8[offset-last_step:offset], label='Ori LM Loss (8 Exps)', linestyle='-', linewidth=0.5)
# axins.plot(steps[offset-last_step:offset], lm_loss_ori_exp4[offset-last_step:offset], label='Ori LM Loss (4 Exps)', linestyle='-', linewidth=0.5)
# axins.plot(steps[offset-last_step:offset], lm_loss_ori_exp2[offset-last_step:offset], label='Ori LM Loss (2 Exps)', linestyle='-', linewidth=0.5)
#
# # 设置插图的显示区域
# x1, x2 = min(steps[offset-last_step:offset]), max(steps[offset-last_step:offset])
# y1, y2 = 2.4, 2.8
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticks([])
# axins.set_yticks([])
#
# mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray")

"""
"""

# 显示图形
plt.tight_layout()
plt.savefig("finewebedu_loss.png", dpi=400)
# plt.show()