#!/bin/bash

# 1. 基础配置
SESSION_NAME="cluster_train"        # tmux 会话的名称
TASKS_NUM=${GEMINI_TASKS_NUM:-4}    # 默认任务数为 4，如果没有设置环境变量的话

# 共同执行的指令
CMD_CD="cd /gfs/space/private/hehaowei/LLaMA-Factory/"
CMD_CONDA="conda activate llama_factory_py312"
# 获取传入的任务类型，默认为 mhc
TASK_TYPE=${1:-mhc}

case "$TASK_TYPE" in
    baseline)
        CMD_TRAIN="bash run_dsv32_6B_mca_basline_pretrain.sh"
        ;;
    baseline_moe_bia_update_ratio_5000)
        CMD_TRAIN="bash run_dsv32_6B_mca_basline_moe_bias_update_ratio_5000_pretrain.sh"
        ;;
    baseline_moe_bia_update_ratio_2000)
        CMD_TRAIN="bash run_dsv32_6B_mca_basline_moe_bias_update_ratio_2000_pretrain.sh"
        ;;
    jitter)
        CMD_TRAIN="bash run_dsv32_6B_mca_moe_jitter_noise_pretrain.sh"
        ;;
    mhc)
        CMD_TRAIN="bash run_dsv32_6B_mca_mhc_pretrain.sh"
        ;;
    mhc2streams)
        CMD_TRAIN="bash run_dsv32_6B_mca_mhc_2streams_pretrain.sh"
        ;;
    mhc6streams)
        CMD_TRAIN="bash run_dsv32_6B_mca_mhc_6streams_pretrain.sh"
        ;;
    muon)
        CMD_TRAIN="bash run_dsv32_6B_mca_muon_pretrain.sh"
        ;;
    *)
        echo "Error: 未知的任务类型 '$TASK_TYPE'"
        echo "Usage: $0 [baseline|jitter|mhc]"
        exit 1
        ;;
esac

# 2. 创建一个新的 tmux 会话 (后台运行)
# 如果会话已存在，会报错，所以最好确保没有同名会话
tmux new-session -d -s $SESSION_NAME

# 3. 循环创建窗口并执行指令
for (( i=0; i<$TASKS_NUM; i++ ))
do
    # 动态获取机器 IP 的环境变量值 (Bash 的间接变量引用)
    VAR_NAME="GEMINI_IP_taskrole1_$i"
    TARGET_IP=${!VAR_NAME}
    
    # 如果环境变量未在脚本外导出，也可以直接让 tmux 发送原样字符串：
    # TARGET_IP="\$GEMINI_IP_taskrole1_$i"

    # 如果是第 0 个任务，重命名默认的第一个窗口；否则创建新窗口
    if [ $i -eq 0 ]; then
        tmux rename-window -t $SESSION_NAME:$i "node_$i"
    else
        tmux new-window -t $SESSION_NAME -n "node_$i"
    fi

    # 4. 向对应窗口发送 SSH 命令并回车 (C-m 代表回车 Enter)
    tmux send-keys -t $SESSION_NAME:node_$i "ssh $TARGET_IP" C-m

    # 等待 1-2 秒，确保 SSH 连接成功。
    # 如果网络较慢或机器响应慢，可以适当调大这个数字。
    sleep 1

    # 5. SSH 连上后，发送共同的指令
    tmux send-keys -t $SESSION_NAME:node_$i "$CMD_CD" C-m
    tmux send-keys -t $SESSION_NAME:node_$i "$CMD_CONDA" C-m
    tmux send-keys -t $SESSION_NAME:node_$i "$CMD_TRAIN" C-m
    
    echo "已启动节点 $i: $TARGET_IP"
done

# 6. 全部发送完毕后，附加(Attach)到这个 tmux 会话中查看进度
echo "所有任务下发完毕，正在进入 tmux 会话..."
tmux attach-session -t $SESSION_NAME