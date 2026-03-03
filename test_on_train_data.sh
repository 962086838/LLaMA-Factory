#!/bin/bash

# 这是一个使用 LLaMA-Factory 将数据集全部用于测试（评估/预测）的示例脚本。
# 根据你的思路，可以通过不指定 `--dataset`，而是指定 `--eval_dataset` 将训练集名称传入，
# 并且设置 `--do_train false` 和 `--do_eval true`（或 `--do_predict true`）。
#
# 这样就可以把通常用于训练的数据集，完全当作测试集来用。

HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 deepspeed  src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage pt \
    --model_name_or_path /gfs/space/private/hehaowei/model_ckpt/36B_1.82T_torch \
    --trust_remote_code \
    --eval_dataset fineweb_edu_10b \
    --dataset_dir "data" \
    --template default \
    --finetuning_type lora \
    --output_dir saves/eval/36B_1.82T_torch_8K \
    --do_train false \
    --do_eval true \
    --per_device_eval_batch_size 6 \
    --cutoff_len 2048 \
    --overwrite_cache true \
    --preprocessing_num_workers 48 \
    --max_samples 10000 \
    --tokenized_path tokenized_path/fineweb_edu_10b_eval_telechat36 

    

HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 deepspeed  src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage pt \
    --model_name_or_path /gfs/space/private/hehaowei/model_ckpt/36B_5.13T_L4.5.4-100B_torch \
    --trust_remote_code \
    --eval_dataset fineweb_edu_10b \
    --dataset_dir "data" \
    --template default \
    --finetuning_type lora \
    --output_dir saves/eval/36B_5.13T_L4.5.4-100B_torch \
    --do_train false \
    --do_eval true \
    --per_device_eval_batch_size 6 \
    --cutoff_len 2048 \
    --overwrite_cache true \
    --preprocessing_num_workers 48 \
    --max_samples 10000 \
    --tokenized_path tokenized_path/fineweb_edu_10b_eval_telechat36 



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file examples/accelerate/fsdp_config_8gpu.yaml \
    src/train.py \
    --stage pt \
    --model_name_or_path /gfs/space/private/hehaowei/model_ckpt/k8s_tasks_yt_sft_105B_formal3_checkpoint_unified_ckpt_tmp \
    --trust_remote_code \
    --eval_dataset fineweb_edu_10b \
    --dataset_dir "data" \
    --template default \
    --finetuning_type full \
    --output_dir saves/eval/k8s_tasks_yt_sft_105B_formal3_checkpoint_unified_ckpt_tmp \
    --do_train false \
    --do_eval true \
    --per_device_eval_batch_size 2 \
    --cutoff_len 2048 \
    --overwrite_cache true \
    --preprocessing_num_workers 48 \
    --max_samples 10000 \
    --tokenized_path tokenized_path/fineweb_edu_10b_eval_telechat105 
