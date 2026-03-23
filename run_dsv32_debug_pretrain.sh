cp /gfs/space/private/hehaowei/ROLL/mcore_adapter/src/mcore_adapter/models/model_config.py /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/mcore_adapter/models/
cp /gfs/space/private/hehaowei/ROLL/mcore_adapter/src/mcore_adapter/models/deepseek_v3/__init__.py /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/mcore_adapter/models/deepseek_v3/
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 USE_MCA=1 torchrun --nnodes=$GEMINI_TASKS_NUM --node_rank=$GEMINI_TASK_INDEX --master_addr=$GEMINI_HOST_IP_taskrole1_0 --master_port=29500 --nproc_per_node=8 src/train.py \
    --stage pt \
    --do_train \
    --train_from_scratch \
    --model_name_or_path DeepseekV32-small-mca \
    --dataset fineweb_edu_10b \
    --finetuning_type full \
    --output_dir saves/deepseekv32-small-mca-debug-2machine/pt/fineweb_edu_10b \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 500 \
    --learning_rate 1.0e-5 \
    --warmup_steps 100 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --preprocessing_num_workers 32 \
    --tokenized_path /gfs/space/private/hehaowei/LLaMA-Factory/tokenized_path/fineweb_edu_350b_deepseekv32_merged \
    --bf16 \
    --cutoff_len 4096 \
    --trust_remote_code \
    --flash_attn fa2 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4  \
    --sequence_parallel False  

