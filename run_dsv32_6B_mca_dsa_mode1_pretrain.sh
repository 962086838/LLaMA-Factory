cp -r /gfs/space/private/hehaowei/LLaMA-Factory/mcore_adapter/src/mcore_adapter/* /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/mcore_adapter/
cp /gfs/space/private/hehaowei/LLaMA-Factory/package_files/moe_utils.py /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/megatron/core/transformer/moe/moe_utils.py
cp /gfs/space/private/hehaowei/LLaMA-Factory/package_files/finalize_model_grads.py /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/megatron/core/distributed/finalize_model_grads.py
cp /gfs/space/private/hehaowei/LLaMA-Factory/package_files/router.py /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/megatron/core/transformer/moe/router.py
cp -r /gfs/space/private/hehaowei/LLaMA-Factory/megatron_core_patch/* /root/miniconda3/envs/llama_factory_py312/lib/python3.12/site-packages/megatron/core/
sleep 3
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1  USE_MCA=1 torchrun --nnodes=$GEMINI_TASKS_NUM --node_rank=$GEMINI_TASK_INDEX --master_addr=$GEMINI_HOST_IP_taskrole1_0 --master_port=29500 --nproc_per_node=8 src/train.py \
    --stage pt \
    --do_train \
    --train_from_scratch \
    --model_name_or_path DeepseekV32-6B-mca \
    --dataset fineweb_edu_350b \
    --finetuning_type full \
    --output_dir saves/DeepseekV32-6B-mca-dsa-mode1/pt/fineweb_edu_350b \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 5000 \
    --learning_rate 1.0e-5 \
    --warmup_steps 1000 \
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
    --sequence_parallel True \
    --experimental_attention_variant dsa \
    --dsa_indexer_n_heads 1 \
    --dsa_indexer_head_dim 128 \
    --dsa_indexer_topk 32 \
    --dsa_indexer_loss_coeff 0.1 \
    --report_to wandb \
    --run_name DeepseekV32-6B-mca-dsa-mode1_pt_fineweb_edu_350b
