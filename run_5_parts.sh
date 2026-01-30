#!/bin/bash

# Define common variables
CONFIG="examples/train_full/telechat3_pt_cpu_350B.yaml"
CACHE_DIR="./huggingface_cache"

# Loop to run 5 parts
for i in {1..5}
do
    DATASET="fineweb_edu_350b_part${i}"
    TOKENIZED_PATH="tokenized_path/fineweb_edu_350b_part${i}"
    OUTPUT_DIR="saves/telechat3-1p8b/pt/fineweb-edu-350b-cpu-part${i}"

    echo "Starting job for Part ${i}..."
    echo "Dataset: ${DATASET}"
    echo "Tokenized Path: ${TOKENIZED_PATH}"
    
    # Run in background (remove & to run sequentially)
    # Using 'nohup' so it persists if the shell closes, or just '&' if interactive.
    # To avoid OOM if running locally on one machine, you might want these sequential.
    # But since the user wants to distribute, I'll print the command they can potentialy adapt.
    
    # Actual command
    HF_DATASETS_CACHE="${CACHE_DIR}" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train \
        "${CONFIG}" \
        --dataset "${DATASET}" \
        --tokenized_path "${TOKENIZED_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        > "run_part_${i}.log" 2>&1 &
        
    echo "Job ${i} started. Log: run_part_${i}.log"
done

wait
echo "All jobs finished."
