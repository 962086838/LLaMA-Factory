#!/bin/bash

# Check if the environment variable is set
if [ -z "${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX}" ]; then
    echo "Error: GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX is not set."
    echo "This script expects to run in an environment where this variable is defined (0-20)."
    exit 1
fi

# Calculate Part ID (0-based index -> 1-based part id)
PART_ID=$((GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX + 1))

echo "Running on node index: ${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX}"
echo "Selected Part ID: ${PART_ID}"

CONFIG_FILE="examples/train_full/telechat3_pt_cpu_350B_part${PART_ID}.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file ${CONFIG_FILE} does not exist."
    echo "Please run 'python generate_part_configs.py' first."
    exit 1
fi

# Execute the training command
echo "Executing: llamafactory-cli train ${CONFIG_FILE}"

HF_DATASETS_CACHE="./huggingface_cache" \
USE_MODELSCOPE_HUB=1 \
DISABLE_VERSION_CHECK=1 \
llamafactory-cli train "${CONFIG_FILE}"
