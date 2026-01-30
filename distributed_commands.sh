# Part 1
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/telechat3_pt_cpu_350B_part1.yaml

# Part 2
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/telechat3_pt_cpu_350B_part2.yaml

# Part 3
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/telechat3_pt_cpu_350B_part3.yaml

# Part 4
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/telechat3_pt_cpu_350B_part4.yaml

# Part 5
HF_DATASETS_CACHE="./huggingface_cache" USE_MODELSCOPE_HUB=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/telechat3_pt_cpu_350B_part5.yaml
