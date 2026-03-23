import mcore_adapter.models.model_factory as model_factory
import time
import os
from mcore_adapter.models.model_factory import exists_hf_config, exists_mca_config, configure_resized_vocab_size, load_state_dict_from_checkpoint, VirtualModels, ModelConverter, save_config_and_state_dict, get_thd_data_on_this_cp_rank
from megatron.core import mpu
import torch
import warnings


def _patched_from_pretrained(
    cls,
    model_name_or_path: str,
    args=None,
    use_cpu_initialization: bool = False,
    tokenizer=None,
) -> "VirtualModels":
    load_start_time = time.time()
    config = cls.config_class.from_pretrained(model_name_or_path, args)
    # if int(os.getenv("RANK", "0")) == 0:
    #     print(f"Saving final MLATransformerConfig parameters to 30B-full-feature-config.json")
    #     with open("30B-full-feature-config.json", "w", encoding="utf-8") as f:
    #         f.write(config.to_json_string())

    config.use_cpu_initialization = use_cpu_initialization

    resized_vocab_size = None
    if tokenizer is not None:
        resized_vocab_size = configure_resized_vocab_size(config.padded_vocab_size, len(tokenizer))
        if resized_vocab_size:
            config.padded_vocab_size = resized_vocab_size

    models = VirtualModels(cls, config=config)

    print(
        f"number of parameters on (tensor, pipeline, expert) model parallel rank "
        f"({mpu.get_tensor_model_parallel_rank()}, {mpu.get_pipeline_model_parallel_rank()}, "
        f"{mpu.get_expert_model_parallel_rank()}): {sum(p.nelement() for p in models.parameters())}"
    )

    mca_ckpt_exist = exists_mca_config(model_name_or_path)
    dist_config_match = False
    if mca_ckpt_exist:
        old_mca_config = cls.config_class.from_pretrained(model_name_or_path)
        dist_config_match = config.distribute_config_match(old_mca_config)

    # Patch: allow passing if args.train_from_scratch is true, bypass loading states and errors
    if args is not None and getattr(args, "train_from_scratch", False):
        print("Bypassing state dict load due to train_from_scratch")
        return models

    if mca_ckpt_exist and dist_config_match:
        if resized_vocab_size:
            raise ValueError(
                "The tokenizer length is longer than the vocab embedding size, and the resize embedding"
                "layer is not supported loading mca ckpt. Please check the tokenizer and ckpt."
            )
        state_dict = load_state_dict_from_checkpoint(model_name_or_path)
    else:
        if not exists_hf_config(model_name_or_path):
            raise ValueError(
                f"{model_name_or_path} is not valid for current training, because not exists hf ckpt "
                f"and not mca_ckpt_exist: {mca_ckpt_exist} or not dist_config_match: {dist_config_match}"
            )
        state_dict = {}
        converter = ModelConverter(config, resized_vocab_size=resized_vocab_size)
        for i in range(len(models)):
            key = "model"
            if len(models) > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                key = f"{key}{i}"
            try:
                state_dict[key] = converter.load_mca_state_dict_from_hf(model_name_or_path, vp_stage=i)
            except Exception as e:
                print(f"Failed to load HF state dict for stage {i}: {e}. Initializing randomly.")
                continue

    missing_keys, unexpected_keys = models.load_state_dict(state_dict, strict=False)
    if missing_keys:
        missing_keys = [key for key in missing_keys if not key.endswith("._extra_state")]
    if unexpected_keys and config.tie_embeddings_and_output_weights:
        unexpected_keys = [key for key in unexpected_keys if not key.endswith("output_layer.weight")]
    
    # Patch step: downgrade assert to warning
    if unexpected_keys:
        warnings.warn(f"unexpected_keys: {unexpected_keys}")
    if missing_keys:
        warnings.warn(f"missing_keys: {missing_keys}")
        
    print(f"End loading, cost: {time.time() - load_start_time:0.3f}s")
    return models


def apply_mcore_patch():
    # Patch the `from_pretrained` class method on the PretrainedModel class via model_factory
    model_factory.PretrainedModel.from_pretrained = classmethod(_patched_from_pretrained)
    
