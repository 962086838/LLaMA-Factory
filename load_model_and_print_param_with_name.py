from transformers import AutoModelForCausalLM
import torch


def load_and_print_model_param(checkpoint_path):
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)

    # 打印模型结构（可选，用于调试）
    # print("Model architecture:")
    # print(model)
    for name, param in model.named_parameters():
        # print(name)
        if name == "transformer.h.21.mlp.local_experts.0.down_proj.bias":
            print(f"{param}, {param.shape}")
    # 获取指定参数
    # target_param = model.module.transformer.h.22.self_attention.dense.weight

    # 打印参数信息
    # print("\nParameter shape:", target_param.shape)
    # print("Parameter values:")
    # print(target_param)


if __name__ == "__main__":
    checkpoint_path = "/gemini/space/thu/hehaowei/LLaMA-Factory/saves/TeleChat-1B-HWMoE-exp2/full/pt_exp2_16machine_z2/checkpoint-22000"
    load_and_print_model_param(checkpoint_path)