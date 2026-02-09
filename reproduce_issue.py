from transformers import AutoConfig, AutoModelForCausalLM
import os
import sys

# Ensure the current directory is in sys.path if needed, but AutoModel should handle it
model_path = "/gfs/space/private/hehaowei/LLaMA-Factory/Telechat3-1p8B-mhyperconnection"

print(f"Loading model from {model_path}")

try:
    print("Loading Config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("Config loaded successfully")
    
    print("Loading Model...")
    # We use from_config to mimic the error path more closely if possible, or just from_pretrained
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    print("Model loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
