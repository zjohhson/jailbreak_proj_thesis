from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download
import subprocess
import yaml

with open('./config.yaml', "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
hf_token = config["HUGGINGFACE_TOKEN"]
subprocess.run(["huggingface-cli", "login", "--token", hf_token])

model_id = "meta-llama/LlamaGuard-7b"
# hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
# device = "cuda"
dtype = torch.bfloat16
# print(torch.cuda.is_available())
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

tokenizer.save_pretrained("../models/llama_guard")
model.save_pretrained("../models/llama_tokenizer")