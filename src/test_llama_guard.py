from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import yaml
import subprocess
import pandas as pd

huggingface_bin_path = "/home/user/.local/bin"
os.environ["PATH"] = f"{huggingface_bin_path}:{os.environ['PATH']}"
with open('./config.yaml', "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
hf_token = config["HUGGINGFACE_TOKEN"]
subprocess.run(["huggingface-cli", "login", "--token", hf_token])

def main():

    model_id = "meta-llama/LlamaGuard-7b"
    device = "cuda"
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained("../models/llama_tokenizer")
    model = AutoModelForCausalLM.from_pretrained("../models/llama_guard", torch_dtype=dtype, device_map=device)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    df = pd.read_csv('../data/pca_data.csv')

    request = list(df['prompt'])
    response = list(df['response'])
    labels = list(df['label'])

    def moderate_with_template(chat):
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        print('here!')
        prompt_len = input_ids.shape[-1]
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    results_df = pd.DataFrame({
       'Request': [],
       'Response': [],
       'Estimate': [],
       'Label': []
       })
    
    for i in range(len(request[:1])):
        req = request[i]
        res = response[i]
        label = labels[i]
        chat = [
            {"role": "user", "content": f"{req}"},
            {"role": "assistant", "content": f"{res}"},
        ]
        rating = moderate_with_template(chat)
        score = 0
        if 'unsafe' in rating:
            score = 1
        else:
            score = 0

        results_df.loc[i] = [
              req,
              res,
              score,
              label
            ]
    results_df.to_csv(f'../data/LlamaGuard_ratings.csv')

if __name__ == "__main__":
    main()