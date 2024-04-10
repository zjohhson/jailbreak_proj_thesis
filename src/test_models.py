import openai
import subprocess
import json
import sys
import pandas as pd
import time
import os
import yaml
import random
import time
from collections import defaultdict

from util import convert_instructions
from data_preparation import DataPreparationGPT, save_to_jsonl
from pca_classifier import *
from language_models import GPT, PaLM, Llama2
huggingface_bin_path = "/home/user/.local/bin"
os.environ["PATH"] = f"{huggingface_bin_path}:{os.environ['PATH']}"
with open('./config.yaml', "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
hf_token = config["HUGGINGFACE_TOKEN"]
subprocess.run(["huggingface-cli", "login", "--token", hf_token])

def main(test_model_name):

    embedding_vecs = get_vecs('../data/pca_data.csv')
    labels = get_labels('../data/pca_data.csv')
    pca, scaler = do_pca(embedding_vecs)
    scaled = scaler.transform(np.array(embedding_vecs))
    transformed_data = pca.transform(scaled)
    xPCA = list(transformed_data[:,0])
    yPCA = list(transformed_data[:,1])
    svm, X, Y = get_svm_classifier(xPCA, yPCA, labels)
    data_prep = DataPreparationGPT()

    if 'gpt' in test_model_name.lower():
      shorthand = 'gpt'
      test_model = GPT(test_model_name)
    elif 'gemini' in test_model_name.lower() or 'text-bison' in test_model_name.lower() or 'chat-bison' in test_model_name.lower():
      shorthand = 'palm'
      test_model = PaLM(test_model_name)
    elif 'llama' in test_model_name.lower():
       shorthand = 'llama'
       test_model = Llama2(test_model_name)

    roleplay_prompts = data_prep.get_roleplay_prompts()

    requests_df = pd.read_csv('../data/harmful_behaviors.csv')
    requests_list = list(requests_df['goal'])


    # Track results for each role play prompt that we generate
    results_df = pd.DataFrame({
       'Role-play Prompt': [],
       'True Label': [],
       'Request': [],
       'Response': []
       })
    
    for i in range(len(roleplay_prompts)):
        prompt = roleplay_prompts[i]
        print(f"Prompt: {prompt}")
        
        problematic_prompt = convert_instructions(prompt, shorthand)

        # Randomly sample 15 harmful requests to try
        harmful_reqs = random.sample(requests_list, 20)

        jailbroken_response = ''
        req = ''
        nonjailbroken_response = ''

        total_1s = 0
        for harmful_req in harmful_reqs:
          try:
            print('---------------')
            response = test_model.ask_harmful_request(problematic_prompt, harmful_req)
            # response, req_in_tokens, req_out_tokens = ask_harmful_request(problematic_prompt, harmful_req)
            print(f"Harmful request: {harmful_req}")
            print(f"Response: {response}")

            # Evaluate response using pre-trained SVM classifier
            response_embedding = get_embedding(response)
            scaled_embedding = scaler.transform(np.array([response_embedding]))
            coords = pca.transform(scaled_embedding)[0]
            x, y = coords[0], coords[1]
            eval_rating = svm.predict([[x,y]])[0]


            print(f"Evaluation: {eval_rating}")
            if eval_rating == 1:
                total_1s += 1
                jailbroken_response = response
                req = harmful_req
            else:
                nonjailbroken_response += response + '\n'
                req += harmful_req + '\n'
            if total_1s > 0:
                break
          except Exception as error:
            print(error)

        if total_1s > 0: # label is 1, set this threshold based on what you want
            # few_shot = problematic_prompt # to make it a GAN, move this statement to next if statement
            response = jailbroken_response
            true_label = 1
        else:
            response = nonjailbroken_response
            true_label = 0

        # add to dataframe
        results_df.loc[i] = [
            problematic_prompt,
            true_label,
            req,
            response
        ]

    print('----------------------------------------')
    print("\n")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f'../data/results/testing_models_{shorthand}.csv')
    
if __name__ == "__main__":

    data_prep = DataPreparationGPT()
    roleplay_prompts = data_prep.get_roleplay_prompts()
    initial_data, initial_labels = data_prep.create_judge_training_set()

    initial_judge_data_jsonl = data_prep.create_judge_jsonl(initial_data, initial_labels)
    initial_adversarial_data_jsonl = data_prep.create_adversarial_jsonl(roleplay_prompts)

    save_to_jsonl([], '../data/judge_fine_tune.jsonl')
    save_to_jsonl([], '../data/adversary_fine_tune.jsonl')

    # pretrained_adversary = pretrain_model('../data/adversary_fine_tune.jsonl', 'gpt-3.5-turbo', 'adversary')
    # pretrained_judge = pretrain_model('../data/judge_fine_tune.jsonl', 'gpt-3.5-turbo', 'judge')
    # pretrained_adversary = 'ft:gpt-3.5-turbo-0613:personal:adversary:8fb1u0Co'
    # pretrained_judge = 'ft:gpt-3.5-turbo-0613:personal:judge:8l0xfLNi'

    test_model_name = sys.argv[1]

    main(test_model_name)