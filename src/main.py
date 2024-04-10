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

from judge import load_judge
from adversary import load_adversary
from util import similar_prompt_exists, noncompliant, convert_instructions
# from conversers import ask_harmful_request
from data_preparation import DataPreparationGPT, csv_to_list, append_to_jsonl, save_to_jsonl
from pca_classifier import *
from language_models import GPT, PaLM

def pretrain_model(file_name, base_model, suffix):
    if suffix == 'judge':
      lm = load_judge(base_model)
    elif suffix == 'adversary':
      lm = load_adversary(base_model)
    else:
       raise Exception("Suffix must be judge or adversary!")
    return lm.fine_tune(file_name)

def main(iterations, fine_tuning_iterations, adversarial_model_name, judge_model_name, test_model_name):
    counter = 0 # change
    embedding_vecs = get_vecs('../data/pca_data.csv')
    labels = get_labels('../data/pca_data.csv')
    pca, scaler = do_pca(embedding_vecs)
    scaled = scaler.transform(np.array(embedding_vecs))
    transformed_data = pca.transform(scaled)
    xPCA = list(transformed_data[:,0])
    yPCA = list(transformed_data[:,1])
    svm, X, Y = get_svm_classifier(xPCA, yPCA, labels)
    data_prep = DataPreparationGPT()
    few_shot = None # few shot prompting for adversary

    if 'gpt' in test_model_name.lower():
      shorthand = 'gpt'
      test_model = GPT(test_model_name)
    elif 'gemini' in test_model_name.lower() or 'text-bison' in test_model_name.lower() or 'chat-bison' in test_model_name.lower():
      shorthand = 'palm'
      test_model = PaLM(test_model_name)
    # elif 'claude' in test_model_name.lower():
    #   test_model = Claude(test_model_name)
    #   shorthand = 'claude'

    roleplay_prompts = data_prep.get_roleplay_prompts()
    # initial_data, initial_labels = data_prep.create_judge_training_set()

    # initial_judge_data_jsonl = data_prep.create_judge_jsonl(initial_data, initial_labels)
    # initial_adversarial_data_jsonl = data_prep.create_adversarial_jsonl(roleplay_prompts)

    # save_to_jsonl(initial_judge_data_jsonl, '../data/adversary_fine_tune.jsonl')
    # save_to_jsonl(initial_adversarial_data_jsonl, '../data/judge_fine_tune.jsonl')

    adversary_jsonl_ft = []
    adversarial_generations = []

    requests_df = pd.read_csv('../data/harmful_behaviors.csv')
    requests_list = list(requests_df['goal'])

    ####
    # plot_svm_boundary(xPCA, yPCA, labels)
    ###

    # Track results for each role play prompt that we generate
    results_df = pd.DataFrame({
       'Fine-Tuning Iteration': [],
       'Role-play Prompt': [],
       'Judge Guess': [],
       'True Label': [],
       'Time': [],
       'Input Tokens': [],
       'Output Tokens': []
       })
    
    approved_reqs_df = pd.DataFrame({
       'Approved Requests': []
    })

    denied_reqs_df = pd.DataFrame({
       'Denied Requests': []
    })

    adversaryLM = load_adversary(adversarial_model_name)
    judgeLM = load_judge(judge_model_name)
    
    for ft_iteration in range(fine_tuning_iterations):
        print('here')
        print(f"Fine-tuning iteration {ft_iteration + 1}")

        judge_jsonl_ft = []

        for i in range(iterations):
            few_shot = random.sample(roleplay_prompts, 3)
            start = time.time()
            print(f"Round {i+1}:")

            results_df_row = ft_iteration * iterations + i

            total_in_tokens = 0
            total_out_tokens = 0

            # Adversarial model generates a problematic prompt
            is_noncompliant = True
            is_similar = True
            if not len(adversarial_generations): # ensuring we don't repeat prompts or add noncompliant responses
               while is_noncompliant:
                  problematic_prompt, problematic_in_tokens, problematic_out_tokens = adversaryLM.generate_problematic_prompt(few_shot)
                  print('1', problematic_prompt)
                  is_noncompliant = noncompliant(problematic_prompt)
            while is_noncompliant or is_similar:
               print('2', problematic_prompt)
               problematic_prompt, problematic_in_tokens, problematic_out_tokens = adversaryLM.generate_problematic_prompt(few_shot)
               is_noncompliant = noncompliant(problematic_prompt)
               is_similar = similar_prompt_exists(problematic_prompt, adversarial_generations)
            problematic_prompt = convert_instructions(problematic_prompt, shorthand)
            adversarial_generations.append(problematic_prompt)
            total_in_tokens += problematic_in_tokens
            total_out_tokens += problematic_out_tokens
            print(f"Problematic prompt: {problematic_prompt}")

            judge_answer, judge_in_tokens, judge_out_tokens = judgeLM.judge_prompt(problematic_prompt)
            total_in_tokens += judge_in_tokens
            total_out_tokens += judge_out_tokens

            # TO DO: turn this into a helper function that is specific to LM
            try: # Sometimes format of output is not uniform, making sure this doesn't cause exception and stop code
              if len(judge_answer) > 2:
                judge_rating = int(judge_answer.split()[1])
              else:
                judge_rating = int(judge_answer)
            except:
              print('3', judge_answer)
              pass
            print(f"Judge's answer: {judge_answer}")
            print(judge_rating)

            # Randomly sample 15 harmful requests to try
            harmful_reqs = random.sample(requests_list, 20)

            approved_reqs = []
            denied_reqs = []
            total_1s = 0
            for harmful_req in harmful_reqs:
              try:
                print('---------------')
                response, req_in_tokens, req_out_tokens = test_model.ask_harmful_request(problematic_prompt, harmful_req)
                # response, req_in_tokens, req_out_tokens = ask_harmful_request(problematic_prompt, harmful_req)
                total_in_tokens += req_in_tokens
                total_out_tokens += req_out_tokens
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
                  approved_reqs.append(harmful_req)
                else:
                  denied_reqs.append(harmful_req)
                if total_1s > 0:
                  break
              except Exception as error:
                print(error)

            if total_1s > 0: # label is 1, set this threshold based on what you want
                # few_shot = problematic_prompt # to make it a GAN, move this statement to next if statement
                roleplay_prompts.append(problematic_prompt)
                true_label = 1
                data_prep.append_to_judge_jsonl(problematic_prompt, 1, judge_jsonl_ft)
                if judge_rating == 0:
                    data_prep.append_to_adversarial_jsonl(problematic_prompt, adversary_jsonl_ft)
                # else:
                  # judge_correct += 1
            else:
                true_label = 0
                data_prep.append_to_judge_jsonl(problematic_prompt, 0, judge_jsonl_ft)
                # data_prep.append_to_judge_jsonl(roleplay_prompts[counter], 1, judge_jsonl_ft)
                counter += 1
                # if judge_rating == 0:
                  # judge_correct += 1
                
            end = time.time()

            # add to dataframe
            results_df.loc[results_df_row] = [
              ft_iteration,
              problematic_prompt,
              judge_rating,
              true_label,
              end - start, 
              total_in_tokens,
              total_out_tokens
            ]

            approved_reqs_df.loc[results_df_row] = [
               approved_reqs
            ]

            denied_reqs_df.loc[results_df_row] = [
               denied_reqs
            ]

        # if ft_iteration == 0: # for when we are using gpt-3.5-turbo as base, not pretrained
        #    adversaryLM = load_adversary(adversaryLM.fine_tune('../data/adversary_fine_tune.jsonl'))
        # else:
        # if len(adversary_jsonl_ft) >= 10:
        #   append_to_jsonl(adversary_jsonl_ft, '../data/adversary_fine_tune.jsonl')
        #   adversaryLM = load_adversary(adversaryLM.fine_tune('../data/adversary_fine_tune.jsonl'))
        #   adversary_jsonl_ft = []
        # if ft_iteration == 0:
        #    judgeLM = load_judge(judgeLM.fine_tune('../data/judge_fine_tune.jsonl'))
        # else:
        append_to_jsonl(judge_jsonl_ft, '../data/judge_fine_tune.jsonl')
        judgeLM = load_judge(judgeLM.fine_tune('../data/judge_fine_tune.jsonl'))

        print('----------------------------------------')
        print("\n")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f'../data/results/{timestr}.csv')
    approved_reqs_df.to_csv(f'../data/results/{timestr}_approved.csv')
    denied_reqs_df.to_csv(f'../data/results/{timestr}_denied.csv')
    
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

    pretrained_adversary = 'gpt-4-turbo-preview'
    pretrained_judge = 'gpt-4-turbo-preview'
    test_model_name = 'gpt-4-turbo-preview'

    main(15, 10, pretrained_adversary, pretrained_judge, test_model_name)