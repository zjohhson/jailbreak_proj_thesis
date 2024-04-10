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

def main():
    embedding_vecs = get_vecs('../data/pca_data.csv')
    labels = get_labels('../data/pca_data.csv')
    pca, scaler = do_pca(embedding_vecs)
    scaled = scaler.transform(np.array(embedding_vecs))
    transformed_data = pca.transform(scaled)
    xPCA = list(transformed_data[:,0])
    yPCA = list(transformed_data[:,1])
    svm, X, Y = get_svm_classifier(xPCA, yPCA, labels)
    responses_df = pd.read_csv('../data/testing_models_llama.csv')
    responses = list(responses_df['Response'])
    prompts = list(responses_df['Role-play Prompt'])
    requests = list(responses_df['Request'])

    results_df = pd.DataFrame({
       'Role-play Prompt': [],
       'Request': [],
       'True Label': []
       })
    
    for i in range(len(responses)):
      response = responses[i]
      request = requests[i]
      prompt = prompts[i]
      response_embedding = get_embedding(response)
      scaled_embedding = scaler.transform(np.array([response_embedding]))
      coords = pca.transform(scaled_embedding)[0]
      x, y = coords[0], coords[1]
      eval_rating = svm.predict([[x,y]])[0]
      print(f"Evaluation: {eval_rating}")
      results_df.loc[i] = [
              prompt,
              request,
              eval_rating
            ]

    results_df.to_csv(f'../data/llama_evaluation.csv')
    
if __name__ == "__main__":

    main()