import json
import ast 
from utils_cheatsheet import *
import openai 
import time 
import random 

openai.api_type = "azure"
openai.api_key = ""
openai.api_base = "https://snova.openai.azure.com"
openai.api_version = "2024-12-01-preview"
openai.azure_endpoint="https://snova.openai.azure.com/"


# ========================================
dataset_path = "/import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/data/finlora/train/financebench_train.jsonl"


with open(dataset_path, 'r') as json_file:
    all_samples = list(json_file)

random.shuffle(all_samples)
all_samples = all_samples[:50]

question_counts = 1

old_cheatsheet = "(empty)"

for sample in all_samples:
    time.sleep(60)
    print(f"==========processing question {question_counts}==========")
    task_dict = ast.literal_eval(sample)
    all_context  = task_dict["context"]
    question, context = all_context.split("\nDocument Pages Context")
    gt_answer = task_dict["target"]
   
    # generate cheatsheet
    prompt = cheatsheet_gen_prompt.format(old_cheatsheet, question, context, gt_answer)

    response = openai.ChatCompletion.create(
                    engine="Internal_Copilot",
                    messages=[
                      {
                         "role": "user",
                         "content": prompt
                      }
                    ],
                temperature=0.0
    )

    response = response.choices[0].message.content
    new_cheatsheet = extract_cheatsheet(response, old_cheatsheet)
    old_cheatsheet = new_cheatsheet
    question_counts += 1

# save generated cheatsheet
open("generated_training_cheatsheets/financebench_cheatsheet_train_gpt_4.1.txt", "w+").write(new_cheatsheet)
