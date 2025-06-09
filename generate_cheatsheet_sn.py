import json
import ast 
from utils_cheatsheet import *
import openai 
import time 
import random 

api_key = "9bee3459-3e28-47b9-b0e6-2e54b923ab49"
base_url = "https://api.sambanova.ai/v1"

dataset_path = "/import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/data/finlora/train/financebench_train.jsonl"

client = openai.OpenAI(api_key=api_key, base_url=base_url)

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

    response = client.chat.completions.create(
                    model="Llama-4-Maverick-17B-128E-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )

    response = response.choices[0].message.content
    new_cheatsheet = extract_cheatsheet(response, old_cheatsheet)
    old_cheatsheet = new_cheatsheet
    question_counts += 1

# save generated cheatsheet
open("generated_training_cheatsheets/financebench_cheatsheet_train_llama4_prompt.txt", "w+").write(new_cheatsheet)
