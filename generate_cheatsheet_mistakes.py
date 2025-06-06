import json
import ast 
from utils_cheatsheet import *
import openai 
import time 

openai.api_type = "azure"
openai.api_key = ""
openai.api_base = "https://snova.openai.azure.com"
openai.api_version = "2024-12-01-preview"
openai.azure_endpoint="https://snova.openai.azure.com/"


# ========================================
dataset_path = "/import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/data/finlora/train/financebench_train.jsonl"

mistakes = [18, 32, 38,  7, 25, 82, 54, 62, 57, 76, 20, 64, 14, 36, 24, 70, 60,
       12, 27, 83, 15,  3, 21, 73,  5, 75, 84, 74, 28, 50, 51,  0, 44, 65,
       29, 45, 56, 40, 11, 72, 77, 10, 49, 71, 85, 67, 81,  9,  1, 63]

with open(dataset_path, 'r') as json_file:
    all_samples = list(json_file)


question_counts = 1

old_cheatsheet = "(empty)"

for index, sample in enumerate(all_samples):
    if index not in mistakes: 
        continue 
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
open("generated_training_cheatsheets/financebench_cheatsheet_train_gpt_4o_with_mistakes.txt", "w+").write(new_cheatsheet)
