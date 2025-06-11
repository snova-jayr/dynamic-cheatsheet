import argparse 
import json
import ast 
from utils_cheatsheet import *
import openai 
import time 
import random 

#### API key information ####

openai.api_type = "azure"
openai.api_key = ""
openai.api_base = "https://snova.openai.azure.com"
openai.api_version = "2024-12-01-preview"
openai.azure_endpoint="https://snova.openai.azure.com/"

###---------------------####


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.dataset_path, 'r') as json_file:
        all_samples = list(json_file)

    if args.num_samples != -1: 
        random.shuffle(all_samples)
        all_samples = all_samples[:args.num_samples]

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
    open(args.save_path, "w+").write(new_cheatsheet)

if __name__ == "__main__":
    main()
