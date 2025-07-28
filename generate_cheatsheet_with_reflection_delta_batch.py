import argparse 
import ast 
import json
import openai
import random 
import re 
import time 
import os
from metrics import qa_score 

from utils_claude import *


#### API key information ####

api_key = os.environ['SAMBANOVA_API_KEY']
base_url = "https://api.sambanova.ai/v1"

###---------------------####


def extract_cheatsheet(
    response: str,
    old_cheatsheet: str,
) -> str:
    """
    Extracts the cheatsheet from the model response.

    Arguments:
        response : str : The response from the model.
        old_cheatsheet : str : The old cheatsheet to return if the new one is not found.
    Returns:
        str : The extracted cheatsheet (if not found, returns the old cheatsheet).
    """
    response = response.strip()
    # <cheatsheet> (content) </cheatsheet>
    if "<cheatsheet>" in response:
        try:
            txt = response.split("<cheatsheet>")[1].strip()
            txt = txt.split("</cheatsheet>")[0].strip()
            return txt
        except:
            return old_cheatsheet
    else:
        return old_cheatsheet

def extract_answer(
    response: str,
) -> str:
    """
    Extracts the final answer from the model response.

    Arguments:
        response : str : The response from the model.

    Returns:
        str : The extracted final answer (if not found, returns "No final answer found").
    """


    matches = re.findall(r"Finish\[(.*?)\]", response)

    if matches:
        last_answer = matches[-1]
        return last_answer
    else:
        return "No final answer found"


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--curator_model", type=str, default="Llama-4-Maverick-17B-128E-Instruct")
    parser.add_argument("--reflector_model", type=str, default="Llama-4-Maverick-17B-128E-Instruct")
    parser.add_argument("--generator_model", type=str, default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args 

def initialize_client():
    # SAMBANOVA client 
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client 

def relaxed_check_xbrl(final_answer, gt_answer):
    pred = final_answer.split(",")
    label = gt_answer.split(",")
    label = [val.lower().strip() for val in label]
    count = 0 
    for prediction in pred: 
        prediction  = prediction.lower().strip()
        if prediction in label: count += 1
    score = count/len(pred)
    if score > 0.5: return True 

def relaxed_check(final_answer, gt_answer):
    # used for financebench 
    score = qa_score(final_answer, gt_answer)
    if score > 0.4: return True 
    return False 

def api_with_backoff(client, model, prompt):
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        # Calculate exponential backoff (base 2)
        sleep_time = min(2 ** retry_count, 60)  # Cap at 60 seconds to avoid too long waits
        print(f"Attempt {retry_count + 1} failed. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        retry_count += 1
    return response

def main():
    args = parse_args()
    client = initialize_client()

    with open(args.dataset_path, 'r') as json_file:
        all_samples = list(json_file)

    if args.num_samples != -1: 
        random.seed(42)
        random.shuffle(all_samples)
        all_samples = all_samples[:args.num_samples]

    question_counts = 1

    old_cheatsheet = "(empty)"

    batch_size = args.batch_size
    batch_index = 1
    for sample in all_samples:
        if batch_index > batch_size:
            old_cheatsheet = "(empty)"
            batch_index = 1
        # time.sleep(60) # to avoid hitting rate limits 
        print(f"==========processing question {question_counts}==========")
        task_dict = ast.literal_eval(sample)
        reflection = "(empty)"
        all_context  = task_dict["context"]

        # FinanceBench 
        #question, context = all_context.split("\nDocument Pages Context")

        # XBRL finer 
        index = all_context.index("Answer the following 4 independent questions by providing only")
        context, question = all_context[:index], all_context[index:]

        gt_answer = task_dict["target"]
   
        # get answer from generator 
        gen_prompt = generator_prompt.format(old_cheatsheet, reflection, question, context)
        
        response = api_with_backoff(client, args.generator_model, gen_prompt)

        gen_response = response.choices[0].message.content
        final_answer = extract_answer(gen_response)

        if not relaxed_check_xbrl(final_answer, gt_answer): 
            for i in range(args.max_num_rounds):
                # reflect 
                reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer)
                # reflection with gpt 
                # reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer, old_cheatsheet)
            
                response = api_with_backoff(client, args.reflector_model, reflection_prompt)
                reflection = response.choices[0].message.content
             
                # generate after reflection 
                gen_prompt = generator_prompt.format(old_cheatsheet, reflection, question, context)
            
                response = api_with_backoff(client, args.generator_model, gen_prompt)
                gen_response = response.choices[0].message.content
                final_answer = extract_answer(gen_response)
                if relaxed_check_xbrl(final_answer, gt_answer): break 

        # generate cheatsheet
        cur_prompt = curator_prompt.format(old_cheatsheet, reflection, question, gen_response)
        response = api_with_backoff(client, args.curator_model, cur_prompt)
        response = response.choices[0].message.content
        new_cheatsheet = extract_cheatsheet(response, old_cheatsheet)
        old_cheatsheet = new_cheatsheet
        if question_counts % batch_size == 0:
            # save generated cheatsheet
            save_path = f"{args.save_path}/batch_size_{args.batch_size}/trial_question_{question_counts}.txt"
            dir_path = os.path.dirname(save_path)
            os.makedirs(dir_path, exist_ok=True)
            open(save_path, "w+").write(new_cheatsheet)
        question_counts += 1
        batch_index += 1

if __name__=="__main__":
    main()
