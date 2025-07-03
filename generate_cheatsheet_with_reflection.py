import argparse 
import ast 
import json
import openai
import random 
import re 
import time 
from metrics import qa_score 

from utils_gpt_v2 import *


#### API key information ####

api_key = ""
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

    for sample in all_samples:
        time.sleep(60) # to avoid hitting rate limits 
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
        
        response = client.chat.completions.create(
                    model=args.generator_model,
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.0
        )

        gen_response = response.choices[0].message.content
        final_answer = extract_answer(gen_response)

        if not relaxed_check_xbrl(final_answer, gt_answer): 
            for i in range(args.max_num_rounds):
                # reflect 
                #reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer)
                # reflection with gpt 
                reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer, old_cheatsheet)
            
                response = client.chat.completions.create(
                            model=args.reflector_model,
                            messages=[{"role": "user", "content": reflection_prompt}],
                            temperature=0.0
                )
                reflection = response.choices[0].message.content
             
                # generate after reflection 
                gen_prompt = generator_prompt.format(old_cheatsheet, reflection, question, context)
            
                response = client.chat.completions.create(
                            model=args.generator_model,
                            messages=[{"role": "user", "content": gen_prompt}],
                            temperature=0.0
                )
                gen_response = response.choices[0].message.content
                final_answer = extract_answer(gen_response)
                if relaxed_check_xbrl(final_answer, gt_answer): break 

        # generate cheatsheet
        cur_prompt = curator_prompt.format(old_cheatsheet, reflection, question, gen_response)
        response = client.chat.completions.create(
                    model=args.curator_model,
                    messages=[{"role": "user", "content": cur_prompt}],
                    temperature=0.0
        )
        response = response.choices[0].message.content
        new_cheatsheet = extract_cheatsheet(response, old_cheatsheet)
        old_cheatsheet = new_cheatsheet
        if question_counts % 10 == 0:
            # save generated cheatsheet
            open(f"{args.save_path}/trial_question_{question_counts}.txt", "w+").write(new_cheatsheet)
        question_counts += 1

if __name__=="__main__":
    main()
