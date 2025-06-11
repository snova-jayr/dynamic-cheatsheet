import argparse 
import ast 
import json
import openai
import random 
import time 

from utils_reflection import *


#### API key information ####

api_key = "9bee3459-3e28-47b9-b0e6-2e54b923ab49"
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
    if "<answer>" in response:
        # <answer> (content) </answer>
        try:
            txt = response.split("<answer>")[-1].strip()
            txt = txt.split("</answer>")[0].strip()
            return txt
        except:
            return "No final answer found"
    else:
        if not("FINAL ANSWER" in response):
            return "No final answer found"
        try:
            response = response.split("FINAL ANSWER")[-1].strip()
            if response[0] == ":":
                response = response[1:].strip()

            # First decide whether to split by "```" or "'''" based on the presence of "```" or "'''"
            idx_1 = response.find("'''")
            idx_2 = response.find("```")
            if min(idx_1, idx_2) != -1:
                if idx_1 < idx_2:
                    response = response.split("'''")[1].strip()
                else:
                    response = response.split("```")[1].strip()
            else:
                if idx_1 == -1:
                    response = response.split("```")[1].strip()
                else:
                    response = response.split("'''")[1].strip()

            # Special case for P3-Test task: If the first line contains "python" then remove it
            if response.split("\n")[0].strip().lower() == "python":
                response = "\n".join(response.split("\n")[1:]).strip()
            return response
        except:
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
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client 


def main():
    args = parse_args()
    client = initialize_client()

    with open(args.dataset_path, 'r') as json_file:
        all_samples = list(json_file)

    question_counts = 1

    old_cheatsheet = "(empty)"

    for sample in all_samples:
        time.sleep(60) # to avoid hitting rate limits 
        print(f"==========processing question {question_counts}==========")
        task_dict = ast.literal_eval(sample)
        reflection = "(empty)"
        all_context  = task_dict["context"]
        question, context = all_context.split("\nDocument Pages Context")
        gt_answer = task_dict["target"]
   
        # get answer from generator 
        gen_prompt = generator_prompt.format(old_cheatsheet, reflection, context, question)

        response = client.chat.completions.create(
                    model=args.generator_model,
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.0
        )

        gen_response = response.choices[0].message.content
        final_answer = extract_answer(gen_response)
        
        if final_answer != gt_answer: 
            for i in range(args.max_num_rounds):
                # reflect 
                reflection_prompt = reflector_prompt.format(gen_response, question, context, final_answer, gt_answer)
                response = client.chat.completions.create(
                            model=args.reflector_model,
                            messages=[{"role": "user", "content": reflection_prompt}],
                            temperature=0.0
                )
                reflection = response.choices[0].message.content
             
                # generate after relfection 
                gen_prompt = generator_prompt.format(old_cheatsheet, reflection, context, question)
                response = client.chat.completions.create(
                            model=args.generator_model,
                            messages=[{"role": "user", "content": gen_prompt}],
                            temperature=0.0
                )
                gen_response = response.choices[0].message.content
                final_answer = extract_answer(gen_response)
                if final_answer == gt_answer: break 

        # generate cheatsheet
        cur_prompt = curator_prompt.format(old_cheatsheet, question, context, gen_response)

        response = client.chat.completions.create(
                    model=args.curator_model,
                    messages=[{"role": "user", "content": cur_prompt}],
                    temperature=0.0
        )

        response = response.choices[0].message.content
        new_cheatsheet = extract_cheatsheet(response, old_cheatsheet)
        old_cheatsheet = new_cheatsheet
        question_counts += 1
        break

    # save generated cheatsheet
    open(args.save_path, "w+").write(new_cheatsheet)


if __name__=="__main__":
    main()
