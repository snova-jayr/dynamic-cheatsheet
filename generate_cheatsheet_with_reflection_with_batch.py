import argparse 
import ast 
import json
import asyncio 
import openai
import random 
import re 
import time 
from metrics import qa_score 

from utils_claude_batched  import *


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
    parser.add_argument("--reflector_aggregator_model", type=str, default="Llama-4-Maverick-17B-128E-Instruct")
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
    return False 

def relaxed_check(final_answer, gt_answer):
    # used for financebench 
    score = qa_score(final_answer, gt_answer)
    if score > 0.4: return True 
    return False 

async def call_openai_chat(client, sample, args):
    return client.chat.completions.create(
                 model=args.generator_model,
                messages=sample,
                temperature=0.0
    )

async def run_batch(client, all_gen_queries, args):
    responses = await asyncio.gather(*(call_openai_chat(client, sample, args) for sample in all_gen_queries))
    return [r.choices[0].message.content for r in responses]

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

    counter = 0 

    while counter < len(all_samples):
        time.sleep(60) # to avoid hitting rate limits
        batch = all_samples[counter: min(counter + args.batch_size, len(all_samples))]
        print(f"==========processing batch {question_counts}==========")
        batched_reflection = ["(empty)" for i in range(args.batch_size)]
        
        # XBRL finer
        all_gen_queries = []
        all_questions = []
        all_contexts = []
        all_gt_answers = []

        for i, sample in enumerate(batch):
            task_dict = ast.literal_eval(sample)
            all_context = task_dict["context"]
            index = all_context.index("Answer the following 4 independent questions by providing only")
            context, question = all_context[:index], all_context[index:]
            gt_answer = task_dict["target"]
            all_questions.append(question)
            all_contexts.append(context)
            all_gt_answers.append(gt_answer)
            reflection = batched_reflection[i]
            # get answer from generator 
            gen_prompt = generator_prompt.format(old_cheatsheet, reflection, question, context)
            all_gen_queries.append([{"role": "user", "content": gen_prompt}])

        gen_responses = asyncio.run(run_batch(client, all_gen_queries, args))
        final_answers = [extract_answer(gen_response) for gen_response in gen_responses] 
        batched_pass_scores = [relaxed_check_xbrl(final_answer, gt_answer)  for final_answer, gt_answer in zip(final_answers, all_gt_answers)]

        for i in range(args.batch_size):
            
            if batched_pass_scores[i]: continue
            
            question, gt_answer, final_answer, gen_response = all_questions[i], all_gt_answers[i], final_answers[i], gen_responses[i]

            for j in range(args.max_num_rounds):
                # reflect 
                reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer)
                # reflection with gpt 
                #reflection_prompt = reflector_prompt.format(question, gen_response, final_answer, gt_answer, old_cheatsheet)
            
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
        
            batched_reflection[i] = reflection 
        
        # reflector aggregator 
        reflection_aggregator_prompt = reflector_aggregator_prompt.format(batched_reflection[0], batched_reflection[1], batched_reflection[2], batched_reflection[3])
    
        response = client.chat.completions.create(
                        model=args.reflector_aggregator_model,
                        messages=[{"role": "user", "content": reflection_aggregator_prompt}],
                        temperature=0.0
        )
        aggregated_reflection = response.choices[0].message.content
        
        # generate cheatsheet
        cur_prompt = curator_prompt.format(old_cheatsheet, aggregated_reflection, all_questions, gen_responses)
        response = client.chat.completions.create(
                    model=args.curator_model,
                    messages=[{"role": "user", "content": cur_prompt}],
                    temperature=0.0
        )
        response = response.choices[0].message.content
        new_cheatsheet = extract_cheatsheet(response, old_cheatsheet)
        old_cheatsheet = new_cheatsheet
        if question_counts % 5 == 0:
            # save generated cheatsheet
            open(f"{args.save_path}/trial_question_{question_counts}.txt", "w+").write(new_cheatsheet)
        question_counts += 1
        counter += args.batch_size 

if __name__=="__main__":
    main()
