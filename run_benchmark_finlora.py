from datetime import datetime
import json
import os
import pandas as pd
import numpy as np
import argparse
import time 
import torch 
from tqdm import tqdm
import sklearn
import evaluate

from datasets import load_dataset, load_from_disk
from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.evaluation import eval_for_GameOf24, eval_for_multiple_choice, eval_for_exact_matching_with_no_punctuation, eval_equation_balancer

from dotenv import load_dotenv



DATA_DIR = "/import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/data/finlora/test/" 

# Map task names to their JSONL files in the data/test directory
dataset_path = {
    "xbrl_tags_extract":         os.path.join(DATA_DIR, "xbrl_extract_tags_test.jsonl"),
    "xbrl_value_extract":        os.path.join(DATA_DIR, "xbrl_extract_value_test.jsonl"),
    "xbrl_formula_extract":      os.path.join(DATA_DIR, "xbrl_extract_formula_test.jsonl"),
    "xbrl_formula_calc_extract": os.path.join(DATA_DIR, "xbrl_extract_formula_calculations_test.jsonl"),
    "xbrl_finer":                os.path.join(DATA_DIR, "finer_test_batched.jsonl"),
    "xbrl_fnxl":                 os.path.join(DATA_DIR, "fnxl_test_batched.jsonl"),
    "fpb":                       os.path.join(DATA_DIR, "fpb_test.jsonl"),
    "fiqa":                      os.path.join(DATA_DIR, "fiqa_test.jsonl"),
    "tfns":                      os.path.join(DATA_DIR, "tfns_test.jsonl"),
    "nwgi":                      os.path.join(DATA_DIR, "nwgi_test.jsonl"),
    "headline":                  os.path.join(DATA_DIR, "headline_test.jsonl"),
    "ner":                       os.path.join(DATA_DIR, "ner_test.jsonl"),
    "financebench":              os.path.join(DATA_DIR, "financebench_test.jsonl"),
    "xbrl_term":                 os.path.join(DATA_DIR, "xbrl_term_test.jsonl"),
    "formula":                   os.path.join(DATA_DIR, "formula_test.jsonl"),
    "cfa_level1":                os.path.join(DATA_DIR, "cfa_level1_test.jsonl"),
    "cfa_level2":                os.path.join(DATA_DIR, "cfa_level2_test.jsonl"),
    "cfa_level3":                os.path.join(DATA_DIR, "cfa_level3_test.jsonl"),
    "cpa_reg":                   os.path.join(DATA_DIR, "cpa_reg_test.jsonl"),
}


max_new_token_dict = {
    "xbrl_tags_extract": 20,
    "xbrl_value_extract": 20,
    "xbrl_formula_extract": 30,
    "xbrl_formula_calc_extract": 30,
    "xbrl_finer": 100,
    "xbrl_fnxl": 100,
    "fpb": 10,
    "fiqa": 10,
    "tfns": 10,
    "nwgi": 10,
    "headline": 10,
    "ner": 10,
    "financebench": 50,
    "xbrl_term": 50,
    "formula": 50
}

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Arguments to pass to the program.")

    # Dataset name
    parser.add_argument("--dataset", required=True, help="Comma-separated list of dataset keys")

    # Approach name
    parser.add_argument("--approach_name", type=str, default="DynamicCheatsheet_Cumulative", help="Approach name")

    # Model name
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o-mini", help="Model name")

    # Paths to the prompt files
    parser.add_argument("--generator_prompt_path", type=str, default="prompts/simple_generator.txt", help="Path to the generator prompt file")
    parser.add_argument("--cheatsheet_prompt_path", type=str, default=None, help="Path to the cheatsheet prompt file")

    # Additional model-related arguments
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--max_num_rounds", type=int, default=1, help="Maximum number of rounds")

    # Execution flags
    parser.add_argument("--execute_python_code", action="store_true", help="Allow Python code execution")
    parser.add_argument("--initialize_cheatsheet_path", type=str, default=None, help="Path to initialize the cheatsheet")
    parser.add_argument("--retrieve_top_k", type=int, default=3, help="Top-k retrieval for dynamic approaches")

    # Continue from the previous run
    parser.add_argument("--continue_from_last_run_path", type=str, default=None, help="Path to continue from the last run")

    # Additional save-path-related arguments
    parser.add_argument("--save_directory", type=str, default="results", help="Directory to save results")
    parser.add_argument("--additional_flag_for_save_path", type=str, default="", help="Additional flag for the save path")
    
    # Additional inference-time args 
    parser.add_argument("--max_n_samples", type=int, default=-1, help="Maximum number of samples to process")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of the dataset")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    args = parser.parse_args()

    # Convert to a dictionary for compatibility with the rest of the code
    return args

def read_file(file_path: str) -> str:
    """
    Read the file and return the content.
    """
    with open(file_path, "r") as file:
        return file.read()

    
def write_jsonl(file_path, data):
    """
    Save the outputs to a file.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")


def test_fin_tasks(args, data_name="xbrl_finer", prompt_fun=None):
    start_time = time.time()
    results = {}
    
    print(f"Testing model: {args.model_name} on {data_name} with temperature={args.temperature}")

    if data_name not in dataset_path.keys():
        return results

    instructions = pd.read_json(path_or_buf=dataset_path[data_name], lines=True)
    sample_size = len(instructions)

    if args.sample_ratio < 1.0:
        sample_size = int(len(instructions) * args.sample_ratio)
        instructions = instructions.sample(frac=args.sample_ratio, random_state=42)

    # Read the prompt files
    args.generator_prompt = read_file(args.generator_prompt_path)
    if args.cheatsheet_prompt_path:
        args.cheatsheet_prompt = read_file(args.cheatsheet_prompt_path)
    else:
        args.cheatsheet_prompt = "(empty)"


    # Initialize the language model
    model = LanguageModel(
        model_name=args.model_name,
    )

    # Initialize the cheatsheet
    cheatsheet = "(empty)"
    if args.initialize_cheatsheet_path is not None:
        with open(args.initialize_cheatsheet_path, "r") as file:
            cheatsheet = file.read()

    time_stamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
    args.save_path_name = f"{args.save_directory}/{data_name}/{args.model_name}_{args.approach_name}_{time_stamp}.jsonl"

    # Create the directory if it does not exist
    dir_path = os.path.dirname(args.save_path_name)
    os.makedirs(dir_path, exist_ok=True)

    save_param_path = args.save_path_name.replace(".jsonl", "_params.json")
    dir_path = os.path.dirname(save_param_path)
    os.makedirs(dir_path, exist_ok=True)

     # Save the arguments to a file
    with open(save_param_path, "w") as file:
        json.dump(vars(args), file, indent=4)


    # Initialize the questions and the embeddings
    questions = None
    embeddings = None
    '''
    if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"]:
        df = pd.read_csv(f"embeddings/{args.task}.csv")
        questions = df["input"].tolist()
        embeddings = df["embedding"]
        embeddings = embeddings.apply(eval)
        embeddings = np.array(embeddings.tolist()) # (N, 1536)

        # Re-order the embeddings based on the order of the dataset inputs
        dataset_inputs = [example["input"] for example in dataset]
        indices = [questions.index(input) for input in dataset_inputs]
        embeddings = embeddings[indices]
        questions = dataset_inputs
    else:
        questions = [example["input"] for example in dataset]
    '''

    task_start_time = time.time()

    context = instructions['context'].tolist()
    target_list = instructions["target"].tolist()
    target_list = [str(x) for x in target_list]

    total_steps = instructions.shape[0] 
    out_text_list = []
    
    outputs = []
    generator_outputs_so_far = []

    task_pbar = tqdm(range(total_steps))

    for i in task_pbar:
        tmp_context = context[i]

        if not tmp_context:
            break
        
        tmp_target = instructions['target'].tolist()[i]
        
        output_dict = model.advanced_generate(
            approach_name=args.approach_name,
            input_txt=tmp_context,
            cheatsheet=cheatsheet,
            generator_template=args.generator_prompt,
            cheatsheet_template=args.cheatsheet_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_num_rounds=args.max_num_rounds,
            allow_code_execution=args.execute_python_code,
            code_execution_flag="EXECUTE CODE!",
            #original_input_corpus=questions[:i+1],
            #original_input_embeddings=embeddings[:i+1] if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"] else None,
            generator_outputs_so_far=generator_outputs_so_far,
            retrieve_top_k=args.retrieve_top_k,
        )

        generator_outputs_so_far.append(output_dict["final_output"])

        outputs.append({
                "input": tmp_context,
                "target": tmp_target,
                **output_dict,
            })
        cheatsheet = output_dict["final_cheatsheet"]
        final_answer = output_dict["final_output"]

        print(f"@ CHEATSHEET:\n{cheatsheet}")
        print('- ' * 50)
        print(f"INPUT: {tmp_context}")
        print(f"TARGET: {tmp_target}")
        print(f"FINAL ANSWER: {final_answer}")
        print("**" * 50)
        
        with open(args.save_path_name, "a") as f:
            f.write(f"INDEX: {i}\n")
            f.write(f"@ CHEATSHEET:\n{cheatsheet}\n")
            f.write('- ' * 50)
            f.write("\n")
            f.write(f"INPUT: {tmp_context}\n")
            f.write(f"TARGET: {tmp_target}\n")
            f.write(f"FINAL ANSWER: {final_answer}\n")
            f.write("**" * 50)
            f.write("\n")

        out_text_list.append(final_answer)

    if "finer" in data_name or "fnxl" in data_name:
        out_text_list, target_list = process_batched(out_text_list, target_list)

    per_question_time = (time.time() - task_start_time) / sample_size

    final_results = {}

    if data_name == "financebench" or data_name == "xbrl_term":
        metric = evaluate.load("bertscore")
        results = metric.compute(predictions=out_text_list, references=target_list, model_type="ProsusAI/finbert")
        precision = sum(results["precision"]) / len(results["precision"])
        recall = sum(results["recall"]) / len(results["recall"])
        f1 = sum(results["f1"]) / len(results["f1"])
        final_results['precision'] = precision 
        final_results['recall'] = recall 
        final_results['f1'] = f1
        print(
            f"\n✓ {data_name}: precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, Time per question: {per_question_time:.2f}")

    else:
        all_target_type_for_classification = list(set(target_list))
        acc, response = evaluate_accuracy(out_text_list, target_list, all_target_type_for_classification)

        try:
            f1 = sklearn.metrics.f1_score(target_list, response, average='weighted')
        except:
            f1 = -1
            print(f"Error calculating F1 score for {data_name}")
        print(
            f"\n✓ {data_name}: Accuracy: {acc * 100:.3f}%, F1: {f1:.3f}, Time per question: {per_question_time:.2f} s")
        
        final_results['acc'] = acc
        final_results['f1'] = f1
        results = {"task": data_name, "acc": acc, "f1": f1, "time": per_question_time}

    with open(args.save_path_name, "a") as f:
        f.write(f"Task: {data_name}\n")
        for key in final_results:
            f.write(f"{key}: {final_results[key]}\n")
        f.write(f"Per question time: {per_question_time:.2f} minutes\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Sample Ratio: {args.sample_ratio}\n")
        f.write(f"Temperature: {args.temperature}\n")

    return results


def main(args):
    with torch.no_grad():
        for dataset in args.dataset.split(','):
            print("testing:", dataset)
            test_fin_tasks(args, data_name=dataset)
    print("Evaluation Ends.")



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
