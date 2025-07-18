from glob import glob
import random
import ast
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Open JSONL file
dataset_path = "/import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/data/finlora/train/finer_train_batched.jsonl"
num_samples = 100

samba_client = openai.OpenAI(
    base_url="https://api.sambanova.ai/v1", 
    api_key=""
)

with open(dataset_path, 'r') as json_file:
    all_samples = list(json_file)

def get_embedding(input, embeddings):
    for retry in range(10):
        try: 
            response = samba_client.embeddings.create(
                model="E5-Mistral-7B-Instruct",
                input=[input]
            )
            embeddings.append(response.data[0].embedding)
            break
        except:
            sleep_time = min(2 ** retry, 60)  # Cap at 60 seconds to avoid too long waits
            time.sleep(sleep_time)


task_pbar = tqdm(range(len(all_samples)))

question_embeddings = []
for i in task_pbar:
    sample = all_samples[i]
    task_dict = ast.literal_eval(sample)
    all_context  = task_dict["context"]
    index = all_context.index("Answer the following 4 independent questions by providing only")
    context, question = all_context[:index], all_context[index:]
    for retry in range(10):
        try: 
            response = samba_client.embeddings.create(
                model="E5-Mistral-7B-Instruct",
                input=[question]
            )
            question_embeddings.append(response.data[0].embedding)
            break
        except:
            sleep_time = min(2 ** retry, 60)  # Cap at 60 seconds to avoid too long waits
            time.sleep(sleep_time)

with open("data/finlora/train/finer_train_batched_embeddings.txt", 'w+') as f:
    for embed in question_embeddings:
        f.write(str(embed) + "\n")