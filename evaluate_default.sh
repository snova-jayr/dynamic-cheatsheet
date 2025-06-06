SAMBANOVA_API_KEY=[...] python run_benchmark_finlora.py --dataset "financebench"  --approach "default" \
    --model_name "sambanova/Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 512  
