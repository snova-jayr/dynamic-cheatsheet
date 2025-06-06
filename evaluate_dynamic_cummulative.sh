SAMBANOVA_API_KEY=[...] python run_benchmark_finlora.py --dataset "financebench"  --approach "DynamicCheatsheet_Cumulative" \
    --model_name "sambanova/Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking_dc"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --cheatsheet_prompt_path "prompts/curator_prompt_for_financebench.txt" \
     --no_shuffle --max_tokens 8192 --max_num_rounds 1 
