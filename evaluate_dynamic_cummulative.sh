SAMBANOVA_API_KEY=[...] python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "DynamicCheatsheet_Cumulative" \
    --model_name "sambanova/Llama-4-Maverick-17B-128E-Instruct" \
    --save_directory "finlora_derisking_xbrl_final"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --cheatsheet_prompt_path "prompts/curator_prompt_for_financebench.txt" \
     --no_shuffle --max_tokens 32768 --max_num_rounds 1 --sample_ratio 0.06
