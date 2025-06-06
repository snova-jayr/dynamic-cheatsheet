SAMBANOVA_API_KEY=[...]  python run_benchmark_finlora.py --dataset "financebench"  --approach "Dynamic_with_global_cheatsheet" \
    --model_name "sambanova/Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking_dc"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 8192 --max_num_rounds 1 --cheatsheet_prompt_path /import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/generated_training_cheatsheets/financebench_cheatsheet_train_gpt_4o.txt
