SAMBANOVA_API_KEY=[...] python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking_xbrl_final"   \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/ml-sc-scratch2/shubhangiu/jays_dc_repo/dynamic-cheatsheet/generated_training_cheatsheets_xbrl_100_with_compression/trial_question_50.txt
