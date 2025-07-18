python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "xbrl_delta_batch_size_2"   \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_2
