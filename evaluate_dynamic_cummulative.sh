SAMBANOVA_API_KEY=... python run_benchmark.py --task "CUSTOM_AIME_2024" --approach "DynamicCheatsheet_Cumulative" \	
    --model_name "sambanova/DeepSeek-V3-0324" \	
    --save_directory "temp_output" \	
    --generator_prompt_path "prompts/generator_prompt.txt" \	
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \	
    --max_num_rounds 5
