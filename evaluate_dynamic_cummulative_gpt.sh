python run_benchmark.py --task "CUSTOM_AIME_2024" --approach "DynamicCheatsheet_Cumulative" \
    --model_name "openai/gpt-4o" \
    --save_directory "gpt_4o_dc" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --max_num_rounds 1 --no_shuffle 
