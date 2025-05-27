python run_benchmark.py --task "CUSTOM_AIME_2024" --approach "default" \
    --model_name "openai/gpt-4o" \
    --save_directory "gpt_4o_default" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --max_num_rounds 1 --no_shuffle --max_tokens 2500
