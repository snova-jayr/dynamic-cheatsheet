SAMBANOVA_API_KEY=...  python run_benchmark.py --task "CUSTOM_AIME_2024" --approach "DynamicCheatsheet_RetrievalSynthesis" \
    --model_name "sambanova/DeepSeek-R1" \
    --save_directory "temp_output" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_retrieval_synthesis.txt" \
    --max_num_rounds 1  --no_shuffle 
