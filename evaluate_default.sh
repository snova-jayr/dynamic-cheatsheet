SAMBANOVA_API_KEY=9bee3459-3e28-47b9-b0e6-2e54b923ab49 python run_benchmark_finlora.py --dataset "financebench,xbrl_term,formula,xbrl_tags_extract,xbrl_value_extract,xbrl_formula_extract,xbrl_formula_calc_extract,xbrl_finer,xbrl_fnxl"  --approach "default" \
    --model_name "sambanova/Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 512 --sample_ratio 0.1 
