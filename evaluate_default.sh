SAMBANOVA_API_KEY=[...]  python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "default" \
    --model_name "sambanova/Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking_xbrl_final"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768  --sample_ratio 0.06
