SAMBANOVA_API_KEY=9bee3459-3e28-47b9-b0e6-2e54b923ab49 python run_benchmark_finlora.py --dataset "financebench"  --approach "default" \
    --model_name "sambanova/Meta-Llama-3.1-8B-Instruct" \
    --save_directory "finlora_derisking"  \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 512  
