# echo "Runing exp 1 "
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_1/num_train_sample_100/top_k_aggregate_3" --batch_size 1 --num_train_sample 100 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_1; \
# echo "Running exp 2"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_1/num_train_sample_200/top_k_aggregate_3" --batch_size 1 --num_train_sample 200 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_1; \
# echo "Running exp 3"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_1/num_train_sample_300/top_k_aggregate_3" --batch_size 1 --num_train_sample 300 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_1; \
# echo "Running exp 4"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_1/num_train_sample_400/top_k_aggregate_3" --batch_size 1 --num_train_sample 400 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_1; \
# echo "Running exp 5"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_1/num_train_sample_500/top_k_aggregate_3" --batch_size 1 --num_train_sample 500 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_1; \

# echo "Running exp 6"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_2/num_train_sample_100/top_k_aggregate_3" --batch_size 2 --num_train_sample 100 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_2; \
# echo "Running exp 7"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_2/num_train_sample_200/top_k_aggregate_3" --batch_size 2 --num_train_sample 200 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_2; \
# echo "Running exp 8"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_2/num_train_sample_300/top_k_aggregate_3" --batch_size 2 --num_train_sample 300 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_2; \
# echo "Running exp 9"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_2/num_train_sample_400/top_k_aggregate_3" --batch_size 2 --num_train_sample 400 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_2; \
# echo "Running exp 10"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_2/num_train_sample_500/top_k_aggregate_3" --batch_size 2 --num_train_sample 500 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_2; \

# echo "Running exp 11"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_8/num_train_sample_100/top_k_aggregate_3" --batch_size 8 --num_train_sample 100 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_8; \
# echo "Running exp 12"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_8/num_train_sample_200/top_k_aggregate_3" --batch_size 8 --num_train_sample 200 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_8; \
# echo "Running exp 13"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_8/num_train_sample_300/top_k_aggregate_3" --batch_size 8 --num_train_sample 300 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_8; \
# echo "Running exp 14"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_8/num_train_sample_400/top_k_aggregate_3" --batch_size 8 --num_train_sample 400 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_8; \
# echo "Running exp 15"
# python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
#     --model_name "Meta-Llama-3.1-8B-Instruct" \
#     --save_directory "xbrl_delta_batch_size_8/num_train_sample_500/top_k_aggregate_3" --batch_size 8 --num_train_sample 500 --top_k_aggregate 3 \
#     --generator_prompt_path "prompts/generator_prompt.txt" \
#      --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_8; \

echo "Running exp 16"
python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "xbrl_delta_batch_size_4/num_train_sample_100/top_k_aggregate_3" --batch_size 4 --num_train_sample 100 --top_k_aggregate 3 \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_4; \
echo "Running exp 17"
python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "xbrl_delta_batch_size_4/num_train_sample_200/top_k_aggregate_3" --batch_size 4 --num_train_sample 200 --top_k_aggregate 3 \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_4; \
echo "Running exp 18"
python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "xbrl_delta_batch_size_4/num_train_sample_300/top_k_aggregate_3" --batch_size 4 --num_train_sample 300 --top_k_aggregate 3 \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_4; \
echo "Running exp 19"
python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "xbrl_delta_batch_size_4/num_train_sample_400/top_k_aggregate_3" --batch_size 4 --num_train_sample 400 --top_k_aggregate 3 \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_4; \
echo "Running exp 20"
python run_benchmark_finlora.py --dataset "xbrl_finer"  --approach "test_with_global_cheatsheet" \
    --model_name "Meta-Llama-3.1-8B-Instruct" \
    --save_directory "xbrl_delta_batch_size_4/num_train_sample_500/top_k_aggregate_3" --batch_size 4 --num_train_sample 500 --top_k_aggregate 3 \
    --generator_prompt_path "prompts/generator_prompt.txt" \
     --no_shuffle --max_tokens 32768 --sample_ratio 0.06  --max_num_rounds 1 --cheatsheet_prompt_path /import/snvm-sc-scratch2/jerrym/dynamic-cheatsheet/trials_delta/batch_size_4; \
