#!/bin/bash

############################## Live ##############################
# python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation-simulation --max_new_tokens 512

################### Evaluation for Simulation ####################
# python3 main.py --llm_id meta-llama/Meta-Llama-3-8B-Instruct --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Llama-3.2-1B-Instruct --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Llama-3.2-3B-Instruct --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.2 --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.3 --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-Nemo-Instruct-2407 --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id mistralai/Ministral-8B-Instruct-2410 --modality evaluation-simulation --max_new_tokens 512
# python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-simulation --max_new_tokens 512
python3 main.py --llm_id google/gemma-2-9b-it --modality evaluation-simulation --max_new_tokens 512

################# Evaluation for Verification ###################
python3 main.py --llm_id meta-llama/Meta-Llama-3-8B-Instruct --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id meta-llama/Llama-3.2-1B-Instruct --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id meta-llama/Llama-3.2-3B-Instruct --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.2 --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.3 --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id mistralai/Mistral-Nemo-Instruct-2407 --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id mistralai/Ministral-8B-Instruct-2410 --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-verification --max_new_tokens 512
python3 main.py --llm_id google/gemma-2-9b-it --modality evaluation-verification --max_new_tokens 512

#################### Evaluation for Routing #####################
# python3 main.py --llm_id meta-llama/Meta-Llama-3-8B-Instruct --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Llama-3.2-1B-Instruct --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Llama-3.2-3B-Instruct --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.2 --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.3 --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-Nemo-Instruct-2407 --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id mistralai/Ministral-8B-Instruct-2410 --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-routing --max_new_tokens 512
# python3 main.py --llm_id google/gemma-2-9b-it --modality evaluation-routing --max_new_tokens 512

#################### Evaluation for Answer ######################
# python3 main.py --llm_id meta-llama/Meta-Llama-3-8B-Instruct --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Llama-3.2-1B-Instruct --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id meta-llama/Llama-3.2-3B-Instruct --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.2 --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.3 --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id mistralai/Mistral-Nemo-Instruct-2407 --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id mistralai/Ministral-8B-Instruct-2410 --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-answer --max_new_tokens 512
# python3 main.py --llm_id google/gemma-2-9b-it --modality evaluation-answer --max_new_tokens 512

