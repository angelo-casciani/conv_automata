#!/bin/bash

############################## RQ1 ##############################

# Natural Text + No Chunking + Not Refined:
python3 main.py --modality evaluation1 --process_models food_nl_dfg_no_chunking
# BPMN + No chunking + Not Refined
python3 main.py --modality evaluation2 --process_models food_bpmn_no_chunking

############################## RQ2 ##############################

# Natural Text + No Chunking + Refined:
python3 main.py --modality evaluation1.1 --process_models food_nl_dfg_no_chunking
# Natural Text + Fixed Chunking + Not Refined:
python3 main.py --modality evaluation1 --process_models food_nl_dfg_fixed
# Natural Text + Recursive Chunking + Not Refined:
python3 main.py --modality evaluation1 --process_models food_nl_dfg_recursive
# Natural Text + Fixed Chunking + Refined:
python3 main.py --modality evaluation1.1 --process_models food_nl_dfg_fixed
# Natural Text + Recursive Chunking + Refined:
python3 main.py --modality evaluation1.1 --process_models food_nl_dfg_recursive
# BPMN + No chunking + Refined
python3 main.py --modality evaluation3 --process_models food_bpmn_no_chunking
# BPMN + Fixed Chunking + Not Refined:
python3 main.py --modality evaluation2 --process_models food_bpmn_fixed
# BPMN + Fixed Chunking + Refined:
python3 main.py --modality evaluation3 --process_models food_bpmn_fixed
# BPMN + Recursive Chunking + Not Refined:
python3 main.py --modality evaluation2 --process_models food_bpmn_recursive
# BPMN + Recursive Chunking + Refined:
python3 main.py --modality evaluation3 --process_models food_bpmn_recursive
# BPMN + BPMN-specific chunking + Not Refined
python3 main.py --modality evaluation2 --process_models food_bpmn
# BPMN + BPMN-specific chunking + Refined
python3 main.py --modality evaluation3 --process_models food_bpmn

# BPMN + BPMN-specific chunking + Refined + Llama 2 7B
python3 main.py --llm_id meta-llama/Llama-2-7b-chat-hf --modality evaluation3 --process_models food_bpmn
# BPMN + BPMN-specific chunking + Refined + Llama 3 8B
python3 main.py --llm_id meta-llama/Meta-Llama-3-8B-Instruct --modality evaluation3 --process_models food_bpmn
# BPMN + BPMN-specific chunking + Refined + Mistral Instruct 7B v0.2
python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.2 --modality evaluation3 --process_models food_bpmn
# BPMN + BPMN-specific chunking + Refined + Llama 3.1 8B
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_bpmn
# BPMN + BPMN-specific chunking + Refined + Mistral Instruct 7B v0.3
python3 main.py --llm_id mistralai/Mistral-7B-Instruct-v0.3 --modality evaluation3 --process_models food_bpmn
# BPMN + BPMN-specific chunking + Refined + Mixtral 8x7B Instruct v0.1
python3 main.py --llm_id mistralai/Mixtral-8x7B-Instruct-v0.1 --modality evaluation3 --process_models food_bpmn

############################## RQ3 ##############################

# sentence-transformers/all-MiniLM-L6-v2
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_nl_dfg_no_chunking --embed_model_id sentence-transformers/all-MiniLM-L6-v2
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_bpmn --embed_model_id sentence-transformers/all-MiniLM-L6-v2
# sentence-transformers/paraphrase-xlm-r-multilingual-v1
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_nl_dfg_no_chunking --embed_model_id sentence-transformers/paraphrase-xlm-r-multilingual-v1
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_bpmn --embed_model_id sentence-transformers/paraphrase-xlm-r-multilingual-v1
# jtlicardo/bert-finetuned-bpmn
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_nl_dfg_no_chunking --embed_model_id jtlicardo/bert-finetuned-bpmn
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation3 --process_models food_bpmn --embed_model_id jtlicardo/bert-finetuned-bpmn

############################## RQ4 ##############################

# Food Delivery + Fine-tuning int4
python3 main.py --llm_id angeloc1/llama3dot1FoodDel4v05 --modality evaluation6 --process_models food_bpmn
# Food Delivery + Fine-tuning int8
python3 main.py --llm_id angeloc1/llama3dot1FoodDel8v02 --modality evaluation6 --process_models food_bpmn
# Different Processes + No fine-tuning
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation4 --process_models food_bpmn reimbursement_bpmn
# Similar Processes + No fine-tuning
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation5 --process_models food_bpmn ecommerce_bpmn
# Different Processes + Fine-tuning int4
python3 main.py --llm_id angeloc1/llama3dot1DifferentProcesses4 --modality evaluation4 --process_models food_bpmn reimbursement_bpmn
# Similar Processes + Fine-tuning int4
python3 main.py --llm_id angeloc1/llama3dot1SimilarProcesses4 --modality evaluation5 --process_models food_bpmn ecommerce_bpmn
# Different Processes + Fine-tuning int8
python3 main.py --llm_id angeloc1/llama3dot1DifferentProcesses8 --modality evaluation4 --process_models food_bpmn reimbursement_bpmn
# Similar Processes + Fine-tuning int8
python3 main.py --llm_id angeloc1/llama3dot1SimilarProcesses8 --modality evaluation5 --process_models food_bpmn ecommerce_bpmn

############################## Qualitative Evaluation ##############################

python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --process_models food_bpmn_no_chunking --modality live --max_new_tokens 512
