# conv_automata

Proof of concept for enabling conversational reasoning over the automaton representing the LEGO factory.

## Installation

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.

First, you need to clone the repository:
```bash
git clone https://github.com/angelo-casciani/conv_automata
cd conv_automata
```

Create a conda environment:
```bash
conda create -n conv_automata python=3.9 --yes
conda activate conv_automata
```

Run the following command to install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

## LLMs Requirements

Please note that this software leverages the open-source and closed-source LLMs reported in the table:

| Model | HuggingFace Link |
|-----------|-----------|
| meta-llama/Meta-Llama-3-8B-Instruct | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| meta-llama/Llama-3.1-8B-Instruct | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| meta-llama/Llama-3.2-3B-Instruct | [HF link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| mistralai/Mistral-7B-Instruct-v0.2 | [HF link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| mistralai/Mistral-7B-Instruct-v0.3 | [HF link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| mistralai/Mistral-Nemo-Instruct-2407 | [HF link](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| mistralai/Ministral-8B-Instruct-2410 | [HF link](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) |
| Qwen/Qwen2.5-7B-Instruct | [HF link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| google/gemma-2-9b-it | [HF link](https://huggingface.co/google/gemma-2-9b-it) |
| gpt-4o-mini | [OpenAI link](https://platform.openai.com/docs/models) |

Request in advance the permission to use each Llama model for your HuggingFace account.
Retrive your OpenAI API key to use the supported GPT model.

Please note that each of the selected models have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment meeting at least the minimum requirements for these models to run the software effectively.

## Running the Project
Before running the project, it is necessary to insert in the *.env* file your personal HuggingFace token (request the permission to use the Llama models for this token in advance) and OpenAI API key.

Eventually, you can proceed by going in the project directory and executing commands as the following one:
```bash
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality live --max_new_tokens 512
```

To run an evaluation for the simulation (*evaluation-simulation*), for the verification, or for the routing):
```bash
python3 main.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --modality evaluation-simulation --max_new_tokens 512
```

To generate new test sets for the three supported evaluation, run the script *test_sets_generation.py* before running an evaluation.
```bash
python3 test_sets_generation.py
```

A comprehensive list of commands can be found at *src/cmd4tests.sh*.