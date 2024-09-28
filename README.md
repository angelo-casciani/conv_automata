# conv_automata

Proof of concept for enabling conversational reasoning over the automaton representing the LEGO factory.

## Installation

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.

First, you need to clone the repository:
```bash
git clone https://github.com/angelo-casciani/conv_automata
cd conv_automata
```

Run the following command to install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

## LLMs Requirements

Please note that this software leverages the open-source LLMs reported in the table:

| Model | HuggingFace Link |
|-----------|-----------|
| Llama 3.1 8B | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |

Request in advance the permission to use each Llama model for your HuggingFace account.

Please note that each of the selected models have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment meeting at least the minimum requirements for these models to run the software effectively.

## Running the Project
Before running the project, it is necessary to insert in the *.env* file your personal HuggingFace token (request the permission to use the Llama models for this token in advance).

Eventually, you can proceed by going in the project directory and executing the following command:
```bash
python3 main.py
```