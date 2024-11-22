import datetime
import json
import os

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import bfloat16

import llm_factory_interface as factory_interface
from oracle import AnswerVerificationOracle
import uppaal_interface
from utility import log_to_file, retrieve_automata, retrieve_factory, load_csv_questions


llama3_models = ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']
mistral_models = ['mistralai/Mistral-7B-Instruct-v0.2',  'mistralai/Mistral-7B-Instruct-v0.3',
                  'mistralai/Mistral-Nemo-Instruct-2407', 'mistralai/Ministral-8B-Instruct-2410']
qwen_models = ['Qwen/Qwen2.5-7B-Instruct']
openai_models = ['gpt-4o-mini']


def initialize_pipeline(model_identifier, hf_token, max_new_tokens):
    """
    Initializes a pipeline for text generation using a pre-trained language model and its tokenizer.

    Args:
        model_identifier (str): The identifier of the pre-trained language model.
        hf_token (str): The token used for the language model.
        max_new_tokens (int): The maximum number of tokens to generate.

    Returns:
        generate_text: The pipeline for text generation.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = AutoConfig.from_pretrained(
        model_identifier,
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_token
    )
    model.eval()
    # print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        token=hf_token
    )
    if model_identifier in llama3_models:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )
    elif model_identifier in mistral_models:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("[/INST]]")
        ]
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )
    elif model_identifier in qwen_models:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )
    else:
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )

    return generate_text


def generate_prompt_template(model_id):
    path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(path_prompts, 'r') as prompt_file:
        prompts = json.load(prompt_file)

    if model_id in llama3_models:
        template = prompts.get('template-llama_instruct', '')
    elif model_id in mistral_models:
        template = prompts.get('template-mistral', '')
    elif model_id in qwen_models:
        template = prompts.get('template-qwen', '')
    else:
        template = prompts.get('template-generic', '')
    prompt = PromptTemplate.from_template(template)

    return prompt


def initialize_chain(model_id, hf_auth, openai_auth, max_new_tokens):
    if model_id not in openai_models:
        generate_text = initialize_pipeline(model_id, hf_auth, max_new_tokens)
        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        prompt = generate_prompt_template(model_id)
        chain = prompt | hf_pipeline
    else:
        chain = OpenAI(
            api_key=openai_auth,
        )

    return chain


def parse_llm_answer(compl_answer, llm_choice):
    if llm_choice in llama3_models:
        delimiter = '<|start_header_id|>assistant<|end_header_id|>'
    elif llm_choice in mistral_models:
        delimiter = '[/INST]'
    elif llm_choice in qwen_models:
        delimiter = '<|im_start|>assistant'
    else:
        delimiter = 'Answer:'

    index = compl_answer.find(delimiter)
    prompt = compl_answer[:index + len(delimiter)]
    answer = compl_answer[index + len(delimiter):]

    return prompt, answer


def produce_answer_interface_llm(question, model_id, llm_chain, answer_phase):
    path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(path_prompts, 'r') as prompt_file:
        prompts = json.load(prompt_file)
    prompt, answer = ('', '')
    if model_id not in openai_models:
        if answer_phase == 'routing':
            sys_mess = prompts.get('system_message_routing', '')
            context = prompts.get('context_routing', '')
        elif answer_phase == 'negative_response':
            sys_mess = prompts.get('system_message_negative', '')
            context = ''
        complete_answer = llm_chain.invoke({"question": question,
                                            "context": context,
                                            "system_message": sys_mess})
        prompt, answer = parse_llm_answer(complete_answer, model_id)
    else:
        if answer_phase == 'routing':
            sys_mess = prompts.get('system_message_routing', '')
            context = prompts.get('context_routing', '')
            prompt = f'{sys_mess}\nHere is the context: {context}\n' + f'Here is the user question: {question}\nAnswer: '
            completion = llm_chain.chat.completions.create(
                model = model_id,
                messages = [
                    {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                    {"role": "user", "content": f'Here is the user question: {question}\nAnswer: '},
                ]
            )
            answer = completion.choices[0].message.content.strip()
        elif answer_phase == 'negative_response':
            sys_mess = prompts.get('system_message_negative', '')
            context = ''
            prompt = f'{sys_mess}\nHere is the context: {context}\n' + f'Here is the user question: {question}\nAnswer: '
            completion = llm_chain.chat.completions.create(
                model = model_id,
                messages = [
                    {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                    {"role": "user", "content": f'Here is the user question: {question}\nAnswer: '},
                ]
            )
            answer = completion.choices[0].message.content.strip()

    return prompt, answer


def produce_answer_simulation(question, choice, llm_simpy, llm_answer):
    path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(path_prompts, 'r') as prompt_file:
        prompts = json.load(prompt_file)
    factory_data = retrieve_factory()
    station_names = ', '.join([station for station in factory_data['stations']])
    if choice not in openai_models:
        sys_mess = prompts.get('system_message_simulation', '') + prompts.get('shots_simulation', '')
        context = prompts.get('context_simulation', '').replace('LABELS', station_names)
        complete_answer = llm_simpy.invoke({"question": question,
                                            "context": context,
                                            "system_message": sys_mess})
        prompt, answer = parse_llm_answer(complete_answer, choice)
        print(complete_answer)

        if llm_answer is not None:
            results = factory_interface.interface_with_llm(answer)
            sys_mess = prompts.get('system_message_results_sim', '')
            context = f"The labels for the stations are: {station_names}\nResults from the simulation: {results}"
            complete_answer = llm_answer.invoke({"question": question,
                                                "context": context,
                                                "system_message": sys_mess})
            prompt, answer = parse_llm_answer(complete_answer, choice)
    else:
        sys_mess = prompts.get('system_message_simulation', '') + prompts.get('shots_simulation', '')
        context = prompts.get('context_simulation', '').replace('LABELS', station_names)
        prompt = f'{sys_mess}\nHere is the context: {context}\n' + f'Here is the user question: {question}\nAnswer: '
        completion = llm_simpy.chat.completions.create(
            model = choice,
            messages = [
                {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                {"role": "user", "content": f'Here is the user question: {question}\nAnswer: '},
            ]
        )
        answer = completion.choices[0].message.content.strip()
        # print(prompt + '\n' + answer)

        if llm_answer is not None:
            results = factory_interface.interface_with_llm(answer)
            sys_mess = prompts.get('system_message_results_sim', '')
            context = f"The labels for the stations are: {station_names}\nResults from the simulation: {results}"
            completion = llm_answer.chat.completions.create(
                model = choice,
                messages = [
                    {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                    {"role": "user", "content": f'Here is the user question: {question}\nAnswer: '},
                ]
            )
            answer = completion.choices[0].message.content.strip()

    return prompt, answer


def produce_answer_uppaal(question, choice, llm_uppaal, llm_answer):
    path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(path_prompts, 'r') as prompt_file:
        prompts = json.load(prompt_file)
    automata_data = retrieve_automata()
    if choice not in openai_models:
        sys_mess = prompts.get('system_message_verification', '') + prompts.get('shots_verification', '')
        context = prompts.get('context_verification', '').replace('STATES', str(list(automata_data['transitions'].keys())))
        complete_answer = llm_uppaal.invoke({"question": question,
                                            "context": context,
                                            "system_message": sys_mess})
        prompt, answer = parse_llm_answer(complete_answer, choice)
        print(complete_answer)

        if llm_answer is not None:
            results = uppaal_interface.interface_with_llm(answer)
            sys_mess = prompts.get('system_message_results', '')
            context = f'Results from Uppaal: {results}'
            complete_answer = llm_answer.invoke({"question": question,
                                                "context": context,
                                                "system_message": sys_mess})
            prompt, answer = parse_llm_answer(complete_answer, choice)
    else:
        sys_mess = prompts.get('system_message_verification', '') + prompts.get('shots_verification', '')
        context = prompts.get('context_verification', '').replace('STATES', str(list(automata_data['transitions'].keys())))
        prompt = f'{sys_mess}\nHere is the context: {context}\n' + f'Here is the user question: {question}\nAnswer: '
        completion = llm_uppaal.chat.completions.create(
            model = choice,
            messages = [
                {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                {"role": "user", "content": f'Here is the user question: {question}\nAnswer: '},
            ]
        )
        answer = completion.choices[0].message.content.strip()
        # print(prompt + '\n' + answer)

        if llm_answer is not None:
            results = uppaal_interface.interface_with_llm(answer)
            sys_mess = prompts.get('system_message_results', '')
            context = f'Results from Uppaal: {results}'
            prompt = f'{sys_mess}\nHere is the context: {context}\n' + f'Here is the user question: {question}\nAnswer: '
            completion = llm_answer.chat.completions.create(
                model = choice,
                messages = [
                    {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                    {"role": "user", "content": f'Here is the user question: {question}\nAnswer: '},
                ]
            )
            answer = completion.choices[0].message.content.strip()
    return prompt, answer


def generate_response(question, curr_datetime, model_id, model_factory, model_uppaal, model_answer, info_run):
    complete_prompt, answer = produce_answer_interface_llm(question, model_id, model_answer, 'routing')
    print(f'Prompt: {complete_prompt}\n')
    print(f'Answer: {answer}\n')
    print('--------------------------------------------------')

    if 'uppaal_verification' in answer.lower():
        complete_prompt, answer = produce_answer_uppaal(question, model_id, model_uppaal, model_answer)
    elif 'factory_simulation' in answer.lower():
        complete_prompt, answer = produce_answer_simulation(question, model_id, model_factory, model_answer)
    else:
        complete_prompt, answer = produce_answer_interface_llm(question, model_id, model_answer, 'negative_response')

    print(f'Prompt: {complete_prompt}\n')
    print(f'Answer: {answer}\n')
    print('--------------------------------------------------')

    log_to_file(f'Query: {complete_prompt}\n\nAnswer: {answer}\n\n##########################\n\n',
                curr_datetime, info_run)


def live_prompting(choice_llm, model_factory, model_uppaal, model_answer, info_run):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query you want to ask (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break
        
        generate_response(query, current_datetime, choice_llm, model_factory, model_uppaal, model_answer, info_run)
        print()


def evaluate_performance(choice_llm, lang_chain, llm_answer, test_filename, info_run):
    questions = load_csv_questions(test_filename)
    oracle = AnswerVerificationOracle(info_run)
    count = 0
    prompt, answer = '', ''
    for el in questions:
        question = el[0]
        expected_answer = el[1]
        # test_type = el[2]
        oracle.add_question_expected_answer_pair(question, expected_answer)
        if test_filename == 'simulation.csv':
            prompt, answer = produce_answer_simulation(question, choice_llm, lang_chain, None)
        elif test_filename == 'verification.csv':
            prompt, answer = produce_answer_uppaal(question, choice_llm, lang_chain, None)
        elif test_filename == 'routing.csv':
            prompt, answer = produce_answer_interface_llm(question, choice_llm, llm_answer, 'routing')
        """elif test_filename == 'answer.csv':
            if test_type == 'simulation':
                prompt, answer = produce_answer_simulation(question, choice_llm, lang_chain, llm_answer)
            elif test_type == 'verification':
                prompt, answer = produce_answer_uppaal(question, choice_llm, lang_chain, llm_answer)"""
        oracle.verify_answer(prompt, question, answer)
        count += 1
        print(f'Processing answer for question {count} of {len(questions)}...')

    print('Validation process completed. Check the output file.')
    oracle.write_results_to_file()