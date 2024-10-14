from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import bfloat16

import datetime
import llm_factory_interface as factory_interface
import uppaal_interface
from utility import log_to_file, retrieve_automata


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
    if 'meta-llama/Meta-Llama-3' in model_identifier or 'llama3dot1' in model_identifier:
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
    # template = """<s>[INST]
    # <<SYS>>
    # {system_message}
    # <</SYS>>
    # <<CONTEXT>>
    # {context}
    # <</CONTEXT>>
    # <<QUESTION>>
    # {question}
    # <</QUESTION>>
    # <<ANSWER>> [/INST]"""

    template = """<s>[INST]
        <<SYS>>
        {system_message}
        <</SYS>>
        <<CONTEXT>>
        {context}
        <</CONTEXT>>
        <<QUESTION>>
        {question}
        <</QUESTION>>
        <<ANSWER>> [/INST]"""

    # template = """{system_message}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer: """

    # template_llama3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # {system_message}<|eot_id|>
    # <|start_header_id|>user<|end_header_id|>
    # <|start_context_id|>
    # {context}
    # <|end_context_id|>
    # <|start_question_id|>
    # {question}
    # <|end_question_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    template_llama3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> {system_message}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the context: {context}
    Here is the question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    if 'meta-llama/Meta-Llama-3' in model_id or 'llama3dot1' in model_id:
        prompt = PromptTemplate.from_template(template_llama3)
    else:
        prompt = PromptTemplate.from_template(template)

    return prompt


def initialize_chain(model_id, hf_auth, max_new_tokens):
    generate_text = initialize_pipeline(model_id, hf_auth, max_new_tokens)
    hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    prompt = generate_prompt_template(model_id)
    chain = prompt | hf_pipeline

    return chain


def parse_llm_answer(compl_answer, llm_choice):
    if 'meta-llama/Meta-Llama-3' in llm_choice or 'llama3dot1' in llm_choice:
        delimiter = '<|start_header_id|>assistant<|end_header_id|>'
    elif 'Question:' in compl_answer:
        delimiter = 'Answer: '
    else:
        delimiter = '[/INST]'

    index = compl_answer.find(delimiter)
    prompt = compl_answer[:index + len(delimiter)]
    answer = compl_answer[index + len(delimiter):]

    return prompt, answer


def produce_answer_interface_llm(question, model_id, llm_chain, answer_phase):
    prompt, answer = ('', '')
    if answer_phase == 'routing':
        automata_data = retrieve_automata()
        sys_mess = """You are a conversational gateway that redirects user queries related to the factory's automata 
        to the appropriate task handler based on the context.
        You don't have to answer to the user question!
        Just use it to route the request to the appropriate task handler"""
        context = f"""Rules for routing:
        - If the query involves verifying temporal properties (e.g., deadlocks, reachability) in the factory automata,
          generate a response containing the string "uppaal_verification."
        - If the query involves simulating the factory automata, predicting the next event, or estimating costs, 
          generate a response containing the string "factory_simulation."
        - For any other queries that don't match these categories, inform the user that their request does not align 
          with the supported tasks.
        The labels for the events are: {automata_data['event_symbols']}.
        The automaton states are: {list(automata_data['transitions'].keys())}."""
        complete_answer = llm_chain.invoke({"question": question,
                                            "context": context,
                                            "system_message": sys_mess})
        prompt, answer = parse_llm_answer(complete_answer, model_id)
    elif answer_phase == 'negative_response':
        sys_mess = """You are a conversational interface towards an automaton of the LEGO factory.
        Answer the user question saying it does not match any allowed tasks."""
        context = ''
        complete_answer = llm_chain.invoke({"question": question,
                                            "context": context,
                                            "system_message": sys_mess})
        prompt, answer = parse_llm_answer(complete_answer, model_id)

    return prompt, answer


def produce_answer_simulation(question, choice, llm_simpy, llm_answer):
    automata_data = retrieve_automata()
    sys_mess = """You are a conversational interface towards an automaton of the LEGO factory.
    Use the following pieces of context to generate from the user question a JSON object containing one of the allowed task and the provided event sequence (if any).
    If the user question does not match an allowed tasks kindly refuse to answer.
    
    Examples:
    1. Input (Natural Language Query): Carry out a simulation to check if the sequence _load_1, _process_1, _fail_1 leads to a failure.
       Output (LLM answer with Uppaal Query): {"task": "simulation", "events_sequence": ["s11", "s12", "s14"], "query_nl": "Carry out a simulation to check if the sequence _load_1, _process_1, _fail_1 leads to a failure."}
    2. Input (Natural Language Query): Simulate the execution of the production process.
       Output (LLM answer with Uppaal Query): {"task": "simulation", "events_sequence": [], "query_nl": "Simulate the execution of the production process."}
    3. Input (Natural Language Query): What is the next event possible after executing  _load_1, _process_1? 
       Output (LLM answer with Uppaal Query): {"task": "event_prediction", "events_sequence": ["s11", "s12"], "query_nl": "What is the next event possible after executing  _load_1, _process_1?"}
    4. Input (Natural Language Query): What is the cost of executing load_1, process_1, unload_1, load_2?
       Output (LLM answer with Uppaal Query): {"task": "simulation_cost", "events_sequence": ["s11", "s12", "s13", "s21"], "query_nl": "What is the cost of executing load_1, process_1, unload_1, load_2?"}"""
    context = f"The allowed tasks are: simulation, event_prediction, simulation_with_cost.\n\nThe labels for the events are: {automata_data['event_symbols']}"
    complete_answer = llm_simpy.invoke({"question": question,
                                        "context": context,
                                        "system_message": sys_mess})
    prompt, answer = parse_llm_answer(complete_answer, choice)

    print(complete_answer)
    results = factory_interface.interface_with_llm(answer)
    sys_mess = """You are a conversational interface towards an automata of the LEGO factory.
    Report the results given by the factory automata provided in the context to the user.
    If you are not able to derive the answer from the context, just say that you don't know, don't try to make up an answer."""
    context = f"The labels for the events are: {automata_data['event_symbols']}\n"
    context += f'Results from the automaton: {results}'
    complete_answer = llm_answer.invoke({"question": question,
                                         "context": context,
                                         "system_message": sys_mess})
    prompt, answer = parse_llm_answer(complete_answer, choice)

    return prompt, answer


def produce_answer_uppaal(question, choice, llm_uppaal, llm_answer):
    automata_data = retrieve_automata()
    sys_mess = """You are an assistant that translates natural language queries into a JSON object containing 
    the Uppaal syntax to be verified against a timed automaton. The automaton has several locations (states) such as q_1, q_6, and q_11, and uses variables like x (a clock), Tcdf (for time bounds), and loc_entity, edge_entity (for entities in the system). Remember that x is a local clock in the template instance s, so it must be referenced as s.x.

    The properties in Uppaal are typically expressed using Computation Tree Logic (CTL) and involve temporal operators such as:
    A<>: "Always eventually" (checks if something will always eventually happen).
    E<>: "Exists eventually" (checks if there is at least one path where something eventually happens).
    A[]: "Always" (checks if something always holds).
    E[]: "Exists" (checks if there's a path where something holds forever).
    simulate[...]: Simulates the system behavior over a time period.
    p --> q (in Uppaal, written as p --> q): Whenever p holds, q will eventually hold (not allowed within the scope of a quantifier).
    Given a query in natural language, you will translate it into an Uppaal query compatible with the model.

    Examples:
    1. Input (Natural Language Query): "Check if the system will always eventually reach state q_1."
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Check if the system will always eventually reach state q_1.", "uppaal_query": "A<> s.q_1"}
    2. Input (Natural Language Query): "Is there a path where state q_1 is reached?"
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Is there a path where state q_1 is reached?", "uppaal_query": "E<> s.q_1"}
    3. Input (Natural Language Query): "Simulate the system for up to 30 time units and track the states of all locations and the entity's location and edge."
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Simulate the system for up to 30 time units and track the states of all locations and the entity's location and edge.", "uppaal_query": "simulate[<=30]{s.q_1, s.q_6, s.q_11, s.q_10, s.q_0, s.q_13, s.__init__, s.q_3, s.q_4, s.q_12, s.q_14, s.q_9, s.q_5, s.q_7, s.q_8, s.q_2, s.loc_entity, s.edge_entity}"}
    4. Input (Natural Language Query): "Check if the system can always stay in state q_0 for up to 30 time units."
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Check if the system can always stay in state q_0 for up to 30 time units.", "uppaal_query": "A[] s.q_0 && s.x <= 30"}
    5. Input (Natural Language Query): "Is there any path where the system stays in state q_3 forever?"
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Is there any path where the system stays in state q_3 forever?", "uppaal_query": "E[] s.q_3"}
    6. Input (Natural Language Query): "Simulate the system over time and check the location and edge entity variables."
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Simulate the system over time and check the location and edge entity variables.", "uppaal_query": "E[] s.q_3"}
    7. Input (Natural Language Query): "Check if state q_6 is reachable within 15 time units."
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Check if state q_6 is reachable within 15 time units.", "uppaal_query": "E<> s.q_6 && s.x <= 15"}
    8. Input (Natural Language Query): "Whenever the system reaches q_0, it will eventually reach q_6."
       Output (LLM answer with Uppaal Query): {"task": "verification", "query_nl": "Whenever the system reaches q_0, it will eventually reach q_6.", "uppaal_query": "s.q_0 --> s.q_6"}
    
    Use these examples to translate additional queries and construct the Uppaal syntax based on the model provided in the context. Always ensure that the queries fit the automaton's structure, and when referring to local variables like clocks, prefix them with the template instance name (e.g., s.x for the clock x in the DiscoveredSystem template)."""
    context = f"The automaton states are: {list(automata_data['transitions'].keys())}"
    complete_answer = llm_uppaal.invoke({"question": question,
                                         "context": context,
                                         "system_message": sys_mess})
    prompt, answer = parse_llm_answer(complete_answer, choice)

    print(complete_answer)
    results = uppaal_interface.interface_with_llm(answer)
    sys_mess = """You are a conversational interface towards the Uppaal verifier.
    Report the results given by Uppaal provided in the context to the user.
    If you are not able to derive the answer from the context, just say that you don't know, don't try to make up an answer."""
    context = f'Results from Uppaal: {results}'
    complete_answer = llm_answer.invoke({"question": question,
                                         "context": context,
                                         "system_message": sys_mess})
    prompt, answer = parse_llm_answer(complete_answer, choice)

    return prompt, answer


def generate_response(question, curr_datetime, model_id, model_factory, model_uppaal, model_answer):
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
                curr_datetime)


def live_prompting(choice_llm, model_factory, model_uppaal, model_answer):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query you want to ask (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break

        generate_response(query, current_datetime, choice_llm, model_factory, model_uppaal, model_answer)
        print()


"""def evaluate_rag_pipeline(eval_oracle, lang_chain, vect_db, dict_questions, choice, num_chunks):
    count = 0
    for question, answer in dict_questions.items():
        eval_oracle.add_prompt_expected_answer_pair(question, answer)
        prompt, answer = produce_answer(question, lang_chain, vect_db, choice, num_chunks)
        if 'meta-llama/Meta-Llama-3' in choice or 'llama3dot1' in choice:
            eval_oracle.verify_answer(answer, prompt, True)
        else:
            eval_oracle.verify_answer(answer, prompt)
        count += 1
        print(f'Processing answer for activity {count} of {len(dict_questions)}...')

    print('Validation process completed. Check the output file.')
    eval_oracle.write_results_to_file()"""
