from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16

import datetime
from utility import log_to_file
from vector_store import retrieve_context

def initialize_embedding_model(embedding_model_id, dev, batch_size):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={'device': dev},
        encode_kwargs={'device': dev, 'batch_size': batch_size}
        # multi_process=True
    )

    return embedding_model


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
    Here is the context: \n\n {context} \n\n
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


def produce_answer(question, llm_chain, vectdb, choice, num_chunks, live=False):
    sys_mess = "Use the following pieces of context to answer the question at the end."
    if not live:
        sys_mess = sys_mess + " Answer 'yes' if true or 'no' if false."
        # "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    context = retrieve_context(vectdb, question, num_chunks)
    complete_answer = llm_chain.invoke({"question": question,
                                        "system_message": sys_mess,
                                        "context": context})
    if 'meta-llama/Meta-Llama-3' in choice or 'llama3dot1' in choice:
        index = complete_answer.find('<|start_header_id|>assistant<|end_header_id|>')
        prompt = complete_answer[:index + len('<|start_header_id|>assistant<|end_header_id|>')]
        answer = complete_answer[index + len('<|start_header_id|>assistant<|end_header_id|>'):]
    elif 'Question:' in complete_answer:
        index = complete_answer.find('Answer: ')
        prompt = complete_answer[:index]
        answer = complete_answer[index + len('Answer: '):]
    else:
        index = complete_answer.find('[/INST]')
        prompt = complete_answer[:index + len('[/INST]')]
        answer = complete_answer[index + len('[/INST]'):]
    return prompt, answer


def produce_answer_live(question, curr_datetime, model_chain, vectordb, choice, num_chunks):
    complete_prompt, answer = produce_answer(question, model_chain, vectordb, choice, num_chunks, True)
    print(f'Prompt: {complete_prompt}\n')
    print(f'Answer: {answer}\n')
    print('--------------------------------------------------')

    log_to_file(f'Query: {complete_prompt}\n\nAnswer: {answer}\n\n##########################\n\n',
                curr_datetime)


def live_prompting(model1, vect_db, choice, num_chunks):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break

        produce_answer_live(query, current_datetime, model1, vect_db, choice, num_chunks)
        print()


def evaluate_rag_pipeline(eval_oracle, lang_chain, vect_db, dict_questions, choice, num_chunks):
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
    eval_oracle.write_results_to_file()