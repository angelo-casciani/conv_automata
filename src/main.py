from argparse import ArgumentParser
from dotenv import load_dotenv
from torch import cuda
import warnings

# from oracle import AnswerVerificationOracle
from pipeline import *
from utility import *
# from vector_store import *


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
load_dotenv()
hf_auth = os.getenv('HF_TOKEN')
"""url = os.getenv('QDRANT_URL')
grpc_port = int(os.getenv('QDRANT_GRPC_PORT'))
collection_name = 'process-rag'"""
SEED = 10
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-13b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--modality', type=str, default='live')
    """parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model identifier')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context',
                        default=10)
    parser.add_argument('--process_models', nargs='+', help='The process model(s) to analyze', default=['food_bpmn'])
    parser.add_argument('--rebuild_vectordb', type=str2bool, help='Rebuild the vector index', default=True)
    """
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    """embed_model_id = args.embed_model_id
    embed_model = initialize_embedding_model(embed_model_id, device, args.batch_size)

    if args.rebuild_vectordb:
        delete_qdrant_collection(url, grpc_port, collection_name)
    qdrant = initialize_vector_store(url, grpc_port, collection_name, embed_model)
    num_docs = args.num_documents_in_context

    if 'food_bpmn' in args.process_models:
        filename_bpmn = 'food_delivery.bpmn'
        content = load_process_representation(filename_bpmn)
        semantic_chunks = parse_bpmn_in_chunks(content)
        qdrant = store_vectorized_semantically_chunked_bpmn(semantic_chunks, filename_bpmn, embed_model, url, grpc_port, collection_name)
    if 'food_bpmn_no_chunking' in args.process_models:
        filename_bpmn = 'food_delivery_cleaned.bpmn'
        content = load_process_representation(filename_bpmn)
        qdrant = store_vectorized_textual_dfg(content, filename_bpmn, embed_model, url, grpc_port, collection_name)
    if 'food_bpmn_fixed' in args.process_models:
        filename_bpmn = 'food_delivery.bpmn'
        content = load_process_representation(filename_bpmn)
        qdrant = store_fixed_chunked_text(content, filename_bpmn, embed_model, url, grpc_port, collection_name)
    if 'food_bpmn_recursive' in args.process_models:
        filename_bpmn = 'food_delivery.bpmn'
        content = load_process_representation(filename_bpmn)
        qdrant = store_recursive_chunked_text(content, filename_bpmn, embed_model, url, grpc_port, collection_name)
    if 'food_nl_dfg_fixed' in args.process_models:
        filename_ad = 'food_delivery_activities.txt'
        activities_definition = load_process_representation(filename_ad)
        filename_cf = 'food_delivery_flow.txt'
        control_data_flow = load_process_representation(filename_cf)
        store_fixed_chunked_text(activities_definition, filename_cf, embed_model, url, grpc_port, collection_name)
        qdrant = store_fixed_chunked_text(control_data_flow, filename_cf, embed_model, url, grpc_port, collection_name)
    if 'food_nl_dfg_recursive' in args.process_models:
        filename_ad = 'food_delivery_activities.txt'
        activities_definition = load_process_representation(filename_ad)
        filename_cf = 'food_delivery_flow.txt'
        control_data_flow = load_process_representation(filename_cf)
        store_recursive_chunked_text(activities_definition, filename_cf, embed_model, url, grpc_port, collection_name)
        qdrant = store_recursive_chunked_text(control_data_flow, filename_cf, embed_model, url, grpc_port, collection_name)
    if 'food_nl_dfg_no_chunking' in args.process_models:
        filename_ad = 'food_delivery_activities.txt'
        activities_definition = load_process_representation(filename_ad)
        filename_cf = 'food_delivery_flow.txt'
        control_data_flow = load_process_representation(filename_cf)
        store_vectorized_textual_dfg(activities_definition, filename_cf, embed_model, url, grpc_port, collection_name)
        qdrant = store_vectorized_textual_dfg(control_data_flow, filename_cf, embed_model, url, grpc_port, collection_name)
    if 'reimbursement_bpmn' in args.process_models:
        filename_bpmn = 'reimbursement.bpmn'
        content = load_process_representation(filename_bpmn)
        semantic_chunks = parse_bpmn_in_chunks(content)
        qdrant = store_vectorized_semantically_chunked_bpmn(semantic_chunks, filename_bpmn, embed_model, url, grpc_port, collection_name)
    if 'ecommerce_bpmn' in args.process_models:
        filename_bpmn = 'ecommerce.bpmn'
        content = load_process_representation(filename_bpmn)
        semantic_chunks = parse_bpmn_in_chunks(content)
        qdrant = store_vectorized_semantically_chunked_bpmn(semantic_chunks, filename_bpmn, embed_model, url, grpc_port, collection_name)"""

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = initialize_chain(model_id, hf_auth, max_new_tokens)

    """questions = {}
    if 'evaluation' in args.modality:
        if args.modality == 'evaluation1':
            questions = load_csv_questions('1_questions_answers_not_refined_for_DFG.csv')
        elif args.modality == 'evaluation1.1':
            questions = load_csv_questions('1.1_questions_answers_refined_for_DFG.csv')
        elif args.modality == 'evaluation2':
            questions = load_csv_questions('2_questions_answers_not_refined.csv')
        elif args.modality == 'evaluation3':
            questions = load_csv_questions('3_questions_answers_refined.csv')
        elif args.modality == 'evaluation4':
            questions = load_csv_questions('4_questions_answers_different_processes.csv')
        elif args.modality == 'evaluation5':
            questions = load_csv_questions('5_questions_answers_similar_processes.csv')
        elif args.modality == 'evaluation6':
            questions = load_csv_questions('6_questions_answers_refined_ft.csv')
        oracle = AnswerVerificationOracle()
        evaluate_rag_pipeline(oracle, chain, qdrant, questions, model_id, num_docs)
    else:
        live_prompting(chain, qdrant, model_id, num_docs)"""

    qdrant = ''
    num_docs = 0
    live_prompting(chain, qdrant, model_id, num_docs)

if __name__ == "__main__":
    seed_everything(SEED)
    main()
