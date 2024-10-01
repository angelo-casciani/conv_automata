from argparse import ArgumentParser
from dotenv import load_dotenv
from torch import cuda
import warnings

# from oracle import AnswerVerificationOracle
from pipeline import *
from utility import *


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
    parser.add_argument('--llm_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='LLM model identifier')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=512)
    parser.add_argument('--modality', type=str, default='live')
    args = parser.parse_args()

    return args


def main():
    print("""Welcome! The tasks that are possible at the moment on the LEGO Factory automata are:
          1. Failure Mode Analysis
          2. Event Prediction
          3. Process Cost Computation
          4. Verification\n""")

    args = parse_arguments()

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = initialize_chain(model_id, hf_auth, max_new_tokens)
    chain2 = initialize_chain(model_id, hf_auth, max_new_tokens)

    qdrant = ''
    num_docs = 0
    # live_prompting(chain, qdrant, model_id, num_docs)
    live_prompting(chain, qdrant, model_id, num_docs, chain2)

if __name__ == "__main__":
    seed_everything(SEED)
    main()
