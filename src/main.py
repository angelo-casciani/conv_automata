from argparse import ArgumentParser
from dotenv import load_dotenv
from torch import cuda
import warnings

from pipeline import *
from utility import *


DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
SEED = 10
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--llm_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='LLM model identifier')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=512)
    parser.add_argument('--modality', type=str, default='live', help='Modality to use between: evaluation, live')
    args = parser.parse_args()

    return args


def main():
    print("""Welcome! The tasks that are possible on the LEGO Factory are:
          - Discrete simulation of the production in a specified time interval in units of time (SimPy);
          - Discrete simulation of the production of a specified number of pieces (SimPy);
          - Next Event Prediction (SimPy);
          - Verification of temporal properties on the automaton representing the factory (Uppaal).\n""")

    args = parse_arguments()

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain_factory = initialize_chain(model_id, HF_AUTH, max_new_tokens)
    chain_uppaal = initialize_chain(model_id, HF_AUTH, max_new_tokens)
    chain_answer = initialize_chain(model_id, HF_AUTH, max_new_tokens)

    run_data = {
        'LLM ID': model_id,
        'Max Generated Tokens LLM': max_new_tokens,
        'Interaction Modality': args.modality
    }

    if 'evaluation' in args.modality:
        pass
    else:
        live_prompting(model_id, chain_factory, chain_uppaal, chain_answer, run_data)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
