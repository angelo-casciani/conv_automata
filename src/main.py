from argparse import ArgumentParser
from dotenv import load_dotenv
from torch import cuda
import warnings

from pipeline import *
from utility import *


DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SEED = 10
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--llm_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='LLM model identifier')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=512)
    parser.add_argument('--modality', type=str, default='live', help='Modality to use between: evaluation-simulation, evaluation-verification, evaluation-routing, live')
    args = parser.parse_args()

    return args


def main():
    print("""Welcome! The tasks that are possible on the LEGO Factory are:
          - Discrete simulation of the production in a specified time interval in units of time (SimPy);
          - Discrete simulation of the production of a specified number of pieces (SimPy);
          - Prediction of the next station in the production line (SimPy);
          - Verification of temporal properties on the automaton representing the factory (Uppaal).\n""")

    args = parse_arguments()

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain_factory = initialize_chain(model_id, HF_AUTH, OPENAI_API_KEY, max_new_tokens)
    chain_uppaal = initialize_chain(model_id, HF_AUTH, OPENAI_API_KEY, max_new_tokens)
    chain_answer = initialize_chain(model_id, HF_AUTH, OPENAI_API_KEY, max_new_tokens)

    run_data = {
        'LLM ID': model_id,
        'Max Generated Tokens LLM': max_new_tokens,
        'Interaction Modality': args.modality
    }

    if 'evaluation-simulation' in args.modality:
        evaluate_performance(model_id, chain_factory, chain_answer, 'simulation.csv', run_data)
    elif 'evaluation-verification' in args.modality:
        evaluate_performance(model_id, chain_uppaal, chain_answer, 'verification.csv', run_data)
    elif 'evaluation-routing' in args.modality:
        evaluate_performance(model_id, chain_factory, chain_answer, 'routing.csv', run_data)
    elif 'evaluation-answer' in args.modality:
        evaluate_performance(model_id, chain_factory, chain_answer, 'answer.csv', run_data)
    else:
        live_prompting(model_id, chain_factory, chain_uppaal, chain_answer, run_data)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
