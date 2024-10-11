from argparse import ArgumentParser
from dotenv import load_dotenv
from torch import cuda
import warnings

# from oracle import AnswerVerificationOracle
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
    parser.add_argument('--modality', type=str, default='live')
    args = parser.parse_args()

    return args


def main():
    print("""Welcome! The tasks that are possible on the LEGO Factory automata are:
          - Simulation of the whole production process;
          - Simulation with specified sequence of events;
          - Event Prediction;
          - Simulation with cost analysis;
          - Verification of temporal properties.\n""")

    args = parse_arguments()

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = initialize_chain(model_id, HF_AUTH, max_new_tokens)
    chain2 = initialize_chain(model_id, HF_AUTH, max_new_tokens)

    live_prompting(chain, model_id, chain2, verification=True)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
