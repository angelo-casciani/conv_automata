import datetime
import os


class AnswerVerificationOracle:
    def __init__(self, info_run):
        self.question_with_expected_answer_pairs = {}
        self.accuracy = 0
        self.results = []
        self.run_info = info_run

    def add_question_expected_answer_pair(self, question, expected_answer):
        self.question_with_expected_answer_pairs[question] = expected_answer

    """ Verifying the answer correctness.

    This method checks whether the model's answer matches the expected answer for a given prompt.
    """

    def verify_answer(self, prompt, question, model_answer):
        result = {
            'prompt': prompt,
            'question': question,
            'model_answer': model_answer,
            'expected_answer': None,
            'verification_result': None
        }
        expected_answer = self.question_with_expected_answer_pairs.get(question)
        if expected_answer is not None:
            result['expected_answer'] = expected_answer
            expected_answer_formatted = expected_answer.lower().replace(' ', '')
            model_answer_formatted = model_answer.lower().replace('\n', ' ').replace(' ', '')
            # Check for unrelated questions in routing evaluations
            if expected_answer == "no_answer":
                model_answer_formatted = model_answer.lower()
                result['verification_result'] = False
                if ('uppaal_verification' not in model_answer_formatted and 'factory_simulation' not in model_answer_formatted) or ('uppaal_verification' in model_answer_formatted and 'factory_simulation' in model_answer_formatted):
                    result['verification_result'] = True
            else:
                result['verification_result'] = expected_answer_formatted in model_answer_formatted
            print(f"Answer: {model_answer}\nExpected_answer: {result['expected_answer']}\nResult: {result['verification_result']}")
        self.results.append(result)

        return result['verification_result']

    """ Computing the metrics for the run.
    
       This method computes and stores the metrics for the run.
    """

    def compute_stats(self):
        total_results = len(self.results)
        correct_results = sum(int(result['verification_result']) for result in self.results)

        self.accuracy = (correct_results / total_results) * 100 if total_results > 0 else 0

    """ Writing the verification results to a file.

    This method produces in output the results of the validation procedure. 
    """

    def write_results_to_file(self):
        file_path = os.path.join(os.path.dirname(__file__), "..", "tests", "validation",
                                 f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        self.compute_stats()

        with open(file_path, 'w') as file:
            file.write('INFORMATION ON THE RUN\n\n')
            for key in self.run_info.keys():
                file.write(f"{key}: {self.run_info[key]}\n")
            file.write('\n-----------------------------------\n')
            file.write(f"Accuracy: {self.accuracy:.2f}%\n")
            file.write("-----------------------------------\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(f"Verification Result: {result['verification_result']}\n")
                file.write("\n#####################################################################################\n")
