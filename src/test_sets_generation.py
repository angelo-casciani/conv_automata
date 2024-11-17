import csv
import os
import random


simulation_tasks = {
    "sim_with_time": [
        ("Carry out a simulation for {time} units of time to check how many pieces are produced.", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
        ("How many pieces can be produced in {time} units of time?", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
        ("What is the mean processing time after executing the factory for {time} time units?", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
        ("What is the mean waiting time after the factory's simulation for {time} time units?", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
        ("Tell me the mean transfer time after executing the factory for {time} time units?", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
        ("What is the mean processing time of {station} after the factory's execution for {time} time units?", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
        ("What is the mean waiting time of {station} after executing the factory for {time} time units?", 
         {"task": "sim_with_time", "simulation_time": "{time}", "target_pieces": "", "stations_sequence": []}),
    ],
    "sim_with_number_products": [
        ("Simulate the execution of the production process to produce {pieces} pieces.", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("How much time is needed to produce {pieces} pieces in the factory?", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("Run a simulation to estimate the time required to produce {pieces} products.", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("What is the mean processing time after executing the factory to generate {pieces} products?", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("What is the mean waiting time for producing {pieces} pieces?", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("Tell me the mean transfer time after executing the factory to produce {pieces} units?", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("What is the mean processing time of {station} to produce {pieces} pieces?", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
        ("What is the mean waiting time of {station} for creating {pieces} products?", 
         {"task": "sim_with_number_products", "simulation_time": "", "target_pieces": "{pieces}", "stations_sequence": []}),
    ],
    "event_prediction": [
        ("What is the next production station after {sequence}?", 
         {"task": "event_prediction", "simulation_time": "", "target_pieces": "", "stations_sequence": "{sequence}".split(", ")}),
        ("Predict the next station given this sequence: {sequence}.", 
         {"task": "event_prediction", "simulation_time": "", "target_pieces": "", "stations_sequence": "{sequence}".split(", ")}),
        ("Based on the sequence {sequence}, what is the most likely next station?", 
         {"task": "event_prediction", "simulation_time": "", "target_pieces": "", "stations_sequence": "{sequence}".split(", ")}),
    ]
}


time_range = range(200, 5001, 500)
pieces_range = range(50, 501, 50)
stations = ["Station1", "Station2", "Station3", "Station4", "Station5"]
tasks_proportions = [0.4, 0.4, 0.2]


def generate_stats_file(filename, samples):
    stats = {task: 0 for task in simulation_tasks.keys()}
    for question, answer in samples:
        for task in stats.keys():
            if task in answer:
                stats[task] += 1
                break
    stats_output_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets', f'{filename}_stats.txt')
    with open(stats_output_path, mode="w", encoding="utf-8") as file:
        for task, count in stats.items():
            file.write(f"{task}: {count}\n")

    print(f"Generated statistics and saved to {stats_output_path}")


def generate_simulation_samples(filename, samples):
    output_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets', f'{filename}.csv')
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "answer", "evaluation_type"])
        for question, answer in samples:
            writer.writerow([question, answer.replace("'", '"'), "simulation"])
    print(f"Generated {len(samples)} samples and saved to {output_path}")


def main():
    samples = []
    for _ in range(300):
        task = random.choices(
            population=list(simulation_tasks.keys()),
            weights=tasks_proportions,
            k=1)[0]
        template, answer_template = random.choice(simulation_tasks[task])
        
        if task == "sim_with_time":
            time_value = random.choice(time_range)
            if "{station}" in template:
                station_value = random.choice(stations)
                question = template.format(station=station_value, time=time_value)
            else:
                question = template.format(time=time_value)
            answer = answer_template.copy()
            answer["simulation_time"] = time_value
        elif task == "sim_with_number_products":
            pieces_value = random.choice(pieces_range)
            if "{station}" in template:
                station_value = random.choice(stations)
                question = template.format(station=station_value, pieces=pieces_value)
            else:
                question = template.format(pieces=pieces_value)
            answer = answer_template.copy()
            answer["target_pieces"] = pieces_value
        elif task == "event_prediction":
            sequence_length = random.randint(1, 4)
            sequence = ", ".join(stations[:sequence_length])
            question = template.format(sequence=sequence)
            answer = answer_template.copy()
            answer["stations_sequence"] = sequence.split(", ")        
        samples.append((question, str(answer)))

    generate_simulation_samples('simulation', samples)
    generate_stats_file('simulation', samples)


if __name__ == "__main__":
    main()
    """with open(os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets', 'simulation.csv'), mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= 3:
                break
            question, answer = row
            print(f"Question: {question}\nAnswer: {answer}\n")"""