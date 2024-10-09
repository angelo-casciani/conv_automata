from lego_factory import FactoryAutomata, WeightedFactory
from utility import extract_json
import io
import simpy
import sys


def trigger_simulation(events_sequence=None):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    env = simpy.Environment()
    factory = FactoryAutomata(env)
    results = ''
    if events_sequence:
        env.process(factory.execute_event_sequence(events_sequence))
    else:
        env.process(factory.run_factory())
    env.run(until=100)
    sys.stdout = sys.__stdout__
    results = captured_output.getvalue()
    return results


def trigger_event_prediction(events_sequence):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    env = simpy.Environment()
    factory = FactoryAutomata(env)
    results = ''
    process = env.process(factory.execute_event_sequence(events_sequence))
    env.run(until=process)
    factory.predict_next_event()
    sys.stdout = sys.__stdout__
    results = captured_output.getvalue()
    return results


def trigger_simulation_with_cost(events_sequence=None):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    env = simpy.Environment()
    factory = WeightedFactory(env)
    results = ''
    if events_sequence:
        env.process(factory.execute_event_sequence(events_sequence))
    else:
        env.process(factory.run_factory())
    env.run(until=100)
    sys.stdout = sys.__stdout__
    results = captured_output.getvalue()
    return results

"""
TO SUPPORT AGAIN WITH THE UPPAAL INTEGRATION

def trigger_verification(events_sequence):
    factory = FactoryAutomata()
    results = ''
    for event in events_sequence:
        res = factory.trigger(event)
        results += f"{res}\n"
    results += factory.verify_process_completion()
    return results
"""


def interface_with_llm(llm_answer):
    json_request = extract_json(llm_answer)
    task = json_request.get("task")
    events_sequence = json_request.get("events_sequence")
    factory_output = ''

    if task == "simulation":
        return trigger_simulation(events_sequence)
    elif task == "event_prediction":
        return trigger_event_prediction(events_sequence)
    elif task == "simulation_cost":
        return trigger_simulation_with_cost(events_sequence)
    # elif task == "verification":
    #    return trigger_verification(events_sequence)
    
    json_request["results"] = factory_output
    print(json_request)
    return json_request


    """def main():
    simulation_output = trigger_simulation(['s11', 's12', 's13', 's21', 's22', 's23'])
    print(simulation_output)
    predicted_event = trigger_event_prediction(['s11', 's12', 's13', 's21', 's22', 's23'])
    print(predicted_event)
    simulation_output = trigger_simulation_with_cost()
    print(simulation_output)
    
    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "simulation", "events_sequence": ["s11", "s12", "s14"]}'
    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "simulation", "events_sequence": []}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)
    
    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "event_prediction", "events_sequence": ["s11"]}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)

    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "simulation_cost", "events_sequence": ["s11", "s12", "s14"], "results": null}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)
    
    
    TO TEST AGAIN WITH THE UPPAAL INTEGRATION
    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "verification", "events_sequence": ["s11", "s12", "s13", "s21", "s22", "s23", "s41", "s42", "s43", "s51", "s52", "s53"]}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)
    

if __name__ == "__main__":
    main()"""