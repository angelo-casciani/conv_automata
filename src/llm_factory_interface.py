from lego_factory import FactoryAutomata, WeightedFactory
from utility import extract_json


def trigger_failure_mode_analysis(events_sequence):
    factory = FactoryAutomata()
    results = ''
    for event in events_sequence:
        res = factory.trigger(event)
        results += f"{res}\n"
    return results


def trigger_event_prediction(events_sequence):
    factory = FactoryAutomata()
    results = ''
    for event in events_sequence:
        res = factory.trigger(event)
        results += f"{res}\n"
    results += factory.predict_next_event()
    return results


def trigger_verification(events_sequence):
    factory = FactoryAutomata()
    results = ''
    for event in events_sequence:
        res = factory.trigger(event)
        results += f"{res}\n"
    results += factory.verify_process_completion()
    return results


def compute_process_cost(events_sequence):
    weighted_factory = WeightedFactory()
    results = ''
    for event in events_sequence:
        res = weighted_factory.trigger(event)
        results += f"{res}\n"
    results += f'Total execution cost: {weighted_factory.get_cost()}'
    return results


def interface_with_llm(llm_answer):
    json_request = extract_json(llm_answer)
    task = json_request.get("task")
    events_sequence = json_request.get("events_sequence")
    factory_output = ''

    if task == "failure_mode_analysis":
        return trigger_failure_mode_analysis(events_sequence)
    elif task == "event_prediction":
        return trigger_event_prediction(events_sequence)
    elif task == "process_cost":
        return compute_process_cost(events_sequence)
    elif task == "verification":
        return trigger_verification(events_sequence)
    
    json_request["results"] = factory_output
    print(json_request)
    return json_request


"""def main():
    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "failure_mode_analysis", "events_sequence": ["s11", "s12", "s14"]}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)

    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "event_prediction", "events_sequence": ["s11"]}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)

    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "verification", "events_sequence": ["s11", "s12", "s13", "s21", "s22", "s23", "s41", "s42", "s43", "s51", "s52", "s53"]}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)

    llm_input = 'Given the user request, this is the JSON to invoke the factory: {"task": "process_cost", "events_sequence": ["s11", "s12", "s14"], "results": null}'
    factory_output = interface_with_llm(llm_input)
    print(factory_output)

if __name__ == "__main__":
    main()"""