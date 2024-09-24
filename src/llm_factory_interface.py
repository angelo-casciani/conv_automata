from lego_factory import FactoryAutomata, WeightedFactory

def trigger_failure_mode_analysis(events_sequence):
    factory = FactoryAutomata()
    for event in events_sequence:
        factory.trigger(event)

def trigger_event_prediction(factory):
    return factory.predict_next_event()

def trigger_verification(factory):
    factory.verify_process_completion()

def compute_process_cost(weighted_factory, events_sequence):
    for event in events_sequence:
        weighted_factory.trigger(event)
    return weighted_factory.get_cost()

def interface_with_llm(task, factory_fsm, events_sequence=None):
    if task == "failure_mode_analysis":
        trigger_failure_mode_analysis(factory_fsm, events_sequence)
    elif task == "event_prediction":
        return trigger_event_prediction(factory_fsm)
    elif task == "process_cost":
        return compute_process_cost(factory_fsm, events_sequence)
    elif task == "verification":
        trigger_verification(factory_fsm)