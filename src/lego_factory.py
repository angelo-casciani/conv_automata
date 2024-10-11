import random
from utility import retrieve_automata


class FactoryAutomata:
    def __init__(self, env):
        self.env = env
        self.state = '_init_'
        data = retrieve_automata()
        self.transitions = data['transitions']
        self.event_symbols = data['event_symbols']
        self.event_probabilities = data['event_probabilities']
        self.failure_states = data['failure_states']
        self.path = []

    def get_state(self):
        return self.state

    def get_transitions(self):
        return self.transitions

    def get_event_symbols(self):
        return self.event_symbols

    def get_failure_states(self):
        return self.failure_states

    def get_path(self):
        return self.path

    def trigger(self, event):
        if event in self.transitions[self.state]:
            # Simulate a delay for each transition
            yield self.env.timeout(random.randint(1, 3))

            self.state = self.transitions[self.state][event]
            self.path.append(event)
            event_label = self.event_symbols[event]
            
            print(f"{self.env.now}: Transitioned to {self.state} after event {event}")

            if event_label in self.failure_states:
                print(f"{self.env.now}: Failure encountered at state {self.state}, Path: {self.path}")
                yield self.env.process(self.handle_failure())
        else:
            print(f"{self.env.now}: Invalid event {event} for state {self.state}")


    def handle_failure(self):
        # Simulate repair time for failure
        repair_duration = random.randint(5, 15)
        print(f"{self.env.now}: Repairing failure, this will take {repair_duration} units of time.")
        yield self.env.timeout(repair_duration)
        print(f"{self.env.now}: Repair completed. Resuming operations.")

    def run_factory(self):
        # Simulate the execution until completion (q_12)
        while self.state != 'q_12':
            next_event = self.predict_next_event()
            yield self.env.process(self.trigger(next_event))
            yield self.env.timeout(random.randint(1, 3))

    def predict_next_event(self):
        # Predict the next event based on the current state's probabilities.
        possible_events = list(self.transitions[self.state].keys())
        #return random.choice(possible_events)
        probabilities = [self.event_probabilities[self.state][event] for event in possible_events]
        next_event = random.choices(possible_events, weights=probabilities, k=1)[0]
        print(f"{self.env.now}: Predicted next event to execute: {next_event}")
        return next_event

    def execute_event_sequence(self, event_sequence):
        # Simulate the execution of a specified sequence of events.
        for event in event_sequence:
            yield self.env.process(self.trigger(event))
            yield self.env.timeout(random.randint(1, 3))


class WeightedFactory(FactoryAutomata):
    def __init__(self, env):
        super().__init__(env)
        self.cost = 0

    def get_cost(self):
        return self.cost

    def trigger(self, event):
        transition_cost = self.get_transition_cost(event)
        self.cost += transition_cost
        yield from super().trigger(event)
        print(f"{self.env.now}: Total execution cost after event {event}: {self.cost}")

    def get_transition_cost(self, event):
        data = retrieve_automata()
        cost_mapping = data['cost_mapping']
        return cost_mapping.get(event, 0)


"""def main():
    print('Simulate factory and trigger a failure.')
    env = simpy.Environment()
    #factory = FactoryAutomata(env)
    factory = WeightedFactory(env)
    event_sequence = ['s11', 's12', 's13', 's21', 's22', 's23']

    # Run the factory simulation process
    #env.process(factory.run_factory())
    env.process(factory.execute_event_sequence(event_sequence))
    env.run(until=100)

    print('Next event prediction')
    print(factory.predict_next_event())
    print('\n')

if __name__ == "__main__":
    main()"""