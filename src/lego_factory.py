import random

class FactoryAutomata:
    def __init__(self, env):
        self.env = env
        self.state = '_init_'
        self.transitions = {
            '_init_': {'s11': 'q_0'},
            'q_0': {'s15': 'q_17', 's12': 'q_1', 's13': 'q_2'},
            'q_1': {'s15': 'q_17', 's13': 'q_2', 's14': 'q_3'},
            'q_2': {'s21': 'q_4'},
            'q_3': {'s13': 'q_2'},
            'q_4': {'s22': 'q_5'},
            'q_5': {'s23': 'q_6'},
            'q_6': {'s41': 'q_7', 's31': 'q_14'},
            'q_7': {'s42': 'q_8'},
            'q_8': {'s43': 'q_9'},
            'q_9': {'s51': 'q_10'},
            'q_10': {'s52': 'q_11'},
            'q_11': {'s54': 'q_13', 's53': 'q_12'},
            'q_12': {'s11': 'q_0'},
            'q_13': {'s53': 'q_12'},
            'q_14': {'s32': 'q_15'},
            'q_15': {'s33': 'q_16'},
            'q_16': {'s51': 'q_10'},
            'q_17': {'s13': 'q_2'}
        }
        self.event_symbols = {
            's11': '_load_1', 's12': '_process_1', 's13': '_unload_1',
            's14': '_fail_1', 's15': '_block_1', 's21': '_load_2',
            's22': '_process_2', 's23': '_unload_2', 's24': '_block_2',
            's31': '_load_3', 's32': '_process_3', 's33': '_unload_3',
            's34': '_block_3', 's41': '_load_4', 's42': '_process_4',
            's43': '_unload_4', 's44': '_block_4', 's51': '_load_5',
            's52': '_process_5', 's53': '_unload_5', 's54': '_fail_5',
            's55': '_block_5'
        }
        self.event_probabilities = {
            '_init_': {'s11': 1},
            'q_0': {'s15': 0.2, 's12': 0.5, 's13': 0.3},
            'q_1': {'s15': 0.3, 's13': 0.5, 's14': 0.2},
            'q_2': {'s21': 1}, 'q_3': {'s13': 1}, 'q_4': {'s22': 1},
            'q_5': {'s23': 1}, 'q_6': {'s41': 0.5, 's31': 0.5},
            'q_7': {'s42': 1}, 'q_8': {'s43': 1}, 'q_9': {'s51': 1},
            'q_10': {'s52': 1}, 'q_11': {'s54': 0.2, 's53': 0.8},
            'q_12': {'s11': 1}, 'q_13': {'s53': 1}, 'q_14': {'s32': 1},
            'q_15': {'s33': 1}, 'q_16': {'s51': 1}, 'q_17': {'s13': 1}
        }
        self.failure_states = ['_fail_1', '_fail_5']
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
        print(f"{self.env.now}: Cost after event {event}: {self.cost}")

    def get_transition_cost(self, event):
        cost_mapping = {'s11': 3, 's12': 5, 's13': 3, 's14': 10, 's15': 8,
                        's21': 3, 's22': 5, 's23': 3, 's24': 8,
                        's31': 3, 's32': 5, 's33': 3, 's34': 8,
                        's41': 3, 's42': 5, 's43': 3, 's44': 8,
                        's51': 3, 's52': 5, 's53': 3, 's54': 10, 's55': 8}
        return cost_mapping.get(event, 0)

"""
def main():
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
    main()
"""