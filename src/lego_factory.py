import random


class FactoryAutomata:
    def __init__(self):
        self.state = '_init_'
        self.transitions = {
            '_init_': {'s11': 'q_0'},
            'q_0': {'s15': 'q_17',
                    's12': 'q_1',
                    's13': 'q_2'},
            'q_1': {'s15': 'q_17',
                    's13': 'q_2',
                    's14': 'q_3'},
            'q_2': {'s21': 'q_4'},
            'q_3': {'s13': 'q_2'},
            'q_4': {'s22': 'q_5'},
            'q_5': {'s23': 'q_6'},
            'q_6': {'s41': 'q_7',
                    's31': 'q_14'},
            'q_7': {'s42': 'q_8'},
            'q_8': {'s43': 'q_9'},
            'q_9': {'s51': 'q_10'},
            'q_10': {'s52': 'q_11'},
            'q_11': {'s54': 'q_13',
                     's53': 'q_12'},
            'q_12': {'s11': 'q_0'},
            'q_13': {'s53': 'q_12'},
            'q_14': {'s32': 'q_15'},
            'q_15': {'s33': 'q_16'},
            'q_16': {'s51': 'q_10'},
            'q_17': {'s13': 'q_2'}
        }
        self.event_symbols = {
            's11': '_load_1',
            's12': '_process_1',
            's13': '_unload_1',
            's14': '_fail_1',
            's15': '_block_1',
            's21': '_load_2',
            's22': '_process_2',
            's23': '_unload_2',
            's24': '_block_2',
            's31': '_load_3',
            's32': '_process_3',
            's33': '_unload_3',
            's34': '_block_3',
            's41': '_load_4',
            's42': '_process_4',
            's43': '_unload_4',
            's44': '_block_4',
            's51': '_load_5',
            's52': '_process_5',
            's53': '_unload_5',
            's54': '_fail_5',
            's55': '_block_5'
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
        self.path.append(event)
        if event in self.transitions[self.state]:
            self.state = self.transitions[self.state][event]
            event_label = self.event_symbols[event]
            if event_label in self.failure_states:
                return f"Failure encountered at state {self.state}, Path: {self.path}"
            else:
                return f"Transitioned to {self.state}"
        else:
            return f"Invalid event {event} for state {self.state}"

    def predict_next_event(self):
        possible_events = list(self.transitions[self.state].keys())
        predicted_event = random.choice(possible_events)
        return f"Predicted next event: {predicted_event}"

    def verify_process_completion(self):
        if self.state == 'q_12':
            return "Process completed successfully in q_12!"
        else:
            return "Process did not complete as expected!"


class WeightedFactory(FactoryAutomata):
    def __init__(self):
        super().__init__()
        self.cost = 0

    def get_cost(self):
        return self.cost

    def trigger(self, event):
        self.path.append(event)
        transition_cost = self.get_transition_cost(event)
        self.cost += transition_cost
        return f"Cost after event {event}: {self.cost}"
        # super().trigger(event)


    def get_transition_cost(self, event):
        cost_mapping = {'s11': 3,
                        's12': 5,
                        's13': 3,
                        's14': 10,
                        's15': 8,
                        's21': 3,
                        's22': 5,
                        's23': 3,
                        's24': 8,
                        's31': 3,
                        's32': 5,
                        's33': 3,
                        's34': 8,
                        's41': 3,
                        's42': 5,
                        's43': 3,
                        's44': 8,
                        's51': 3,
                        's52': 5,
                        's53': 3,
                        's54': 10,
                        's55': 8
                        }
        return cost_mapping.get(event, cost_mapping[event])


"""def main():
    print('Simulate factory and trigger a failure.')
    factory = FactoryAutomata()
    print(factory.trigger('s11'))
    print(factory.trigger('s12'))
    print(factory.trigger('s14'))
    print('\n')

    print('Next event prediction')
    factory = FactoryAutomata()
    print(factory.trigger('s11'))
    print(factory.predict_next_event())
    print('\n')

    print('Simulate factory execution with costs.')
    opt_factory = WeightedFactory()
    opt_factory.trigger('s11')
    opt_factory.trigger('s12')
    opt_factory.trigger('s13')
    print('\n')

    print('Verify process completion (final state q12)')
    factory.trigger('s12')
    factory.trigger('s13')
    factory.trigger('s21')
    factory.trigger('s22')
    factory.trigger('s23')
    factory.trigger('s41')
    factory.trigger('s42')
    factory.trigger('s43')
    factory.trigger('s51')
    factory.trigger('s52')
    factory.trigger('s53')
    print(factory.verify_process_completion())


if __name__ == "__main__":
    main()"""
