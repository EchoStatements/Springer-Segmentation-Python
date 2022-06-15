import numpy as np
from sklearn.linear_model import LogisticRegression

def trainBandPiMatricesSpringer(state_observation_values):

    number_of_states = 4
    pi_vector = 0.25 * np.ones(4)

    B_matrix = []
    statei_values = [np.zeros((0, 4)) for _ in range(number_of_states)]

    for PCGi in range(4):
        for idx in range(4):
            statei_values[idx] = np.concatenate((statei_values, state_observation_values[PCGi][idx]))

    total_observation_sequence = np.concatenate(statei_values, axis=0)

    for state in range(number_of_states):

        length_of_state_samples = statei_values[idx].shape[0]

        length_per_other_state = np.floor(length_of_state_samples / (number_of_states - 1))

        min_length_other_class = np.inf

        for other_state in range(number_of_states):
            samples_in_other_state = statei_values[other_state].shape[0]

            if other_state != state:
                min_length_other_class = np.min(min_length_other_class, samples_in_other_state)

        if length_per_other_state > min_length_other_class:
            length_per_other_state = min_length_other_class

        training_data = [None, np.zeros((0, number_of_states))]

        for other_state in range(number_of_states):
            samples_in_other_state = statei_values[other_state].shape[0]

            if other_state == state:
                indices = np.random.permutation(samples_in_other_state)[int(length_per_other_state * (number_of_states -1)) + 1]
                training_data[0] = statei_values[other_state][indices, :]
            else:
                indices = np.random.permutation(samples_in_other_state)[:length_per_other_state + 1]
                state_data = statei_values[other_state][indices, :]
                training_data[1] = np.concatenate(training_data[1], state_data)

        labels = np.ones(training_data[0].shape[0] + training_data[2].shape[0])
        labels[0:training_data.shape[0]] = 2

        all_data = np.concatenate(training_data, axis=0)

        regressor = LogisticRegression()
        regressor.fit(all_data, labels)
        B = regressor.coef_
        B_matrix[state] = B

    # Might want to make B_matrix and actual ndarray rather than list of ndarrays
    return B_matrix, pi_vector, total_observation_sequence