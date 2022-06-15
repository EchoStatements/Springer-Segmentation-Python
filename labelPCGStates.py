import numpy as np
from scipy.signal import hilbert

def labelPCGStates(envelope, s1_positions, s2_positions, samplingFrequency):

    states = np.zeros(len(envelope))

    mean_S1 = 0.122 * samplingFrequency
    # std_S1 wasn't used in matlab version
    mean_S2 = 0.092 * samplingFrequency
    std_S2 = 0.022 * samplingFrequency

    for i in range(s1_positions.shape[0]):
        upper_bound = int(np.round(min(np.asarray(states.shape[0] - 1), s1_positions[i] + mean_S1)))

        # MIGHT BE AN OFF BY ONE ERROR HERE
        states[max(0, s1_positions[i]-1):min(upper_bound, states.shape[0] - 1) ] = 1

    for i in range(s2_positions.shape[0]):
        lower_bound = int(max(s2_positions[i] - np.floor((mean_S2 + std_S2)), 0))
        upper_bound = int(min(states.shape[0], np.ceil(s2_positions[i] + np.floor(mean_S2) + std_S2)))
        search_window = envelope[lower_bound-1:upper_bound] * (states[lower_bound-1:upper_bound] != 1)

        s2_index = np.argmax(search_window)

        # TODO: this needs verifying
        s2_index = min(states.shape[0], lower_bound + s2_index -1)
        upper_bound = int(min(states.shape[0], np.ceil(s2_index + (mean_S2) / 2)))
        states[np.max(int(np.ceil(s2_index - (mean_S2/2))), 0): upper_bound] = 3

        if i <= s2_positions.shape[0]-1:
            diffs = (s1_positions - s2_positions[i]).astype(float)
            diffs[diffs < 0] = np.inf

            if diffs[diffs < 0].shape[0] == 0:
                end_pos = states.shape[0]
            else:
                end_pos = np.argmin(diffs) - 1
            # there was a 0 * std in the line below that I've removed
            states[int(np.ceil(s2_index + (mean_S2))):end_pos]

    first_location_of_definite_state = np.argmax(states != 0)

    if first_location_of_definite_state > 0:
        if states[first_location_of_definite_state + 1] == 1:
            states[:first_location_of_definite_state] == 4
        else:
            states[:first_location_of_definite_state] == 2

    last_location_of_definite_state = (states.shape[0] - 1) - np.argmax(states[::-1] == 0)

    if last_location_of_definite_state > 0:
        if states[last_location_of_definite_state] == 1:
            states[last_location_of_definite_state:] = 2

        if states[last_location_of_definite_state] == 3:
            states[last_location_of_definite_state] =4

    # This is shakey too
    states = states[:envelope.shape[0]]

    states[states == 0] = 2

    return states
