import numpy as np

def expand_qt(original_qt, old_fs, new_fs, new_length):

    original_qt = original_qt.reshape(-1)
    expanded_qt = np.zeros(new_length)

    indices_of_changes =  np.nonzero(np.diff(original_qt))[0]

    start_index = 0
    for i in range(len(indices_of_changes)):
        end_index = indices_of_changes[i]

        mid_point = int(np.round((end_index - start_index) / 2) + start_index)

        value_at_midpoint = original_qt[mid_point]

        expanded_start_index = int(np.round(start_index / old_fs * new_fs))
        expanded_end_index = int(np.round(start_index) / old_fs * new_fs) - 1

        if expanded_end_index > new_length:
            expanded_end_index = new_length

        expanded_qt[expanded_start_index:expanded_end_index] = value_at_midpoint
        start_index = end_index

    return expanded_qt
