import numpy as np

def normalise_signal(signal):

    mean_of_signal = np.mean(signal)

    standard_deviation = np.std(signal)

    normalised_signal = (signal - mean_of_signal) / standard_deviation

    return normalised_signal