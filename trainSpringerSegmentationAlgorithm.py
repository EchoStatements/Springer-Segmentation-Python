import numpy as np

from getSpringerPCGFeatures import getSpringerPCGFeatures
from labelPCGStates import labelPCGStates
from trainBandPiMatricesSpringer import trainBandPiMatricesSpringer


def trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, Fs):


    # PCGCellArray is a list of recordings
    number_of_states = 4
    numPCGs = len(PCGCellArray)
    # state_observation_values = np.zeros((numPCGs, number_of_states))
    state_observation_values = []

    for PCGi in range(len(PCGCellArray)):
        PCG_audio = PCGCellArray[PCGi]

        S1_locations = annotationsArray[PCGi][ 0].reshape(-1)
        S2_locations = annotationsArray[PCGi][ 1].reshape(-1)

        PCG_Features, featuresFs = getSpringerPCGFeatures(PCG_audio, Fs)

        PCG_states = labelPCGStates(PCG_Features[:, 0], S1_locations, S2_locations, featuresFs)

        these_state_observations = []
        for state_i in range(1, number_of_states + 1):
            these_state_observations.append(PCG_Features[PCG_states == state_i, :])
        state_observation_values.append(these_state_observations)

    logistic_regression_B_matrix, pi_vector, total_obs_distribution, models = trainBandPiMatricesSpringer(state_observation_values)

    return logistic_regression_B_matrix, pi_vector, total_obs_distribution, models