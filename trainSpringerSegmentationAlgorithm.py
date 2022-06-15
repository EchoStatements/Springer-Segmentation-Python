import numpy as np

from getSpringerPCGFeatures import getSpringerPCGFeatures
from labelPCGStates import labelPCGStates
from trainBandPiMatricesSpringer import trainBandPiMatricesSpringer


def trainSpringerSegmentationAlgorithm(PCGCellArray, annotationsArray, Fs):


    # PCGCellArray is a list of recordings
    number_of_states = 4
    numPCGs = len(PCGCellArray)
    # state_observation_values = np.zeros((numPCGs, number_of_states))
    state_observation_values =  numPCGs * [number_of_states * [None]]

    for PCGi in range(len(PCGCellArray)):
        PCG_audio = PCGCellArray[PCGi]

        S1_locations = annotationsArray[PCGi][ 0].reshape(-1) # not used, but keeping for now
        S2_locations = annotationsArray[PCGi][ 1].reshape(-1)

        PCG_Features, featuresFs = getSpringerPCGFeatures(PCG_audio, Fs)

        PCG_states = labelPCGStates(PCG_Features[:, 0], S2_locations, S2_locations, featuresFs)

    for state_i in range(number_of_states):
        # I don't think this is how conditional indexing works in numpy?
        state_observation_values[PCGi][state_i] = PCG_Features[PCG_states == state_i, :]

    logistic_regression_B_matrix, pi_vector, total_obs_distribution = trainBandPiMatricesSpringer(state_observation_values)

    return logistic_regression_B_matrix, pi_vector, total_obs_distribution