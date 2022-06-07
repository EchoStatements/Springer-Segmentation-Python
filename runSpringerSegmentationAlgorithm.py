from getHeartRateSchmidt import getHeartRateSchmidt
from getSpringerPCGFeatures import getSpringerPCGFeatures
from viterbiDecodePCG_Springer import viterbiDecodePCG_Springer


def runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, total_observation_distribution):

    PCG_features, featuresFs = getSpringerPCGFeatures(audio_data, Fs)

    heartRate, systolicTimeInterval = getHeartRateSchmidt(audio_data, Fs)

    _, _, qt = viterbiDecodePCG_Springer(PCG_features, pi_vector, B_matrix, total_observation_distribution, heartRate, systolicTimeInterval, featuresFs)

    assigned_states = expand_qt(qt, featuresFs, Fs, audio_data.shape[0])