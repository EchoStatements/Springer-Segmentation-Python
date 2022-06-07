import default_Springer_HSMM_options as springer_options
from scipy.io import loadmat

example_data = loadmat("example_data.mat")

train_recordings = example_data.example_audio_data[:5]
train_annotations = example_data.example_annotation[:5, :]

test_recordings = example_data.example_audio_data[5]

B_matrix, pi_vector, total_obs_distribution = trainSprinerSegmentationAlgorithm(train_recordings, train_annotations, springer_options.audio_Fs)

numPCGs = len(test_recordings)

for PCGi in range(numPCGs):
    assigned_states = runSpringerSegmentationAlgorithm(test_recordings[PCGi], springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution)