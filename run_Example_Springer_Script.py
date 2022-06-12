import default_Springer_HSMM_options as springer_options
from runSpringerSegmentationAlgorithm import runSpringerSegmentationAlgorithm
from trainSpringerSegmentationAlgorithm import trainSpringerSegmentationAlgorithm
from scipy.io import loadmat
import sys
sys.path.append("../Springer-Segmentation-Code")

example_data = loadmat("../Springer-Segmentation-Code/example_data.mat")["example_data"]
example_audio_data = example_data[0][0][0]
example_annotations = example_data[0][0][1]

# train_recordings = example_data.example_audio_data[:5]
train_recordings = [example_audio_data[:, idx][0].reshape(-1) for idx in range(5)]
# train_annotations = example_data.example_annotation[:5, :]
train_annotations = [ [example_annotations[idx, 0], example_annotations[idx, 1]] for idx in range(5) ]

test_recordings = example_audio_data[:, 5][0].reshape(-1)
test_recordings_list = [example_audio_data[: -1]]

B_matrix, pi_vector, total_obs_distribution = trainSpringerSegmentationAlgorithm(train_recordings, train_annotations, springer_options.audio_Fs)

numPCGs = len(test_recordings)

for PCGi in range(numPCGs):
    assigned_states = runSpringerSegmentationAlgorithm(test_recordings[PCGi], springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution)