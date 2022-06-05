
import numpy as np

def schmidt_spike_removal(original_signal, fs):

    windowsize = np.round(fs/2).astype(int)
    trailingsamples = (original_signal.shape[0] % windowsize).astype(int)
    if trailingsamples == 0:
        sample_frames = np.reshape(original_signal, (windowsize, -1))
    else:
        sample_frames = np.reshape(original_signal[:-trailingsamples], (windowsize, -1))

    MAAs = np.max(np.abs(sample_frames))

    while np.any(MAAs > np.median(MAAs) * 3):

        # Which window has the max MAAs
        window_num = np.argmax(MAAs)
        val = MAAs[window_num, :]

        # What is the position of the spike in the window
        spike_position = np.argmax(np.abs(sample_frames[:, val]))

        # Find zero crossings
        zero_crossings = np.abs( np.diff(np.sign(sample_frames[:, window_num])))>1
        zero_crossings = np.append(zero_crossings, 0)

        pre_spike_crossings = np.where(zero_crossings[:spike_position] == 1)
        if pre_spike_crossings[0].shape[0] == 0:
            spike_start = 0
        else:
            spike_start = pre_spike_crossings[0][-1]

        post_spike_crossings = np.where(zero_crossings[spike_position:] == 1)
        if post_spike_crossings[0].shape[0] == 0:
            spike_end = zero_crossings.shape[0] - 1
        else:
            spike_end = post_spike_crossings[0][0]

        sample_frames[spike_start:spike_end, window_num] = 0.0001

        MAAs = np.max(np.abs(sample_frames))

    despiked_signal = np.reshape(sample_frames, -1)
    despiked_signal = np.append(despiked_signal, original_signal[despiked_signal.shape[0]:])

    return despiked_signal



