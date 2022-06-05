import numpy as np
from matplotlib.pyplot import specgram

"""
def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high):

    # TODO: This seems horribly wrong
    F, T, P, _ = specgram(data, window=np.hamming(500),
                       NFFT=500,
                       noverlap=250,
                       Fs=sampling_frequency
                       )

    low_lim_position = np.argmin(np.abs(F - frequency_limit_low))
    high_lim_position = np.argmin(np.abs(F - frequency_limit_high))

    psd = np.mean(P[low_lim_position:high_lim_position])

    return psd
"""

import scipy.signal as signal

def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high):
    # note that hamming window is implicit in the matlab function - this might be what was messing up the shapes
    f, t, Sxx = signal.spectrogram(data, sampling_frequency, window=('hamming'), nperseg=int(sampling_frequency / 41),
                                   noverlap=int(sampling_frequency / 81), nfft=sampling_frequency)

    # ignore the DC component - springer does this by returning freqs from 1 to round(sampling_frequency/2). We do the same by removing the first row.
    Sxx = Sxx[1:, :]

    low_limit_position = np.where(f == frequency_limit_low)
    high_limit_position = np.where(f == frequency_limit_high)

    # Find the mean PSD over the frequency range of interest:
    psd = np.mean(Sxx[low_limit_position[0][0]:high_limit_position[0][0], :], axis=0)

    return psd