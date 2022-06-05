import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency, lpf_frequency=8):

    B_low, A_low = butter(1, 2 * lpf_frequency / sampling_frequency, btype="low")
    homomorphic_envelope = np.exp(filtfilt(B_low, A_low, np.log(np.abs(hilbert(input_signal))), padlen=3*(max(len(B_low),len(A_low))-1)))

    # Remove spurious spikes in first sample
    homomorphic_envelope[0] = homomorphic_envelope[1]

    return homomorphic_envelope