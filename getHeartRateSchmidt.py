from Homomorphic_Envelope_with_Hilbert import Homomorphic_Envelope_with_Hilbert
from butterworth_low_pass import butterworth_low_pass_filter
from butterworth_high_pass import butterworth_high_pass_filter
from schmidt_spike_removal import schmidt_spike_removal

import numpy as np
from scipy.signal import correlate


def getHeartRateSchmidt(audio_data, Fs):

    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, Fs)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, Fs)

    audio_data = schmidt_spike_removal(audio_data, Fs)

    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)

    y = homomorphic_envelope - homomorphic_envelope.mean()

    c = correlate(y, y)
    c /= c[int(c.shape[0]/2)]

    # check size of this
    signal_autocorrelation = c[homomorphic_envelope.shape[0]:]
    min_idx = 0.5 * Fs
    max_idx = 2 * Fs
    index = np.argmax(signal_autocorrelation[min_idx: max_idx])
    true_idx = index + min_idx - 1

    heartRate = 60. / (true_idx / Fs)

    max_sys_duration = np.round(((60. / heartRate) * Fs) / 2)
    min_sys_duration = np.round(0.2 * Fs)

    # This is definitely wrong
    pos = np.argmax(signal_autocorrelation[min_sys_duration: max_sys_duration])
    systolicTimeInterval = (min_sys_duration + pos) / Fs

    return heartRate, systolicTimeInterval
