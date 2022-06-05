import numpy as np

import default_Springer_HSMM_options as springer_options
from Hilbert_Envelope import Hilbert_Envelope
from butterworth_high_pass import butterworth_high_pass_filter
from butterworth_low_pass import butterworth_low_pass_filter
from get_PSD_feature_Springer_HMM import get_PSD_feature_Springer_HMM
from normalise_signal import normalise_signal
from schmidt_spike_removal import schmidt_spike_removal
from Homomorphic_Envelope_with_Hilbert import Homorphic_Envelope_with_Hilbert

from scipy.signal import resample

def getSpringerPCGFeatures(audio_data, Fs):

    include_wavelet = springer_options.include_wavelet_features
    featureFs = springer_options.audio_segmentation_Fs

    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, Fs)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, Fs)

    audio_data = schmidt_spike_removal(audio_data, Fs)

    homomorphic_envelope = Homorphic_Envelope_with_Hilbert(audio_data, Fs)
    downsampled_homomorphic_envelope = resample(homomorphic_envelope, featureFs, Fs)
    downsampled_homomorphic_envelope = normalise_signal(downsampled_homomorphic_envelope)

    hilbert_envelope = Hilbert_Envelope(audio_data, Fs)
    downsampled_hilbert_envelope = resample(hilbert_envelope, featureFs, Fs)
    downsampled_hilbert_envelope = normalise_signal(downsampled_hilbert_envelope)

    psd = get_PSD_feature_Springer_HMM(audio_data, Fs, 40, 60)
    psd = resample(psd, downsampled_homomorphic_envelope.shape[0], psd.shape[0])
    psd = normalise_signal(psd)

    return downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd
