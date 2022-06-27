import numpy as np

import default_Springer_HSMM_options as springer_options
from Hilbert_Envelope import Hilbert_Envelope
from butterworth_high_pass import butterworth_high_pass_filter
from butterworth_low_pass import butterworth_low_pass_filter
from get_PSD_feature_Springer_HMM import get_PSD_feature_Springer_HMM
from normalise_signal import normalise_signal
from schmidt_spike_removal import schmidt_spike_removal
from Homomorphic_Envelope_with_Hilbert import Homomorphic_Envelope_with_Hilbert
from getDWT import getDWT

from scipy.signal import resample

def getSpringerPCGFeatures(audio_data, Fs, matlab_psd=False):

    include_wavelet = springer_options.include_wavelet_features
    featureFs = springer_options.audio_segmentation_Fs

    audio_data = butterworth_low_pass_filter(audio_data, 2, 400, Fs)
    audio_data = butterworth_high_pass_filter(audio_data, 2, 25, Fs)

    audio_data = schmidt_spike_removal(audio_data, Fs)

    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)
    downsampled_homomorphic_envelope = resample(homomorphic_envelope, int(np.round(homomorphic_envelope.shape[0] * featureFs /Fs)))
    downsampled_homomorphic_envelope = normalise_signal(downsampled_homomorphic_envelope)

    hilbert_envelope = Hilbert_Envelope(audio_data)
    downsampled_hilbert_envelope = resample(hilbert_envelope, int(np.round(hilbert_envelope.shape[0] * featureFs /Fs)))
    downsampled_hilbert_envelope = normalise_signal(downsampled_hilbert_envelope)

    psd = get_PSD_feature_Springer_HMM(audio_data, Fs, 40, 60, use_matlab=matlab_psd)
    psd = resample(psd, int(downsampled_homomorphic_envelope.shape[0]))
    psd = normalise_signal(psd)
    
    #wavelet features
    wavelet_level = 3
    wavelet_name = "rbio3.9"
    
    #audio needs to be longer than 1 second for getDWT to work
    cD, cA = getDWT(audio_data, wavelet_level, wavelet_name)
    
    #wavelet_feature = ...
    #wavelet_feature = ...
    downsampled_wavelet = []
    #downsampled_wavelet = ...
    #downsampled_wavelet = ...

    features = np.stack((downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd, downsampled_wavelet), axis=-1)
    return features, featureFs
