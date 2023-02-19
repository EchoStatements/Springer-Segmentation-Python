import numpy as np
from scipy import signal as signal
from scipy.signal import butter, filtfilt, hilbert
from get_dwt import getDWT
from librosa import resample

def preprocess_audio(audio, process_list):
    """
    Constructs pre-processing pipeline to be applied to audio before features are generated.

    See `get_default_features` for example of use.

    Parameters
    ----------
    audio : ndarray
        numpy array of audio recording
    process_list : list of dicts
        List of processes to be applied to the signal. Each entry is a dict with a name (corresponding to the
        function/filter to be applied), args (the positional arguments to be given to the function) and
        kwargs (the keyword arguments to be given to the function).

    Returns
    -------

    """
    for item in process_list:
        name = item["function"]
        args = item["args"]
        kwargs = item["kwargs"]
        if name == "butterworth_high":
            audio = get_butterworth_high_pass_filter(audio, *args, **kwargs)
        if name == "butterworth_low":
            audio = get_butterworth_low_pass_filter(audio, *args, **kwargs)
        if name == "homomorphic_envelope":
            audio = get_homomorphic_envelope_with_hilbert(audio, *args, **kwargs)
        if name == "hilbert_envelope":
            audio = get_hilbert_envelope(audio, *args, **kwargs)
        if name == "psd":
            audio = get_power_spectral_density(audio, *args, **kwargs)
        if name == "schmidt_spike":
            audio = schmidt_spike_removal(audio, *args, **kwargs)
        if callable(name):
            audio = name(audio, *args, **kwargs)
    return audio

def collect_features(audio, audio_sample_frequency, feature_dict, feature_frequency=50):
    """
    Creates an array of features based on the contents of feature_dict.

    See `get_default_features` for example of use.

    Parameters
    ----------
    audio : ndarray
        The (preprocessed) recording from which features are to be derived.
    audio_sample_frequency : int
        The sample frequency of the audio signal.
    feature_dict : dict
        Dictionary of features to be generatedj
    feature_frequency : int
        Number of features per second of recording

    Returns
    -------

    """
    outputs = []
    desired_output_length = np.ceil(feature_frequency * len(audio) / audio_sample_frequency)
    for key, value in feature_dict.items():
        if key == "butterworth_high":
            output = get_butterworth_high_pass_filter(audio, **value)
        if key == "butterworth_low":
            output = get_butterworth_high_pass_filter(audio, **value)
        if key == "homomorphic_envelope":
            output = get_homomorphic_envelope_with_hilbert(audio, **value)
            output = resample(output, orig_sr=audio_sample_frequency, target_sr=feature_frequency)
        if key == "hilbert_envelope":
            output = get_hilbert_envelope(audio, **value)
            output = resample(output, orig_sr=audio_sample_frequency, target_sr=feature_frequency)
        if key == "psd":
            output = get_power_spectral_density(audio, **value)
            output = resample(output, orig_sr=audio_sample_frequency, target_sr=feature_frequency)
            if output.shape[0] != desired_output_length:
                output = resample(output, orig_sr=output.shape[0] + 1e-9, target_sr=desired_output_length, fix=True)
        if key == "wavelet":
            output = get_wavelet(audio, **value)
            output = resample(output, orig_sr=audio_sample_frequency, target_sr=feature_frequency)
        if callable(key):
            output = key(audio, **value)
        output = normalise_signal(output)
        outputs.append(output)
    features = np.stack(outputs, axis=-1)
    return features


def get_default_features(audio, sample_frequency):
    """
    Default preprocessing and feature generation from audio

    Parameters
    ----------
    audio
    sample_frequency

    Returns
    -------

    """
    process_list = [{"function": "butterworth_low", "args" : [2, 100, sample_frequency], "kwargs" : {}},
                     {"function": "butterworth_high", "args" : [2, 25, sample_frequency], "kwargs" : {}},
                     {"function": "schmidt_spike", "args" : [sample_frequency], "kwargs" : {}},]
    audio = preprocess_audio(audio, process_list=process_list)

    feature_dict = {"homomorphic_envelope" : {"sampling_frequency" : sample_frequency},
                    "hilbert_envelope" : {},
                    "psd" : {"sampling_frequency" : sample_frequency,
                             "frequency_limit_low" : 40,
                             "frequency_limit_high" : 60},
                    "wavelet" : {"sample_frequency" : sample_frequency}
                    }


    features = collect_features(audio, audio_sample_frequency=sample_frequency, feature_dict=feature_dict)
    return features

def get_all_features(audio_data,
                     Fs,
                     use_psd=True,
                     use_wavelet=True,
                     featureFs=50):
    """
    DEPRECATED

    Parameters
    ----------
    audio_data
    Fs
    matlab_psd
    use_psd
    use_wavelet
    featureFs

    Returns
    -------

    """

    audio_data = get_butterworth_low_pass_filter(audio_data, 2, 100, Fs)
    audio_data = get_butterworth_high_pass_filter(audio_data, 2, 25, Fs)
    audio_data = schmidt_spike_removal(audio_data, Fs)

    all_features = []

    homomorphic_envelope = get_homomorphic_envelope_with_hilbert(audio_data, Fs)
    # downsampled_homomorphic_envelope = resample(homomorphic_envelope, int(np.round(homomorphic_envelope.shape[0] * featureFs /recording_frequency)))
    downsampled_homomorphic_envelope = resample(homomorphic_envelope, orig_sr=Fs, target_sr=featureFs)
    downsampled_homomorphic_envelope = normalise_signal(downsampled_homomorphic_envelope)
    all_features.append(downsampled_homomorphic_envelope)

    hilbert_envelope = get_hilbert_envelope(audio_data)
    # downsampled_hilbert_envelope = resample(hilbert_envelope, int(np.round(hilbert_envelope.shape[0] * featureFs /recording_frequency)))
    downsampled_hilbert_envelope = resample(hilbert_envelope, orig_sr=Fs, target_sr=featureFs)
    downsampled_hilbert_envelope = normalise_signal(downsampled_hilbert_envelope)
    all_features.append(downsampled_hilbert_envelope)

    if use_psd:
        psd = get_power_spectral_density(audio_data, Fs, 40, 60, )
        psd = psd / 2
        psd = resample(psd,
                       orig_sr=(1+1e-9),
                       target_sr=downsampled_homomorphic_envelope.shape[0] / len(psd))
        # psd = librosa.util.fix_length(psd, size=downsampled_hilbert_envelope.shape[0], mode="edge")
        psd = normalise_signal(psd)
        all_features.append(psd)

    # wavelet features
    if use_wavelet:
        wavelet_level = 3
        wavelet_name = "rbio3.9"

        if len(audio_data) < Fs * 1.025:
            audio_data = np.concatenate((audio_data, np.zeros((round(0.025 * Fs)))))

        # audio needs to be longer than 1 second for getDWT to work
        cD, cA = getDWT(audio_data, wavelet_level, wavelet_name)

        wavelet_feature = abs(cD[wavelet_level - 1, :])
        wavelet_feature = wavelet_feature[:len(homomorphic_envelope)]

        downsampled_wavelet = resample(wavelet_feature, orig_sr=Fs, target_sr=featureFs)
        downsampled_wavelet = normalise_signal(downsampled_wavelet)
        all_features.append(downsampled_wavelet)

    features = np.stack(all_features, axis=-1)
    return features

def get_wavelet(audio_data, sample_frequency):
    wavelet_level = 3
    wavelet_name = "rbio3.9"

    if len(audio_data) < sample_frequency * 1.025:
        audio_data = np.concatenate((audio_data, np.zeros((round(0.025 * sample_frequency)))))

    # audio needs to be longer than 1 second for getDWT to work
    cD, cA = getDWT(audio_data, wavelet_level, wavelet_name)

    wavelet_feature = abs(cD[wavelet_level - 1, :])
    return wavelet_feature


def get_butterworth_high_pass_filter(original_signal,
                                     order,
                                     cutoff,
                                     sampling_frequency):
    """

    Parameters
    ----------
    original_signal
    order
    cutoff
    sampling_frequency

    Returns
    -------

    """
    B_high, A_high = butter(order, 2 * cutoff / sampling_frequency, btype="highpass")
    high_pass_filtered_signal = filtfilt(B_high, A_high, original_signal, padlen=3*(max(len(B_high),len(A_high))-1))
    return high_pass_filtered_signal


def get_butterworth_low_pass_filter(original_signal,
                                    order,
                                    cutoff,
                                    sampling_frequency):
    """

    Parameters
    ----------
    original_signal
    order
    cutoff
    sampling_frequency

    Returns
    -------

    """
    B_low, A_low = butter(order, 2 * cutoff / sampling_frequency, btype="lowpass")

    # padlen made equivalent to matlabs using https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
    low_pass_filtered_signal = filtfilt(B_low, A_low, original_signal, padlen=3*(max(len(B_low),len(A_low))-1))
    return low_pass_filtered_signal


def get_homomorphic_envelope_with_hilbert(input_signal, sampling_frequency, lpf_frequency=8):
    """

    Parameters
    ----------
    input_signal
    sampling_frequency
    lpf_frequency

    Returns
    -------

    """

    B_low, A_low = butter(1, 2 * lpf_frequency / sampling_frequency, btype="low")
    homomorphic_envelope = np.exp(filtfilt(B_low, A_low, np.log(np.abs(hilbert(input_signal))), padlen=3*(max(len(B_low),len(A_low))-1)))

    # Remove spurious spikes in first sample
    homomorphic_envelope[0] = homomorphic_envelope[1]

    return homomorphic_envelope


def get_hilbert_envelope(input_signal):
    """

    Parameters
    ----------
    input_signal

    Returns
    -------

    """

    hilbert_envelope = np.abs(hilbert(input_signal))

    return hilbert_envelope


def get_power_spectral_density(data, sampling_frequency, frequency_limit_low, frequency_limit_high):
    """

    Parameters
    ----------
    data
    sampling_frequency
    frequency_limit_low
    frequency_limit_high
    use_matlab

    Returns
    -------

    """
    # note that hamming window is implicit in the matlab function - this might be what was messing up the shapes
    f, t, Sxx = signal.spectrogram(data, sampling_frequency, window=('hamming'), nperseg=int(sampling_frequency / 41),
                                   noverlap=int(sampling_frequency / 81), nfft=sampling_frequency)
    # ignore the DC component - springer does this by returning freqs from 1 to round(sampling_frequency/2). We do the same by removing the first row.
    Sxx = Sxx[1:, :]

    low_limit_position = np.where(f == frequency_limit_low)
    high_limit_position = np.where(f == frequency_limit_high)

    psd = np.mean(Sxx[low_limit_position[0][0]:high_limit_position[0][0]+1, :], axis=0)

    return psd


def normalise_signal(signal):
    """

    Parameters
    ----------
    signal

    Returns
    -------

    """

    mean_of_signal = np.mean(signal)

    standard_deviation = np.std(signal)

    normalised_signal = (signal - mean_of_signal) / standard_deviation

    return normalised_signal


def schmidt_spike_removal(original_signal, fs):
    """

    % The spike removal process works as follows:
    % (1) The recording is divided into 500 ms windows.
    % (2) The maximum absolute amplitude (MAA) in each window is found.
    % (3) If at least one MAA exceeds three times the median value of the MAA's,
    % the following steps were carried out. If not continue to point 4.
    % (a) The window with the highest MAA was chosen.
    % (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
    % (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
    % (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
    % (e) The defined noise spike was replaced by zeroes.
    % (f) Resume at step 2.
    % (4) Procedure completed.
    %

    Parameters
    ----------
    original_signal : nd_array of shape (recording_length,)
    fs : float
        Sampling Frequency

    Returns
    -------

    """

    window_size = np.round(fs / 2).astype(int)
    trailing_samples = (original_signal.shape[0] % window_size).astype(int)
    if trailing_samples == 0:
        sample_frames = np.reshape(original_signal, (window_size, -1))
    else:
        sample_frames = np.reshape(original_signal[:-trailing_samples], (window_size, -1))

    MAAs = np.max(np.abs(sample_frames))

    while np.any(MAAs > np.median(MAAs) * 3):

        # Which window has the max MAAs
        window_num = np.argmax(MAAs)
        val = MAAs[window_num, :]

        # What is the position of the spike in the window
        spike_position = np.argmax(np.abs(sample_frames[:, val]))

        # Find zero crossings
        zero_crossings = np.abs(np.diff(np.sign(sample_frames[:, window_num]))) > 1
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



