from scipy.signal import butter, filtfilt

def butterworth_high_pass_filter(original_signal,
                                 order,
                                 cutoff,
                                 sampling_frequency):
    B_high, A_high = butter(order, 2 * cutoff / sampling_frequency, btype="highpass")
    high_pass_filtered_signal = filtfilt(B_high, A_high, original_signal, padlen=3*(max(len(B_high),len(A_high))-1))
    return high_pass_filtered_signal
