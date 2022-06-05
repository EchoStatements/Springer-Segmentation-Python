from scipy.signal import butter, filtfilt

def butterworth_low_pass_filter(original_signal,
                                 order,
                                 cutoff,
                                 sampling_frequency):
    B_low, A_low = butter(order, 2 * cutoff / sampling_frequency, btype="lowpass")
    
    # padlen made equivalent to matlabs using https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
    low_pass_filtered_signal = filtfilt(B_low, A_low, original_signal, padlen=3*(max(len(B_low),len(A_low))-1))
    return low_pass_filtered_signal
