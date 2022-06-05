import numpy as np
from scipy.signal import hilbert

def Hilbert_Envelope(input_signal):

    hilbert_envelope = np.abs(hilbert(input_signal))

    return hilbert_envelope