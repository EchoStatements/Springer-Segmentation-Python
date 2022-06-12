import matlab.engine
import numpy as np

def matlab_spectrogram(data, sampling_frequency, eng=None):
    if eng is None:
        eng = matlab.engine.start_matlab()
    eng.addpath("../Springer-Segmentation-Code/")
    result = eng.spectrogram(matlab.double(data),
                             matlab.double(sampling_frequency / 40.),
                             matlab.double(np.round(sampling_frequency/79.)),
                             matlab.double(np.arange(1, np.round(sampling_frequency/2) + 1)),
                             matlab.double(1000), nargout=4)
    return result[-3], result[-2], result[-1]

if __name__ == "__main__":
    eng = matlab.engine.start_matlab()
    eng.addpath("../Springer-Segmentation-Code/")
    ml_recording = eng.load("recording1.mat")
    recording = np.asarray(ml_recording["r"])
    print(np.asarray(ml_recording["r"])[1])
    result = matlab_spectrogram(recording, 1000)
    print(np.asarray(result[-1]).shape)
    print(np.asarray(result[-1])[-1][-1])