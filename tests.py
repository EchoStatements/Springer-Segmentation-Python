import unittest

import matlab.engine
import numpy as np


class PreprocessingTesting(unittest.TestCase):

    def setUp(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath("../Springer-Segmentation-Code/")

        self.ml_recording = self.eng.load("recording1.mat")
        self.recording = np.asarray(self.ml_recording["r"]).reshape(-1)

    def test_butterworth_filter_code(self):
        from butterworth_low_pass import butterworth_low_pass_filter
        from butterworth_high_pass import butterworth_high_pass_filter

        python_result = butterworth_low_pass_filter(self.recording, 2, 400, 1000)
        matlab_result = self.eng.butterworth_low_pass_filter(self.ml_recording["r"], 2., 400., 1000.)

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result).reshape(-1)))

        python_result = butterworth_high_pass_filter(self.recording, 2, 25, 1000)
        matlab_result = self.eng.butterworth_high_pass_filter(self.ml_recording["r"], 2., 25., 1000.)

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result).reshape(-1)))

    def test_spike_removal(self):
        from schmidt_spike_removal import schmidt_spike_removal

        python_result = schmidt_spike_removal(self.recording, 1000)
        matlab_result = self.eng.schmidt_spike_removal(self.ml_recording["r"], 1000.)

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result).reshape(-1)))

    def test_homomorphic(self):
        from Homomorphic_Envelope_with_Hilbert import Homomorphic_Envelope_with_Hilbert

        python_result = Homomorphic_Envelope_with_Hilbert(self.recording, 1000)
        matlab_result = self.eng.Homomorphic_Envelope_with_Hilbert(self.ml_recording["r"], 1000.)

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result).reshape(-1)))

    def test_normalise(self):
        from normalise_signal import normalise_signal

        python_result = normalise_signal(self.recording)
        matlab_result = normalise_signal(self.ml_recording["r"])

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result).reshape(-1)))

    def test_hilbert(self):
        from Hilbert_Envelope import Hilbert_Envelope

        python_result = Hilbert_Envelope(self.recording)
        matlab_result = self.eng.Hilbert_Envelope(self.ml_recording["r"], 1000.)

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result).reshape(-1)))

