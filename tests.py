import unittest

import matlab.engine
import numpy as np

from get_duration_distributions import get_duration_distributions


class DurationDistributionTest(unittest.TestCase):

    def setUp(self):
        self.eng = matlab.engine.start_matlab()

        self.eng.addpath("../Springer-Segmentation-Code/")
        self.HEART_RATE = 69.7674
        self.SYSTOLIC_TIME = 0.342

    def test_duration_distribution_code(self):
        d_distributions, max_S1, min_S1,\
        max_S2, min_S2, max_systole, min_systole,\
        max_diastole, min_diastole = get_duration_distributions(self.HEART_RATE, self.SYSTOLIC_TIME)

        ml_d_distributions, ml_max_S1, ml_min_S1, \
        ml_max_S2, ml_min_S2, \
        ml_max_systole, ml_min_systole, \
        ml_max_diastole, ml_min_diastole = self.eng.get_duration_distributions_wrapper(self.HEART_RATE, self.SYSTOLIC_TIME, nargout=9)
        
        self.assertTrue(np.allclose(max_S1, np.asarray(ml_max_S1).reshape(-1)))
        self.assertTrue(np.allclose(min_S1, np.asarray(ml_min_S1).reshape(-1)))
        self.assertTrue(np.allclose(max_S2, np.asarray(ml_max_S2).reshape(-1)))
        self.assertTrue(np.allclose(min_S2, np.asarray(ml_min_S2).reshape(-1)))
        self.assertTrue(np.allclose(max_systole, np.asarray(ml_max_systole).reshape(-1)))
        self.assertTrue(np.allclose(min_systole, np.asarray(ml_min_systole).reshape(-1)))
        self.assertTrue(np.allclose(max_diastole, np.asarray(ml_max_diastole).reshape(-1)))
        self.assertTrue(np.allclose(min_diastole, np.asarray(ml_min_diastole).reshape(-1)))

        self.assertTrue(np.allclose(d_distributions, ml_d_distributions))

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

    def test_psd(self):
        from get_PSD_feature_Springer_HMM import get_PSD_feature_Springer_HMM

        python_result = get_PSD_feature_Springer_HMM(self.recording, 1000., 40., 60., use_matlab=True)
        matlab_result = self.eng.get_PSD_feature_Springer_HMM(self.ml_recording["r"], 1000., 40., 60.)

        print(python_result[:5])
        print(np.asarray(matlab_result).reshape(-1)[:5])

        self.assertTrue(np.allclose(python_result, np.asarray(matlab_result)))
