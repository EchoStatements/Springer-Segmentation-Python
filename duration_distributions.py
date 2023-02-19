import numpy as np

class DataDistribution(object):
    """
    Class for managing information about the distribution of lengths of heart phases inferred from data.

    Systolic and diastolic time interval statistics are inferred on a per-example basis from heart rate.
    S1 and S2 time interval stats are either inferred from the whole dataset or fixed to known reasonable
    values, depending on whether training data is provided to the constructor.
    """

    def __init__(self, data=None, features_frequency=50):
        self.feature_frequency = features_frequency
        if data is None:
            self.use_default_priors()
        else:
            self.get_priors_from_data(data)

    def get_distributions(self, heart_rate, systolic_time):
        self.systole_mean = systolic_time * self.feature_frequency - self.s1_mean
        self.diastole_mean = ((60. /heart_rate) - systolic_time - 0.094) * self.feature_frequency

        self.systole_std = (25. / 1000.) * self.feature_frequency
        self.diastole_std = 0.07 * self.diastole_mean + (6. / 1000.) *  self.feature_frequency

        self.systole_max = self.systole_mean + 5 * self.systole_std
        self.diastole_max = self.diastole_mean + 3 * self.diastole_std

        self.systole_min = self.systole_mean - 3 * self.systole_std
        self.diastole_min = self.diastole_mean - 3 * self.diastole_std

        d_distributions = np.zeros((4, 2))

        d_distributions[0, 0] = self.s1_mean
        d_distributions[0, 1] = self.s1_std ** 2

        d_distributions[1, 0] = self.systole_mean
        d_distributions[1, 1] = self.systole_std ** 2

        d_distributions[2, 0] = self.s2_mean
        d_distributions[2, 1] = self.s2_std ** 2

        d_distributions[3, 0] = self.diastole_mean
        d_distributions[3, 1] = self.diastole_std ** 2

        return d_distributions, self.s1_max, self.s1_min, self.s2_max, self.s2_min, self.systole_max, self.systole_min, self.diastole_max, self.diastole_min

    def get_priors_from_data(self, data):
        durations = [[], [], [], []]
        for segmentation in data:
            for row_idx in range(segmentation.shape[0]):
                if segmentation[row_idx, 2] in [1, 2, 3, 4]:
                    duration = segmentation[row_idx, 1] - segmentation[row_idx, 0]
                    durations[int(segmentation[row_idx, 2] - 1)].append(duration)
        s1 = np.array(durations[0])
        systole = np.array(durations[1])
        s2 = np.array(durations[2])
        diastole = np.array(durations[3])

        self.s1_mean = np.mean(s1) * self.feature_frequency
        self.systole_mean = np.mean(systole) * self. feature_frequency
        self.s2_mean = np.mean(s2) * self.feature_frequency
        self.diastole_mean = np.mean(diastole) * self.feature_frequency

        self.s1_std = np.std(s1) * self.feature_frequency
        self.systole_std = np.std(systole) * self.feature_frequency
        self.s2_std = np.std(s2) * self.feature_frequency
        self.diastole_std = np.std(diastole) * self.feature_frequency

        self.s1_min = max(self.s1_mean - 3 * self.s1_std, self.feature_frequency / 50.)
        self.s2_min = max(self.s2_mean - 3 * self.s2_std, self.feature_frequency / 50.)

        self.s1_max = self.s1_mean + 3 * self.s1_std
        self.s2_max = self.s2_mean + 30 * self.s2_std

    def use_default_priors(self):
        self.s1_mean = np.round(0.122 * self.feature_frequency)
        self.s1_std = np.round(0.022 * self.feature_frequency)
        self.s2_mean = np.round(0.094 * self.feature_frequency)
        self.s2_std = np.round(0.022 * self.feature_frequency)

        self.s1_min = (self.s1_mean - 3 * self.s1_std)
        if self.s1_min < self.feature_frequency / 50.:
            self.s1_min = self.feature_frequency / 50.

        self.s2_min = self.s2_mean - 3 * self.s2_std
        if self.s2_min < self.feature_frequency / 50.:
            self.s2_min = self.feature_frequency / 50.

        self.s1_max = self.s1_mean + 3 * self.s1_std
        self.s2_max = self.s2_mean + 3 * self.s2_std
