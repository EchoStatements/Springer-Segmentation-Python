import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from duration_distributions import DataDistribution
from extract_features import get_all_features, get_default_features
from heart_rate import get_heart_rate
from viterbi import viterbi_segment


class SegmentationModel(object):
    """
    Hidden semi-Markov model for segmenting phonocardiograms.
    """

    def __init__(self,
                 feature_extractor=None,
                 sampling_frequency=4000,
                 feature_frequency=50,
                 feature_prob_model="logistic"):
        """

        Parameters
        ----------
        feature_extractor : function or None
            Function to be used for extracting features from audio for use by the model.
        sampling_frequency : int (default=4000)
            Sampling frequency of the original recording
        feature_frequency : int (default=50)
            Sampling frequency of features derived from feature extractor.
        feature_prob_model : str or callable (default="logistic")
            The model to be used to predict the probability of states given observations. "logistic" gives
            a logistic regression model, "rf" gives a random forest. Can also except a class which implements
            `fit` and `predict_proba` and has a constructor that doesn't require any positional arguments.
        """
        self.sampling_frequency = sampling_frequency
        self.feature_frequency = feature_frequency
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = get_default_features
        self.model = feature_prob_model

    def fit(self, recordings, segmentations, data_distribution=None):
        """

        Parameters
        ----------
        recordings : list of ndarrys
            List whose entries are ndarrays of PCG recordings.
        segmentations : list of ndarrays
            List whose entries are of the same length as the corresponding entries of `recordings`,
            and reflects the phase of the heart cycle at that point in the recording
            (1: S1, 2: systole, 3: S2, 4: diastole).

        Returns
        -------
        self

        """
        number_of_states = 4
        state_observation_values = []
        if data_distribution is None:
            self.data_distribution = DataDistribution
        else:
            self.data_distribution = data_distribution

        # Collect per-state observation values
        for recording, segmentation in zip(recordings, segmentations):
            PCG_Features = self.feature_extractor(recording,
                                            self.sampling_frequency,
                                            )

            these_state_observations = []
            for state_i in range(1, number_of_states + 1):
                if PCG_Features.shape[0] != segmentation.shape[0]:
                    min_length = min(PCG_Features.shape[0], segmentation.shape[0])
                    PCG_Features = PCG_Features[:min_length]
                    segmentation = segmentation[:min_length]
                these_state_observations.append(PCG_Features[segmentation == state_i, :])
            state_observation_values.append(these_state_observations)

        self._fit(state_observation_values)
        return self

    def predict(self,
                recording,
                heart_rate=None,
                min_heart_rate=60,
                max_heart_rate=200):
        """
        Predict state sequence for a single recording. Heart rate is used to predict phase durations in the
        hiddem Markov model. If not heart rate is given, a heart rate is inferred from the recording.

        Parameters
        ----------
        recording :
        heart_rate : float, func or None (default=None)
            If a float is given, this is taken to be the heart rate to be assumed in the prediction. If a
            function is given, that function (taking the recording as an argument) is used to determine
            the heart rate. If None, the default function `get_heart_rate` is used.
        min_heart_rate : int (default=60)
            Minimum possible heart rate
        max_heart_rate : int (default=200)
            Maximum possible heart rate

        Returns
        -------
        state_sequence : ndarray

        """

        heart_rates, systolic_time_intervals = get_heart_rate(recording,
                                                             self.sampling_frequency,
                                                             min_heart_rate=min_heart_rate,
                                                             max_heart_rate=max_heart_rate)
        if heart_rate is None:
            heart_rate = heart_rates[0]

        systolic_time_intervals = systolic_time_intervals[0]

        features = self.feature_extractor(recording,
                                    self.sampling_frequency)
        likelihood, _, state_sequence = viterbi_segment(features,
                                                        self.models,
                                                        self.total_obs_distribution,
                                                        distribution=self.data_distribution,
                                                        heart_rate=heart_rate,
                                                        systolic_time=systolic_time_intervals,
                                                        recording_frequency=self.feature_frequency)

        return state_sequence

    def batch_predict(self, recordings,
                      min_heart_rates=None,
                      max_heart_rates=None):
        """
        Predict state sequences for a list of recordings.

        Parameters
        ----------
        recordings : list of ndarrays
        min_heart_rates : int or None (default=None)
            Minimum possible heart rate (if None, 60 is used)
        max_heart_rates :
            Maximum possible heart rate (if None, 200 is used)

        Returns
        -------
        state_sequences : list of ndarrays
            The predicted state sequences for the given recordings

        """
        state_sequences = []
        for idx, recording in enumerate(recordings):
            if min_heart_rates is not None:
                min_hr = min_heart_rates[idx]
            else:
                min_hr = 60
            if max_heart_rates is not None:
                max_hr = max_heart_rates[idx]
            else:
                max_hr = 200
            state_sequences.append(self.predict(recording, min_heart_rate=min_hr, max_heart_rate=max_hr))
        return state_sequences


    def _fit(self, state_observation_values):
        """
        Helper function called by `fit` to learn model and observation distributions from state
        observation values.

        Parameters
        ----------
        state_observation_values

        Returns
        -------

        """

        number_of_states = 4
        num_features = state_observation_values[0][0].shape[1]

        models = []
        statei_values = [np.zeros((0, num_features)) for _ in range(number_of_states)]

        for PCGi in range(len(state_observation_values)):
            for idx in range(4):
                statei_values[idx] = np.concatenate((statei_values[idx], state_observation_values[PCGi][idx]), axis=0)

        total_observation_sequence = np.concatenate(statei_values, axis=0)
        total_obs_distribution = []
        total_obs_distribution.append(np.mean(total_observation_sequence, axis=0))
        total_obs_distribution.append(np.cov(total_observation_sequence.T))

        for state_idx in range(number_of_states):

            length_of_state_samples = statei_values[state_idx].shape[0]

            length_per_other_state = np.floor(length_of_state_samples / (number_of_states - 1))

            min_length_other_class = np.inf

            for other_state_idx in range(number_of_states):
                samples_in_other_state = statei_values[other_state_idx].shape[0]

                if not other_state_idx != state_idx:
                    min_length_other_class = min(min_length_other_class, samples_in_other_state)

            if length_per_other_state > min_length_other_class:
                length_per_other_state = min_length_other_class

            training_data = [None, np.zeros((0, num_features))]

            for other_state_idx in range(number_of_states):
                samples_in_other_state = statei_values[other_state_idx].shape[0]

                if other_state_idx == state_idx:
                    indices = np.random.permutation(samples_in_other_state)[:int(length_per_other_state * (number_of_states -1)) ]
                    training_data[0] = statei_values[other_state_idx][indices, :]
                else:
                    indices = np.random.permutation(samples_in_other_state)[:int(length_per_other_state) + 1]
                    state_data = statei_values[other_state_idx][indices, :]
                    training_data[1] = np.concatenate((training_data[1], state_data), axis=0)

            labels = 2 * np.ones(training_data[0].shape[0] + training_data[1].shape[0])
            labels[0:training_data[0].shape[0]] = 1

            all_data = np.concatenate(training_data, axis=0)

            if self.model == "logistic":
                regressor = LogisticRegression()
            elif self.model == "rf":
                regressor = RandomForestClassifier(max_depth=10)
            else:
                regressor = self.model()
            regressor.fit(all_data, labels)
            models.append(regressor)

        self.models = models
        self.total_obs_distribution = total_obs_distribution
        return models, total_obs_distribution
