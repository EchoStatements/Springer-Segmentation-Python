import default_Springer_HSMM_options as springer_options
import numpy as np
from sklearn.linear_model import LogisticRegression

def viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution,
                              heart_rate, systolic_time, Fs):

   T = observation_sequence.shape[0]
   N = 4
   max_duration_D = np.round( (60./ heart_rate) * Fs)

   delta = np.ones((T + max_duration_D, N)) * -np.inf

   psi = np.zeros((T + max_duration_D, N))

   psi_duration = np.zeros((T + max_duration_D - 1, N))

   observation_probs = np.zeros(T, N)

   for idx in range(N):
      pass