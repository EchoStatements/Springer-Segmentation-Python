import default_Springer_HSMM_options as springer_options
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal

from get_duration_distributions import get_duration_distributions

#
def ind2sub(array_shape, ind):
   ind[ind < 0] = -1
   ind[ind >= array_shape[0] * array_shape[1]] = -1
   rows = (ind.astype('int') / array_shape[1])
   cols = ind % array_shape[1]
   return (rows, cols)

def viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution,
                              heart_rate, systolic_time, Fs, models):

   T = observation_sequence.shape[0]
   N = 4
   max_duration_D = int(np.round( (60./ heart_rate) * Fs))

   delta = np.ones((T + max_duration_D, N)) * -np.inf

   psi = np.zeros((T + max_duration_D, N))

   psi_duration = np.zeros((T + max_duration_D - 1, N))

   observation_probs = np.zeros((T, N))

   for n in range(N):
      pihat = models[n].predict_proba(observation_sequence)

      for t in range(T):

         Po_correction = multivariate_normal.pdf(observation_sequence[t, :].reshape(1, -1), mean=total_obs_distribution[0], cov=total_obs_distribution[1])

         # NEED TO CHECK THAT THIS SHOULD INDEED BE PIHAT, NOT 1-PIHAT
         observation_probs[t, n] = (pihat[t, 1] * Po_correction) / pi_vector[n]

   d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole = get_duration_distributions(heart_rate, systolic_time)

   # line 170 in the matlab code suggests we might get some index errors in the code below, since matlab autoextends vectors
   duration_probs = np.zeros((N, 3 * Fs))
   duration_sum = np.zeros(N)

   for state_j in range(N):
      for d in range(max_duration_D):
         if state_j == 1:
            duration_probs[state_j, d] = multivariate_normal.pdf(d, mean=d_distributions[state_j, 0], cov=d_distributions[state_j, 1])

            if d < min_S1 or d > max_S1:
               duration_probs[state_j, d] = np.finfo(float).tiny



         elif state_j == 3:
            duration_probs[state_j, d] = multivariate_normal.pdf(d, mean=d_distributions[state_j, 0],
                                                                 cov=d_distributions[state_j, 1])

            if d < min_S2 or d > max_S2:
               duration_probs[state_j, d] = np.finfo(float).tiny

         elif state_j == 2:
            duration_probs[state_j, d] = multivariate_normal.pdf(d, mean=d_distributions[state_j, 0], cov=d_distributions[state_j, 1])

            if d < min_systole or d > max_systole:
               duration_probs[state_j, d] = np.finfo(float).tiny

         elif state_j == 4:
            duration_probs[state_j, d] = multivariate_normal.pdf(d, mean=d_distributions[state_j, 0],
                                                             cov=d_distributions[state_j, 1])

            if d < min_diastole or d > max_diastole:
               duration_probs[state_j, d] = np.finfo(float).tiny

      duration_sum[state_j] = np.sum(duration_probs[state_j, :])

   qt = np.zeros((1, delta.shape[0]))

   delta[0, :] = np.log(pi_vector) + np.log(observation_probs[0, :])

   psi[0, :] = -1

   a_matrix = np.array([ [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])

   # We aren't going to be using mex here, and we're not taking it out as an option either (yet)


   for t in range(1, T + max_duration_D - 1):
      for j in range(N):
         for d in range(max_duration_D):
            start_t = t - d
            if start_t < 1:
               start_t = 1
            if start_t > T-1:
               start_t = T-1

            end_t = t
            if t > T:
               end_t = T

            max_index = np.argmax(delta[start_t, :] + np.log(a_matrix[:, j]))
            max_delta = (delta[start_t, :] + np.log(a_matrix[:, j]))[max_index]

            probs = np.prod(observation_probs[start_t: end_t, j])

            if probs == 0:
               probs =  np.finfo(float).tiny
            emission_probs = np.log(probs)

            if emission_probs == 0 or np.isnan(emission_probs):
               emission_probs =  np.finfo(float).tiny

            delta_temp = max_delta + emission_probs + np.log(duration_probs[j, d] / duration_sum[j])

            if delta_temp > delta[t, j]:
               delta[t, j] = delta_temp
               psi[t, j] = max_index
               psi_duration[t, j] = d

   temp_delta = delta[T:, :]
   idx = np.argmax(temp_delta)
   pos, _ = np.unravel_index(idx, temp_delta.shape)

   pos = pos + T

   state = int(np.argmax(delta[pos, :]))

   offset = pos
   preceding_state = psi[offset, state]
   onset = int(offset - psi_duration[offset, state] + 1)

   qt[0, onset:offset] = state

   state = int(preceding_state)

   count = 0

   while onset > 2:

      offset = int(onset - 1)
      preceding_state = psi[offset, state]

      onset = int(offset - psi_duration[offset, state] + 1)

      if onset < 2:
         onset = 1

      qt[0, onset:offset] = state
      state = int(preceding_state)
      count = count + 1

      if count > 1000:
         break

      qt = qt[[0], :T]

   return delta, psi, qt