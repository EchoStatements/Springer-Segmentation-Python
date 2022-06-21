import numpy as np
import default_Springer_HSMM_options as springer_options

def get_duration_distributions(heartrate, systolic_time):
    mean_S1 = np.round(0.122 * springer_options.audio_segmentation_Fs)
    std_S1 = np.round(0.022 * springer_options.audio_segmentation_Fs)
    mean_S2 = np.round(0.094 * springer_options.audio_segmentation_Fs)
    std_S2 = np.round(0.022 * springer_options.audio_segmentation_Fs)



    mean_systole = np.round(systolic_time * springer_options.audio_segmentation_Fs) - mean_S1
    std_systole = (25. / 1000.) * springer_options.audio_segmentation_Fs

    mean_diastole = ((60. / heartrate) - systolic_time - 0.094) * springer_options.audio_segmentation_Fs
    std_diastole = 0.07 * mean_diastole + (6. / 1000.) * springer_options.audio_segmentation_Fs


    d_distributions = np.zeros((4, 2))

    d_distributions[0, 0] = mean_S1
    d_distributions[0, 1] = std_S1 ** 2

    d_distributions[1, 0] = mean_systole
    d_distributions[1, 1] = std_systole ** 2

    d_distributions[2, 0] = mean_S2
    d_distributions[2, 1] = std_S2 ** 2

    d_distributions[3, 0] = mean_diastole
    d_distributions[3, 1] = std_diastole ** 2

    min_systole = mean_systole - 3 * (std_systole + std_S1)
    max_systole = mean_systole + 3 * (std_systole + std_S1)

    min_diastole = mean_diastole - 3. * std_diastole
    max_diastole = mean_diastole + 3. * std_diastole

    min_S1 = (mean_S1 - 3 * std_S1)
    if min_S1 < springer_options.audio_segmentation_Fs / 50.:
        min_S1 = springer_options.audio_segmentation_Fs / 50.

    min_S2 = mean_S2 - 3 * std_S2
    if min_S2 < springer_options.audio_segmentation_Fs / 50.:
        min_S2 = springer_options.audio_segmentation_Fs / 50.

    max_S1 = mean_S1 + 3 * std_S1
    max_S2 = mean_S2 + 3 * std_S2

    return d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole