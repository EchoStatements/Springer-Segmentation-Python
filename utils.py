import os
import random
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm


def create_train_test_split(directory="tiny_test",
                            frac_train=0.8,
                            max_train_size=None,
                            max_test_size=None):

    # create list of patient IDs
    patient_ids = set()
    for file_ in tqdm(sorted(os.listdir(directory))):
        patient_ids.add(file_[:5])
    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)
    max_train_idx = int(np.round(frac_train * len(patient_ids)))
    train_ids = patient_ids[:max_train_idx]
    test_ids = patient_ids[max_train_idx:]

    train_wavs = []
    train_segs = []
    test_wavs = []
    test_segs = []

    for file_ in tqdm(sorted(os.listdir(directory))):
        root, extension = os.path.splitext(file_)
        skip = False
        skip = True if file_[:5] in train_ids and len(train_wavs) == max_train_size else skip
        skip = True if file_[:5] in test_ids and len(test_wavs) == max_test_size else skip
        if extension == ".wav" and not skip:
            segmentation_file = os.path.join(directory, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            _, recording = scipy.io.wavfile.read(os.path.join(directory, file_))

            tsv_segmentation = np.loadtxt(segmentation_file, delimiter="\t")
            if file_[:5] in train_ids:
                train_wavs.append(recording)
                train_segs.append(tsv_segmentation)
            else:
                test_wavs.append(recording)
                test_segs.append(tsv_segmentation)

    return train_wavs, train_segs, test_wavs, test_segs


def get_wavs_and_tsvs(input_folder="tiny_test", return_names=False):
    """

    Parameters
    ----------
    input_folder
    return_names : boolean

    Returns
    -------

    """
    wav_arrays = []
    tsv_arrays = []
    if return_names:
        names = []

    for file_ in tqdm(sorted(os.listdir(input_folder))):
        root, extension = os.path.splitext(file_)
        if extension == ".wav":
            segmentation_file = os.path.join(input_folder, root + ".tsv")
            if not os.path.exists(segmentation_file):
                continue
            _, recording = scipy.io.wavfile.read(os.path.join(input_folder, file_))
            wav_arrays.append(recording)
            if return_names:
                names.append(file_)

            tsv_segmentation = np.loadtxt(segmentation_file, delimiter="\t")
            tsv_arrays.append(tsv_segmentation)
    if return_names:
        return wav_arrays, tsv_arrays, names
    return wav_arrays, tsv_arrays

def get_training_data(directory):
    recordings, segmentations = get_wavs_and_tsvs(input_folder=directory)
    clips = []
    annotations = []
    for rec, seg in zip(recordings, segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    return clips, annotations


def get_heart_rate_from_tsv(tsv_array):
    systole_rows = tsv_array[tsv_array[:, 2] == 2., :]
    differences = np.diff(systole_rows[:, 0])
    heart_rate = 60. / np.median(differences)
    return heart_rate


def create_segmentation_array(recording,
                              tsv_segmentation,
                              recording_frequency,
                              feature_frequency=50):
    """
    Creates two lists: the first is a list of clips from the recording that we can construct segmentations for from
    the data in the csv file; the second is the segmentations themselves. The two lists will be the same length, and
    the i-th entries in both lists will be ndarrays of the same shape, with the entry in the second list being the
    segmentation of the corresponding clip in the first list.


    Parameters
    ----------
    recording
    tsv_segmentation
    recording_frequency : int
        Frequency at which the recording is sampled
    feature_frequency : int
        Frequency of the features extracted in order to train the segmentation. The default, 50, is
        the frequency used in the matlab implementation

    Returns
    -------

    clipped_recordings : list of ndarrays
    segmentations : list of ndarrays

    """

    full_segmentation_array = np.zeros(int(recording.shape[0] * feature_frequency / recording_frequency))

    for row_idx in range(0, tsv_segmentation.shape[0]):
        row = tsv_segmentation[row_idx, :]
        start_point = np.round(row[0] * feature_frequency).astype(int)
        end_point = np.round(row[1] * feature_frequency).astype(int)
        full_segmentation_array[start_point:end_point] = int(row[2])

    start_indices = []
    end_indices = []
    segmentations = []
    segment_started = False
    TOLERANCE = 5
    for idx in range(full_segmentation_array.shape[0]):
        if not segment_started and full_segmentation_array[idx] == 0:
            continue
        if full_segmentation_array[idx] != 0:
            if not segment_started:
                start_indices.append(idx)
                segment_started = True
                tol_counter = 0
            if tol_counter > 0:
                for adjust_idx in range(tol_counter):
                    full_segmentation_array[idx - adjust_idx - 1] = full_segmentation_array[idx - tol_counter - 1]
                tol_counter = 0
        if segment_started and full_segmentation_array[idx] == 0:
            tol_counter += 1
        if tol_counter == TOLERANCE or idx == full_segmentation_array.shape[0] - 1:
            end_indices.append(idx - tol_counter)
            if end_indices[-1] - start_indices[-1] > feature_frequency:
                segmentations.append(full_segmentation_array[start_indices[-1]:end_indices[-1]].astype(int))
            else:
                end_indices = end_indices[:-1]
                start_indices = start_indices[:-1]
            segment_started = False

    clipped_recordings = []
    for start, end in zip(start_indices, end_indices):
        clip = recording[int(start * recording_frequency / feature_frequency):int(end * recording_frequency / feature_frequency)]
        clipped_recordings.append(clip)

    # segmentation_array = segmentation_array[seg_start:seg_end].astype(int)
    return clipped_recordings, segmentations

def upsample_states(original_qt, old_fs, new_fs, new_length):
    """

    Parameters
    ----------
    original_qt
       The states inferred from the recording features (sampled at old_fs)
    old_fs
        The sampling frequency of the features from which the states were derived
    new_fs
        The desired sampling frequency
    new_length
        The desired length of the new signal

    Returns
    -------
    expanded_qt
        the inferred states resampled to be at frequency new_fs

    """

    original_qt = original_qt.reshape(-1)
    expanded_qt = np.zeros(new_length)

    indices_of_changes =  np.nonzero(np.diff(original_qt))[0]
    indices_of_changes = np.concatenate((indices_of_changes, [original_qt.shape[0] - 1]))

    start_index = 0
    for idx in range(len(indices_of_changes)):
        end_index = indices_of_changes[idx]

        # because division by 2 only has 0.0 and 0.5 as fractional parts, we can use ceil instead of round to stay faithful to MATLAB
        #mid_point = int(np.ceil((end_index - start_index) / 2) + start_index)
        # We don't need value at midpoint, we can just use the value at the start_index

        # value_at_midpoint = original_qt[mid_point]
        value_at_midpoint = original_qt[end_index]
#        if start_index != 0:
#            assert original_qt[start_index + 1] == original_qt[end_index]

        expanded_start_index = int(np.round((start_index) / old_fs * new_fs)) + 1
        expanded_end_index = int(np.round((end_index) / old_fs * new_fs))

        if expanded_end_index > new_length:
            expanded_end_index = new_length
        if idx == len(indices_of_changes) - 1:
            expanded_end_index = new_length + 1

        expanded_qt[expanded_start_index - 1:expanded_end_index] = value_at_midpoint
        start_index = end_index


    # expanded_qt = expanded_qt.reshape(-1, 1)
    return expanded_qt
