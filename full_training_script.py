import sys

from duration_distributions import DataDistribution
from segmentation_model import SegmentationModel
from utils import get_wavs_and_tsvs, get_heart_rate_from_tsv, create_segmentation_array, create_train_test_split
import matplotlib.pyplot as plt
import numpy as np

def main(data_dir, heartrates_from_tsv=False):
    # Get training recordings and segmentations
    train_recordings, train_segmentations, \
        test_recordings, test_segmentations = create_train_test_split(directory=data_dir,
                                                                      frac_train = 0.74,
                                                                      max_train_size=500,
                                                                      max_test_size=100)

    # Preprocess into clips with annotations of the same length
    clips = []
    annotations = []
    for rec, seg in zip(train_recordings, train_segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    # Train the model
    model = SegmentationModel()
    data_distribution = DataDistribution()
    model.fit(clips, annotations, data_distribution=data_distribution)

    # Process test set in to clips and annotations
    clips = []
    annotations = []
    heartrates = []
    for rec, seg in zip(test_recordings, test_segmentations):
        clipped_recording, ground_truth = create_segmentation_array(rec,
                                                                    seg,
                                                                    recording_frequency=4000,
                                                                    feature_frequency=50)
        if heartrates_from_tsv:
            heartrate = get_heart_rate_from_tsv(seg)
            for _ in range(len(clipped_recording)):
                heartrates.append(heartrate)
        clips.extend(clipped_recording)
        annotations.extend(ground_truth)

    # Evaluate performance on clips
    idx = 0
    accuracies = np.zeros(len(clips))
    weights = np.zeros(len(clips))
    for rec, seg in zip(clips, annotations):
        if heartrates_from_tsv:
            annotation = model.predict(rec, heart_rate=heartrates[idx])
        else:
            annotation = model.predict(rec)
        plt.plot(annotation, label="predicted")
        plt.plot(seg, label="ground_truth")
        plt.legend()
        plt.show()

        accuracies[idx] = (seg == annotation).mean()
        weights[idx] = seg.shape[0]
        idx += 1
    print(f"average accuracy: {accuracies.mean()}")
    print(f"average weight-corrected accuracy: {np.average(accuracies, weights=weights)}")

if __name__ == "__main__":
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    main(train_dir, heartrates_from_tsv=False)
