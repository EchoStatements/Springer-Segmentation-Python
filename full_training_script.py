import sys
from segmentation_model import SegmentationModel
from utils import get_wavs_and_tsvs, get_heart_rate_from_tsv, create_segmentation_array
import matplotlib.pyplot as plt
import numpy as np

def main(train_dir, test_dir, heartrates_from_tsv=False):
    # Get training recordings and segmentations
    train_recordings, \
        train_segmentations, \
        train_names = get_wavs_and_tsvs(input_folder=train_dir, return_names=True)

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
    model.fit(clips, annotations)

    # Load test set recordings
    test_recordings, \
        test_segmentations, \
        test_names = get_wavs_and_tsvs(input_folder=test_dir, return_names=True)

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
    for rec, seg in zip(clips, annotations):
        if heartrates_from_tsv:
            annotation = model.predict(rec, heart_rate=heartrates[idx])
        else:
            annotation = model.predict(rec)
        plt.plot(annotation)
        plt.plot(seg)
        plt.show()

        accuracies[idx] = (seg == annotation).mean()
        idx += 1
    print(f"average accuracy: {accuracies.mean()}")

if __name__ == "__main__":
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    main(train_dir, test_dir)
