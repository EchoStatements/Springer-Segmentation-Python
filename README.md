# Hidden Markov Model based segmentation for Phonocardiograms

Python implementation of https://github.com/davidspringer/Springer-Segmentation-Code 

## Minimal Example

The following is an example of how to run the code. The example takes on a collection of recordings (`.wav` files) and their corresponding
segmentations (`.tsv` files), and trains the HMM on that data, then runs inference on the first `.wav` in the list.  

```python
from springer_segmentation.utils import get_wavs_and_tsvs
from springer_segmentation.train_segmentation import train_hmm_segmentation
from springer_segmentation.run_segmentation import run_hmm_segmentation

# get list of recordings and corresponding segmentations (in the format given in the tsv)
wavs, tsvs = get_wavs_and_tsvs("tiny_test")

# train the model
models, pi_vector, total_obs_distribution = train_hmm_segmentation(wavs, tsvs)

# get segmentations out of the model for the first wav file in our list
annotation, heart_rate = run_hmm_segmentation(wavs[0],
                                      models,
                                      pi_vector,
                                      total_obs_distribution,
                                      min_heart_rate=60,
                                      max_heart_rate= 200,
                                      return_heart_rate=True)
```

If using this code, please cite:

Summerton, S., Wood, D., Murphy, D., Redfern, O., Benatan, M., Kaisti, M., & Wong, D. C. (2022). Two-stage Classification for Detecting Murmurs from Phonocardiograms Using Deep and Expert Features. In 2022 Computing in Cardiology (CinC), volume 49. IEEE, 2023; 1–4.
