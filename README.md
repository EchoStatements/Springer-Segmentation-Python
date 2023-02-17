# Hidden Markov Model based segmentation for Phonocardiograms

Python implementation of https://github.com/davidspringer/Springer-Segmentation-Code 

## Using This Code

An example of how to use this code can be found in `full_training_script.py`. The script takes two arguments, the directory for the training data and the directory for the test data. The data in both directories is expected to be `.wav` files and corresponding `.tsv` files in the format specified in the [2022 Physionet Challenge](https://moody-challenge.physionet.org/2022/#Oliveira2022)


This code was developed for and used in the following paper, which should be cited if you use the code in your own work.

Summerton, S., Wood, D., Murphy, D., Redfern, O., Benatan, M., Kaisti, M., & Wong, D. C. (2022). Two-stage Classification for Detecting Murmurs from Phonocardiograms Using Deep and Expert Features. In 2022 Computing in Cardiology (CinC), volume 49. IEEE, 2023; 1–4.

Additionally, you should cite the original paper upon which our code is based. 

D. Springer et al., "Logistic Regression-HSMM-based Heart Sound Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.