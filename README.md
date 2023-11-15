# AudioCrossVerification
This repository implements a method to detect tampered audio by cross referencing a query audio with a verified reference audio.

1. DAPS-DataPrep.ipynb first prepares the data (https://ccrma.stanford.edu/~gautham/Site/daps.html) by sampling random 10 second chunks of a reference recording and tampering them with insertion, deletion and replacement.

2. Get MFCC.ipynb then computes the MFCC feature representations of the queries and the references.

3. MFCC_baseline.ipynb provides the naive solution to the audio cross verification problem by comparing the similarity of the reference and the query MFCC features using Euclidean distance.

4. HSTW-MLv0.2-cython.ipynb is one of our solutions to the audio cross verification problem using hidden state time warping (https://github.com/HMC-MIR/HSTW) to align the query to the reference. Using the alignment path, we are able to compute a modified z-score to represent the audio tampering.
5. IQR.ipynb provides an alternative solution, using a hashprint feature representation along with a outlier detection system to classify tampering.


## Citation

Heidi Lei, Arm Wonghirundacha, Irmak Bukey, and TJ Tsai.  Audio Cross Verification Using Dual Alignment Likelihood Ratio Test.  Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023, pp. 1-5.

The paper can be found [here](https://drive.google.com/file/d/1mHuDolme7CnTotRuQCy0o5R2NIymdSU5/view).

### Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No. 1948531.  Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
