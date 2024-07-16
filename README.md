# FootNet v1.0
This codebase accompanies our paper about the development of a deep learning-based emulator of measurement footprints.
The repository contains the following:
* The ``footnet.py`` file contains the function to build the deep learning model, the data loader, and the transformations applied on the input and output fields.
* The ``training_script.ipynb`` demonstrates the setup of the training loop.
* The ``figures`` folder contains scripts used to generate figures shown in the paper.
* Examples of the footprints used in the training process could be downloaded from https://hermes.atmos.washington.edu/footnet_training_samples/footprints_data.tar.gz. The complete training data set could be provided upon request.
* The footprints are provided in Numpy compressed array format, which could be decompressed with Python 3.10.6 and NumPy 1.23.4.
