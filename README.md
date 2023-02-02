## LEAVES

This directory contains the example codes for the LEAVES that submitted to KDD 2023.

The major contribution componenet is the LEAVES module with the differentiable augmentation methods, which programed in ./models/augo_aug.py and ./utils/differentiable_augs.py, respectively.

The datasets used in this study:
- Apnea-ECG: https://physionet.org/content/apnea-ecg/1.0.0/
- Sleep-EDFE: https://www.physionet.org/content/sleep-edfx/1.0.0/
- PAMAP2: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.2/
The processing codes for PTB-XL and Sleep-EDFE is in the ./utils directory


The python environment used in this work can be found in environment.yml

Command of 'run main.py' can be used to run the projects. The settings, such as path of the dataset, hyper-parameters, can be found in configs.py file.
