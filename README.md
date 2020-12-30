# Medical Image Segmentation (fluorescence microscopy)
This is the official repository for the project done by 12 group in the course DD2430 (Project Course in Data Science) at KTH. This project was done Autum 2020.

Datasets used in this study are the following:
1. BBBC039 https://bbbc.broadinstitute.org/BBBC039
2. BBBC004 https://bbbc.broadinstitute.org/BBBC004

To run the experiments the dataset should be download and added to a folder named datasets. A GPU which is compatible with pytorch and tensorflow is also needed.

## DoGNET

The experiments using DoGNET can be run by firstly navigating to the folder called dognet then calling either `python bbbc004.py` or `python bbbc039.py`

## UNET

The experiments using UNET can be run by firstly navigating to the folder called unet then calling either `python unet_experiment_004.py`, `python unet_experiment_039.py` or `python unet_experiment_scilife.py`

## FCM

