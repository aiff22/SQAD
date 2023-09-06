# SQAD
SQAD: Automatic Smartphone Camera Quality Assessment and Benchmarking<br>

TODO List:
- [] add codes for model training
- [] add pre-trained models
- [] add tables for baseline results

## Train the model
For ResNet50-based backbone, there is **no** specific requirement for the environment settings. Just make sure you have installed *pytorch* and *Pillow* properly. If there are some package related issues, follow the error instructions.<br>

For the evaluations with MANIQA, please follow the guidelines provided by [CVPRW 2022: MANIQA](https://github.com/IIGROUP/MANIQA) to set up the environment. Thanks for their great works on image quality assessment.

### Dataset Preparation
* Please down load the [SQAD]() dataset (contains: train, test, sample_crop16), unzip and put them into the `data` folder in the following structure:
```
SQAD
|—— data
|	|—— train
|	|	|—— 01_ASUS_Z00AD
|	|	|	|—— P_20000102_085109.jpg & ...
|	|—— test
|	|	|—— 01_ASUS_Z00AD
|	|—— sample_crop16
|—— prepare_data.py
|—— camera_ground_truth.json
```
To get the image crops for training, simply run:
```
python prepare_data.py
```

Then, the cropped images and the label files are stored in `./data/processed`

## Baselines

## Pre-traind models
