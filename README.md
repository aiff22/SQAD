# SQAD
SQAD: Automatic Smartphone Camera Quality Assessment and Benchmarking<br>

Dataset download: [SQAD](). There are three folders contained: *train*, *test*, and *sample_crop16*. Follow the instructions below to prepare the data for training.<br>

TODO List:
- [] add codes for model training
- [] add pre-trained models
- [] add tables for baseline results

## Train the model
For ResNet50-based backbone, there is **no** specific requirement for the environment settings. Just make sure you have installed *pytorch* and *Pillow* properly. If there are some package related issues, follow the error instructions.<br>

For the evaluations with MANIQA, please follow the guidelines provided by [CVPRW 2022: MANIQA](https://github.com/IIGROUP/MANIQA) to set up the environment. Thanks for their great works on image quality assessment.

### Dataset Preparation
* Please download the [SQAD]() dataset, unzip and put all files into the `data` folder in the following structure:
```
SQAD
|—— data
|	|—— processed
|	|—— train
|	|	|—— 01_ASUS_Z00AD
|	|	|	|—— P_20000102_085109.jpg & ...
|	|	|—— 02_...
|	|—— test
|	|	|—— 01_ASUS_Z00AD
|	|	|—— 02_...
|	|—— sample_crop16
|—— prepare_data.py
|—— camera_ground_truth.json
```
To get the image crops for training, simply run:
```
python prepare_data.py
```
Then, the cropped images and corresponding label files for training are stored in `./data/processed`.<br>

For model testing, we provide a pre-cropped version in the folder `sample_crop16`, which is extracted from the `test` folder.

* To utilize the entire test set, one can also apply the random cropping during *DataLoader* and specify different image crop sizes and desired crop numbers. We provide an *AllDataset* class in `CameraDataset.py` and an example test script that we used for reporting numbers in our *ablations* section.<br>

Note that the running speed for this test would be much slower than the pre-cropped setting, since the dataloader needs to load the full-size image first, which is slightly time-consuming for some high quality photos captured by high-end mobile phones.


## Baselines
Here we provide some baselines for testing on `sample_crop16` subset and the entire `test` set. Within one image, the information levels (i.e. edge details, color vividness) for different parts usually vary, there would be fluctuations for the metrics. In the paper we reported the average for 5 trials, and here are the detailed numbers for each trial. 

## Pre-traind models
