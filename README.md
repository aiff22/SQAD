# SQAD: Automatic Smartphone Camera Quality Assessment and Benchmarking
Official webpage for ICCV'23 paper "SQAD: Automatic Smartphone Camera Quality Assessment and Benchmarking"<br>

Dataset download: [SQAD](). There are three folders contained: *train*, *test*, and *sample_crop16*. Follow the instructions below to prepare the data for training.<br>

You can download our pre-trained models from this link: pre-trained [model zoo](). Here are MANIQA-based models with all 6 quality factors, and ResNet-based models for resolution on cross-camera prediction task. Please refer to Table 4 in the paper for camera set division.

TODO List:
- [ ] add tables for baseline results

## Model training

For the quality factor regression with MANIQA model, please follow the guidelines provided by [CVPRW 2022: MANIQA](https://github.com/IIGROUP/MANIQA) to set up the environment. Thanks for their great works on image quality assessment.<br>

For ResNet50-based backbone, there is **no** specific requirement for the environment settings. Just make sure you have installed *pytorch* and *Pillow* properly. If there are some package related issues, follow the error instructions.

### Dataset Preparation
* Please download the SQAD dataset, unzip and put all files into the `data` folder in the following structure:
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
|—— maniqa_plugin
|—— resnet_
```
To get the image crops for training, simply run:
```
python prepare_data.py
```
Or use the following command to specify the crop number and crop size (based on backbone network) per-image. You can also add `--crop_test_dataset` flag to create a randomly cropped test set for model evaluations.
```
python prepare_data.py --crop_num 32 --crop_size 240 --crop_test_dataset
```
Then, the cropped images and corresponding label files for training are stored in `./data/processed`.<br>

For model testing, we have provided a pre-cropped image sets in the folder `sample_crop16`, which is extracted from the `test` folder. But it is free to use `--crop_test_dataset` flag to create your own test test as well.

### Train with MANIQA
Please clone the MANIQA project and **replace** the original `train_maniqa.py` and `inference.py` by the files with the same name in `maniqa_plugin` folder.
```
git clone https://github.com/IIGROUP/MANIQA.git
```
Then, add `camera29.py` in `maniqa_plugin/data` into the original MANIQA `data` folder. Now you can modify the configurations in `train_maniqa.py` Line136-141 and `inference.py` Line75-78 to play with the model training and quality factor inference.

* The "factor" element in *config* uses the same key name with `camera_ground_truth.json`, they are: resolution, color_acc, noise, dr, psf, aliasing.

* Download the pre-trained models and put them into the save path you specified in `inference.py` for direct evaluation.


### Train with ResNet backbone

To train the regression model with ResNet50 backbone, run `train_regress.py` file in the `resnet_` folder with different quality factors:
```
python train_regress.py --factor dr(resolution/color_acc/...) --batch_size 64 --train_epochs 200 --save_eval_every 1
```

* To utilize the entire test set, one can also apply the random cropping during *DataLoader* and specify different image crop sizes and desired crop numbers. We provide a *FullImageDataset* class in `CameraDataset.py` for exploring. Note that the running speed for this test would be much slower than the pre-cropped setting, since the dataloader needs to load the full-size image first, which is slightly time-consuming for some high quality photos captured by high-end mobile phones.


## Baselines
Here we provide some baselines for testing on `sample_crop16` subset and some randomly generated test set from `test` folder. Within one image, the information levels (i.e. edge details, color vividness) for different parts usually vary, there would be fluctuations for the evaluation metrics if you apply your own cropping.

Results evaluated with `sample_crop16` subset:

 Metrics      | Resolution | Color Accuracy |   Noise  |   DR   |   PSF   | Aliasing
------------- |:----------:|:--------------:|:--------:|:------:|:-------:|:---------:
 *SROCC*      |   0.9292   |     0.9475     |  0.9336  | 0.9546 |  0.9424 |  0.9172
 *PLCC*       |   0.9460   |     0.9592     |  0.9784  | 0.9694 |  0.9823 |  0.9812

 Results evaluated with randomly cropped subset from `test` folder:

 Metrics      | Resolution | Color Accuracy |   Noise  |   DR   |   PSF   | Aliasing
------------- |:----------:|:--------------:|:--------:|:------:|:-------:|:---------:
 *SROCC*      |   0.9475   |     0.9242     |  0.8756  | 0.9328 |  0.9117 |  0.8940
 *PLCC*       |   0.9479   |     0.9414     |  0.9373  | 0.9332 |  0.9704 |  0.9688

* For cross-camera validation, please refer to *Ablations* section in our paper for SROCC and PLCC quantities.
