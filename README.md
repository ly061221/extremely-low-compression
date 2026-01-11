# ELC: Extremely Low Bitrate Image Compression with Decoder-Side Enhancement for Human and Machine 
This repository contains the code to perform and evaluate the image compression method in the following paper: 
>Liya Sha, Shipei Wang, Kunqiang Huang, Chao Yang, Xinpeng Huang and Ping An. Extremely Low Bitrate Image Compression with Decoder-Side Enhancement for Human and Machine.


## Dependencies

- Python=3.7.10
- numpy=1.17.0
- torch=1.8.0
- torchvision=0.9.0
- tqdm=4.65.2

## Dataset
The following datasets are used and needed to be downloaded.
- Oxford102 (download [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).)
- ImageNette2-320 (download [here](https://github.com/fastai/imagenette).)

## Train
### Classification task
Train a base codec model focused on reconstruction at a normal bitrate, and the parameters and path settings are specified in the `./config/base_codec.yaml` file:
```javascript
    python train_base.py
```
>Learned weights will be stored at `train/model/{base_reconstruction}`.

Then load the base codec weight, and run `train_classification_RB.py` training for the classification task, and the settings are specified in the `./config/classification_RB.yaml` file:
```javascript
    python train_classification_RB.py
```
>Learned weights will be stored at `train/model/{rec_cls_RB}`.

### Image reconstruction
Load the model pretrained on the classification task. In the first stage, run `train_classification_decoder.py` training the decoder for reconstruction, and the settings are specified in the `./config/decoder.yaml` file:
```javascript
    python train_classification_decoder.py
```
>Learned weights will be stored at `train/model/{enhanced_decoder}`.

In the second stage,  run `train_classification_hifiD.py` for adversarial training, and the settings are specified in the `./config/classification_hifiD.yaml` file:
```javascript
    python train_classification_hifiD.py
```
>Learned weights will be stored at `train/model/{decoder_hifiD}`.

## Evaluation
Modify the pretrained weight path in the `./config/eval.yaml` file, and then run `eval.py`.
```javascript
    python eval.py
```

## Pretrained Model

Download the pretrained models from the different training stages below, and place them in the `./weights` directory. If you only want to evaluate the performance of the final model, you only need to download the **Final-model**.

|              Stages          |   Pretrained Models    | 
|:----------------------------:|------------------|
|     ResNet50           | [Oxford102-cls](https://github.com/ly061221/extremely-low-compression/releases/download/v1.0.0/res50_oxford_cls.pth) |
|     Classification           | [Base-codec](https://github.com/ly061221/extremely-low-compression/releases/download/v1.0.0/based_pretrained.pth) |
|                              | [cls-codec-FETM](https://github.com/ly061221/extremely-low-compression/releases/download/v1.0.0/cls_RB_0.09bpp.pth) |
|    Reconstruction (final)   | [Final-model](https://github.com/ly061221/extremely-low-compression/releases/download/v1.0.0/hifiD_final_0.09bpp.pth) | 

