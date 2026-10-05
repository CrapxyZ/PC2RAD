<div align="center">

# Multi-class Unsupervised Anomaly Detection via Progressive Capacity-Constrained Feature Reconstruction
</div>
> - 🚧 This README is a work in progress. Additional figures and documentation will be added soon.


**This repository is built upon [Dinomaly](https://github.com/guojiajeremy/Dinomaly). We express our sincere gratitude to the authors for their inspiring and foundational contributions**

## Abstract

Multi-class unsupervised anomaly detection aims to detect and localize defects across categories with a single model trained on normal images. Pretrained feature reconstruction provides a natural solution, but unrestricted feature transfer can also reproduce anomalous details. We propose PC<sup>2</sup>RAD, a progressive capacity-constrained reconstruction framework that separates the choice of reconstruction content from the control of its transmission. Each target stage draws content from its immediate predecessor, and the reconstructed features are propagated to the next decoder stage. Grid Consensus Pooling compresses dense features into spatially indexed regional prototypes. Dual Guidance Attention allocates predecessor prototypes using source- and target-stage relations, while a narrow channel transformation further restricts the content path. Spatial-Semantic Joint Corruption complements these restrictions during training with clean features as reconstruction targets. Experiments on MVTec AD, VisA, and BTAD show competitive image-level detection and pixel-level localization. On MVTec AD, PC<sup>2</sup>RAD achieves 99.6% image-level AUROC, 98.7% pixel-level AUROC, and 96.4% AUPRO. With the same reported backbone and input configuration as Dinomaly, it reduces parameters by 25.4% and MACs by 23.9%. Ablations support the complementary effects of token and channel restrictions and show that the benefit of reconstruction direction depends on the organization of source content.

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.9.13
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090 (24GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory. It is where we keep all the datasets by default.
You can also alter it according to your need, just remember to modify the `data_path` in the code. 

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```


### VisA

Download the VisA dataset from [URL](https://github.com/amazon-science/spot-diff).
Unzip the file to `../VisA/`. Preprocess the dataset to `../VisA_pytorch/` in 1-class mode by their official splitting 
[code](https://github.com/amazon-science/spot-diff).

You can also run the following command for preprocess, which is the same to their official code.

```
python ./prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file ./prepare_data/split_csv/1cls.csv
```
`../VisA_pytorch` will be like:
```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```
 
### BTAD
The BTAD dataset can be downloaded from the official source [URL](https://avires.dimi.uniud.it/papers/btad/btad.zip).
Unzip the file to `../BTech_Dataset_transformed`.
```
|-- BTech_Dataset_transformed
    |-- 01
    |-- 02
    |-- 03
```

## 3. Run Experiments
### Training and evaluation
Set your own configuration in [PC2RAD_train_and_eval.py](PC2RAD_train_and_eval.py) and then
```
python PC2RAD_train_and_eval.py
```

Results will be in `./output_models`, including log and weights of the last iteration.

### Inference
Firstly, run
```
python PC2RAD_inference.py
```
And predicted anomaly maps will be saved. Then run
```
python Blend_anomaly_maps.py
```
to save red-blue heatmaps for visualization.