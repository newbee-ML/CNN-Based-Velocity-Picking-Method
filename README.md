# Reproduced code of HRAwCNN for automatic velocity analysis
This repository replicates a method of velocity analysis, but we do not provide the dataset because it is classified.

Cite: Ferreira R S, Oliveira D A B, Semin D G, et al. Automatic velocity analysis using a hybrid regression approach with convolutional neural networks[J]. IEEE Transactions on Geoscience and Remote Sensing, 2020, 59(5): 4464-4470.

## Data preparation
You need prepara two segy(sgy) files which conclude velocity spectra and CMP gather infomation, and a label file which conclude the velocity labels.

## Implement
There are three parts for implementing the method proposed by Ferreira et. al: 
1) Generate the CropNMO dataset for training Xception Network. 
2) Training Xception Network. 
3) Predict processing

Tips:
You have to change a few path settings, if you want to test these method on your datasets.

### Generate the CropNMO dataset for training Xception Network
```cmd
python GenerateCNNData.py --dataset hade --CropSize 256,256
python GenerateCNNData.py --dataset dq8 --CropSize 256,256
```

### Training Xception Network
```cmd
python XceptionTrainMain.py
```


### Predict processing

```cmd
python HRAwCNNPredMain.py
```