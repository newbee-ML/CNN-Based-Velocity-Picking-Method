# Reproduced code of CNN-Based automatic velocity analysis method
This repository replicates a method of velocity analysis, but we do not provide the dataset because it is classified.

Cite: Park M J, Sacchi M D. Automatic velocity analysis using convolutional neural network and transfer learning[J]. Geophysics, 2020, 85(1): V33-V43.

## Data preparation
You need prepara a segy(sgy) file which includes velocity spectra, and a label file which includes the velocity labels. You have to build the h5 file for the index of samples, as shown in https://github.com/newbee-ML/MIFN-Velocity-Picking/blob/master/utils/BuiltStkDataSet.py

## Implement
There are three parts for implementing the method proposed by Park et al.: 
1) Training VGG16 classifier
2) Test the VGG16 on the spectra 
3) Summary Test results

Tips:
You have to change a few path settings, if you want to test these method on your datasets.

### Generate the CropNMO dataset for training Xception Network
```cmd
Training VGG16 classifier
python VGGTrainMain.py
```

### Test the VGG16 on the spectra 
```cmd
python VGGPredMain.py
```


### Summary Test results

```cmd
python SummaryTest.py
```