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

## Test Result on two field datasets
| **DataSet** | **InputSize** | **LabelLen** | **CropNum** | **lrStart** | **optimizer** | **trainBS** | **valBS** |   **VMAE**   |
|:-----------:|:-------------:|:------------:|:-----------:|:-----------:|:-------------:|:-----------:|:---------:|:------------:|
|   **hade**  |   **128,64**  |    **65**    |    **45**   |  **0.001**  |    **adam**   |    **32**   |   **32**  | **45.03909** |
|   **hade**  |    128,128    |      65      |      45     |    0.001    |      adam     |     128     |     32    |   45.98352   |
|   **hade**  |     128,64    |      65      |      45     |    0.001    |      adam     |      64     |     32    |   46.22609   |
|   **hade**  |    128,128    |      40      |      45     |    0.001    |      adam     |     128     |     32    |   47.02694   |
|   **hade**  |    128,128    |      40      |      45     |    0.001    |      adam     |      64     |     32    |   52.71181   |
|   **hade**  |     128,64    |      40      |      45     |    0.001    |      adam     |     128     |     32    |   57.45961   |
|   **hade**  |     128,64    |      40      |      45     |    0.001    |      adam     |      64     |     32    |   58.46858   |
|   **hade**  |    128,128    |      40      |      45     |    0.001    |      adam     |      32     |     32    |   58.62806   |
|   **hade**  |     64,64     |      65      |      45     |    0.001    |      adam     |      32     |     32    |   62.42237   |
|   **hade**  |     128,64    |      40      |      45     |    0.001    |      adam     |      16     |     32    |   65.54008   |
|   **hade**  |    128,128    |      65      |      45     |    0.001    |      adam     |      16     |     32    |   65.88509   |
|   **hade**  |    128,128    |      65      |      45     |    0.001    |      adam     |      64     |     32    |   69.46835   |
|   **hade**  |     128,64    |      65      |      45     |    0.001    |      adam     |      16     |     32    |   70.92652   |
|   **hade**  |     128,64    |      65      |      45     |    0.001    |      adam     |     128     |     32    |   73.75591   |
|   **hade**  |    128,128    |      40      |      45     |    0.001    |      adam     |      16     |     32    |    89.0888   |
|   **hade**  |     64,64     |      40      |      45     |    0.001    |      adam     |     128     |     32    |   92.88287   |
|   **hade**  |     64,64     |      65      |      45     |    0.001    |      adam     |      64     |     32    |   96.49525   |
|   **hade**  |     64,64     |      65      |      45     |    0.001    |      adam     |     128     |     32    |   106.4157   |
|   **dq8**   |  **128,128**  |    **40**    |    **45**   |  **0.001**  |    **adam**   |   **128**   |   **32**  | **108.6157** |
|   **hade**  |     64,64     |      65      |      45     |    0.001    |      adam     |      16     |     32    |   113.4613   |
|   **hade**  |     64,64     |      40      |      45     |    0.001    |      adam     |      32     |     32    |   116.8338   |
|   **hade**  |     64,64     |      40      |      45     |    0.001    |      adam     |      64     |     32    |   127.6789   |
|   **dq8**   |    128,128    |      65      |      45     |    0.001    |      adam     |     128     |     32    |    137.454   |
|   **hade**  |     64,64     |      40      |      45     |    0.001    |      adam     |      16     |     32    |   141.2642   |
|   **dq8**   |    128,128    |      65      |      45     |    0.001    |      adam     |      64     |     32    |    144.924   |
|   **dq8**   |    128,128    |      40      |      45     |    0.001    |      adam     |      64     |     32    |   152.3124   |
|   **dq8**   |     64,64     |      40      |      45     |    0.001    |      adam     |     128     |     32    |   250.9188   |
|   **dq8**   |    128,128    |      65      |      45     |    0.001    |      adam     |      32     |     32    |   829.7051   |
|   **dq8**   |     64,64     |      65      |      45     |    0.001    |      adam     |      16     |     32    |   844.0259   |
|   **dq8**   |    128,128    |      40      |      45     |    0.001    |      adam     |      32     |     32    |   845.2664   |
|   **dq8**   |    128,128    |      40      |      45     |    0.001    |      adam     |      16     |     32    |   849.8641   |
|   **dq8**   |     64,64     |      65      |      45     |    0.001    |      adam     |      32     |     32    |   871.2813   |
|   **dq8**   |     64,64     |      40      |      45     |    0.001    |      adam     |      32     |     32    |   884.1399   |
|   **dq8**   |     64,64     |      40      |      45     |    0.001    |      adam     |      16     |     32    |   936.2207   |
|   **dq8**   |     64,64     |      40      |      45     |    0.001    |      adam     |      64     |     32    |   945.9525   |
|   **dq8**   |    128,128    |      65      |      45     |    0.001    |      adam     |      16     |     32    |   1133.827   |
|   **hade**  |     128,64    |      40      |      45     |    0.001    |      adam     |      32     |     32    |   1293.908   |
|   **hade**  |    128,128    |      65      |      45     |    0.001    |      adam     |      32     |     32    |   1344.482   |