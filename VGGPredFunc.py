import sys

sys.path.append('..')
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.LoadData import LoadSource, PredLoad

matplotlib.use('Agg')
import warnings

from model.VGG16 import VGG16
from utils.LoadData import LoadSource, interpolation
from utils.PlotTools import plot_spectrum

warnings.filterwarnings("ignore")


def GetPredPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OutputPath', type=str, default='F:\\VelocityPicking\\VGG16', help='Path of Output')
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum')
    parser.add_argument('--EpName', type=int, help='Model Path')
    parser.add_argument('--Resave', type=int, default=0)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--VisualNum', type=int, default=16, help='The batchsize of Predict')

    opt = parser.parse_args()
    return opt


def InvertVel(output, t0Ind, vInd, RowSplit):
    VCenter = (vInd[:-1] + vInd[1:])/2
    MaxIndex = np.argmax(output, axis=1)
    VEst = VCenter[MaxIndex]
    RowSplit[RowSplit==len(t0Ind)] = -1
    T0Est = (t0Ind[RowSplit[:, 0]] + t0Ind[RowSplit[:, 1]])/2
    AutoCurve = np.array([T0Est, VEst]).T
    return AutoCurve


def PredFunc(opt):
    ####################
    # base setting
    ####################
    BasePath = os.path.join(opt.OutputPath, 'Ep-%d'%opt.EpName)
    # setting model parameters
    ParaDict = pd.read_csv(os.path.join(BasePath, 'TrainPara.csv')).to_dict()
    PredSet = str(ParaDict['DataSet'][0])
    DataSetPath = os.path.join(opt.DataSetRoot, PredSet)
    
    # check output folder
    OutputPath = os.path.join(BasePath, 'predict', PredSet)
    if not os.path.exists(OutputPath): os.makedirs(OutputPath) 
    PlotRoot = os.path.join(OutputPath, 'fig')
    if not os.path.exists(PlotRoot): os.makedirs(PlotRoot)
    InputSize = list(map(int, ParaDict['InputSize'][0].split(',')))
    VRange = list(map(int, ParaDict['VRange'][0].split(',')))
    LabelLen = int(ParaDict['LabelLen'][0])
    CropNum = int(ParaDict['CropNum'][0])
    ModelPath = os.path.join(BasePath, 'model', 'Best.pth')
    PredictPath = os.path.join(OutputPath, '%s-VMAE.npy' % PredSet)
    # check gpu is available
    if torch.cuda.device_count() > 0:
        device = opt.GPUNO
    else:
        device = 'cpu'

    #######################################
    # load data from segy, csv index file
    #######################################
    # load source file
    SegyDict, H5Dict, LabelDict = LoadSource(DataSetPath)
    t0Int = np.array(SegyDict['pwr'].samples)
    #########################################
    # split the train, valid and test set
    #########################################
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))
    pwr_index = set(H5Dict['pwr'].keys())
    stk_index = set(H5Dict['stk'].keys())
    gth_index = set(H5Dict['gth'].keys())
    Index = sorted(list((pwr_index & stk_index) & (gth_index & set(HaveLabelIndex))))
    IndexDict = {}
    for index in Index:
        line, cdp = index.split('_')
        IndexDict.setdefault(int(line), [])
        IndexDict[int(line)].append(int(cdp))
    LineIndex = sorted(list(IndexDict.keys()))
    # use the last 20% for test set
    LastSplit2 = int(len(LineIndex)*0.8)
    # use the first sr% (seed rate) for train set and the other for valid set
    testLine = LineIndex[LastSplit2:]
    testIndex = []
    for line in testLine:
        for cdp in IndexDict[line]:
            testIndex.append('%d_%d' % (line, cdp))
    print('Test Num %d' % len(testIndex))
    
    ds = PredLoad(SegyDict, H5Dict, LabelDict, testIndex, t0Int, 
                  VRange=VRange, resize=InputSize, CropNum=CropNum, LabelLen=LabelLen, device=device)

    ###################################
    # load the network
    ###################################
    # load network
    net = VGG16(LabelLen, InputSize=InputSize)

    if device is not 'cpu':
        net = net.cuda(device)
    net.eval()
    # Load the weights of network
    if os.path.exists(ModelPath):
        print("Load Model Successfully! \n(%s)" % ModelPath)
        net.load_state_dict(torch.load(ModelPath)['Weights'])
    else:
        print("There is no such model file!")

    bar = tqdm(total=len(ds))
    VMAEList, count = [], 0
    # predict all of the dataset
    with torch.no_grad():
        for _, (Feats, _, MPCurve, FMIndex) in enumerate(ds):
            out = net(Feats)
            OutVec = out.squeeze().detach().cpu().numpy()
            AutoCurve = InvertVel(OutVec, ds.t0Ind, ds.vInd, ds.IndexSplit)
            APInterp = interpolation(AutoCurve, ds.t0Ind)
            VMAEList.append([FMIndex, np.mean(np.abs(APInterp[:, 1]-MPCurve[:, 1]))])
            bar.update(1)
            bar.set_description('%s-%s: VMAE %.3f' % (PredSet, FMIndex, VMAEList[-1][-1]))
            if count < opt.VisualNum:
                DataDict = ds.GetPlotInfo(FMIndex)
                plot_spectrum(DataDict['spectrum'], ds.t0Ind, DataDict['vInt'], VelCurve=[APInterp, MPCurve], save_path=os.path.join(PlotRoot, '%s-result.png')%FMIndex)
                count+=1
    bar.close()
    # save predict results
    np.save(PredictPath, np.array(VMAEList))
    print('%s Ep-%d VMAE = %.3f' % (PredSet, opt.EpName, np.mean(np.array(VMAEList)[:, 1].astype(np.float32))))



