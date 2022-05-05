import os

import numpy as np
import segyio
import h5py
import torch
import torch.utils.data as data
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from torchvision import transforms


"""
Loading Sample Data from segy, h5, npy file
"""


# make ground truth curve
def interpolation(label_point, t_interval, v_interval=None):
    # sort the label points
    label_point = np.array(sorted(label_point, key=lambda t_v: t_v[0]))

    # ensure the input is int
    t0_vec = np.array(t_interval).astype(int)

    # get the ground truth curve using interpolation
    peaks_selected = np.array(label_point)
    func = interpolate.interp1d(peaks_selected[:, 0], peaks_selected[:, 1], kind='linear', fill_value="extrapolate")
    y = func(t0_vec)
    if v_interval is not None:
        v_vec = np.array(v_interval).astype(int) 
        y = np.clip(y, v_vec[0], v_vec[-1])

    return np.hstack((t0_vec.reshape((-1, 1)), y.reshape((-1, 1))))


def ScaleImage(SpecImg, resize_n, device='cpu'):
    C, H, W = SpecImg.shape
    SpecImg = torch.tensor(SpecImg, device=device).view(1, C, H, W)
    transform = transforms.Resize(resize_n).cuda(device)
    AugImg = transform(SpecImg)  
    return AugImg.squeeze()


# --------- Load source file ----------------------------------
def LoadSource(DataSetPath):
    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'stk': 'vel.stk.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(DataSetPath, 'segy', path), "r", strict=False))

    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'stk': 'StkInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(DataSetPath, 'h5File', path), 'r'))

    # load label.npy
    LabelDict = np.load(os.path.join(DataSetPath, 't_v_labels.npy'), allow_pickle=True).item()

    return SegyDict, H5Dict, LabelDict
    

# -------- load data from segy, h5 and label.npy ------------------
def LoadSingleData(SegyDict, H5Dict, LabelDict, index, mode='train'):
    # data dict
    DataDict = {}
    PwrIndex = np.array(H5Dict['pwr'][index]['SpecIndex'])
    line, cdp = index.split('_')

    DataDict.setdefault('spectrum', np.array(SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T))
    DataDict.setdefault('vInt', np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]]))

    if mode == 'train':
        try:
            DataDict.setdefault('label', np.array(LabelDict[int(line)][int(cdp)]))
        except KeyError:
            DataDict.setdefault('label', np.array(LabelDict[str(line)][str(cdp)]))

    return DataDict


# -------- make label vector --------------------------------------
def MakeLabel(value, VRange):
    LocIndex = np.where((VRange-value)>0)[0][0]-1
    return LocIndex


# -------- pytorch dataset iterator --------------------------------
"""
Load Data: 
    FM: resized spectrum      | shape = 2 * H * W
    label: vector label       | length =  40
    self.index[index]: sample index    | string
"""
class DLSpec(data.Dataset):
    def __init__(self, SegyDict, H5Dict, LabelDict, index, t0Ind, 
                 VRange=(2000, 4000), resize=(128, 128), CropNum=45, LabelLen=40, mode='train', device='cpu'):
        self.SegyDict = SegyDict
        self.LabelDict = LabelDict
        self.H5Dict = H5Dict
        self.index = index
        self.t0Ind = t0Ind
        self.vInd = np.linspace(VRange[0], VRange[1], LabelLen+1).astype(np.int32)
        self.resize = resize
        self.mode = mode
        self.VRange = VRange
        IndexSplit = np.linspace(0, len(self.t0Ind), num=CropNum+1).astype(np.int32)
        self.IndexSplit = np.array([IndexSplit[0:-1], IndexSplit[1:]]).T
        self.device = device

    def __getitem__(self, index):
        #########################################
        # load the data from segy and h5 file
        #########################################
        DataDict = LoadSingleData(self.SegyDict, self.H5Dict, self.LabelDict,
                                  self.index[index], self.mode)
        PwrImg = DataDict['spectrum']
        #########################################
        # make mask
        #########################################
        if self.mode != 'train':
            np.random.seed(index)
        start, end = self.IndexSplit[np.random.randint(0, self.IndexSplit.shape[0])]
        MaskImg = np.zeros_like(PwrImg)
        MaskImg[start: end, :] = PwrImg[start: end, :]
        #########################################                          
        # down sampling & concatenate
        #########################################
        FuseImg = np.concatenate((PwrImg[np.newaxis, ...], MaskImg[np.newaxis, ...]), axis=0)
        FuseImg = ScaleImage(FuseImg, self.resize, self.device)
        FuseImg = (FuseImg-torch.min(FuseImg))/(torch.max(FuseImg)-torch.min(FuseImg)+1e-6)
        #########################################
        # make label
        #########################################
        # interpolate the velocity curve
        VCAllRes = interpolation(DataDict['label'], self.t0Ind, v_interval=DataDict['vInt'])
        # get manual velocity 
        VRef = VCAllRes[int((start+end)/2), 1]
        Label = MakeLabel(VRef, self.vInd)
        return FuseImg, Label, self.index[index]

    def __len__(self):
        return len(self.index)


class PredLoad(data.Dataset):
    def __init__(self, SegyDict, H5Dict, LabelDict, index, t0Ind, 
                 VRange=(2000, 4000), resize=(128, 128), CropNum=45, LabelLen=40, device='cpu'):
        self.SegyDict = SegyDict
        self.LabelDict = LabelDict
        self.H5Dict = H5Dict
        self.index = index
        self.t0Ind = t0Ind
        self.vInd = np.linspace(VRange[0], VRange[1], LabelLen+1).astype(np.int32)
        self.resize = resize
        self.VRange = VRange
        IndexSplit = np.linspace(0, len(self.t0Ind), num=CropNum+1).astype(np.int32)
        self.IndexSplit = np.array([IndexSplit[0:-1], IndexSplit[1:]]).T
        self.device = device

    def __getitem__(self, index):
        #########################################
        # load the data from segy and h5 file
        #########################################
        DataDict = LoadSingleData(self.SegyDict, self.H5Dict, self.LabelDict,
                                  self.index[index])
        PwrImg = DataDict['spectrum']
        # interpolate the velocity curve
        VCAllRes = interpolation(DataDict['label'], self.t0Ind, v_interval=DataDict['vInt'])
        MaskSpecList, LabelList = [], []
        for start, end in self.IndexSplit:
            #########################################
            # make mask
            #########################################
            MaskImg = np.zeros_like(PwrImg)
            MaskImg[start: end, :] = PwrImg[start: end, :]
            #########################################                          
            # down sampling & concatenate
            #########################################
            MaskSpec = MaskImg[np.newaxis, ...]
            #########################################
            # make label
            #########################################
            # get manual velocity 
            VRef = VCAllRes[int((start+end)/2), 1]
            Label = MakeLabel(VRef, self.vInd)
            # add to list
            MaskSpecList.append(MaskSpec)
            LabelList.append(Label[np.newaxis, ...])
        
        #########################################
        # resize and normalization
        #########################################
        ConcatSpec = np.concatenate([PwrImg[np.newaxis, ...]] + MaskSpecList, axis=0)
        ConcatSpec = (ConcatSpec - np.min(ConcatSpec)) / np.ptp(ConcatSpec)
        ScaleImg = ScaleImage(ConcatSpec, self.resize, self.device)
        FinalFeats = []
        for i in range(1, ScaleImg.shape[0]):
            FinalFeats.append(torch.concat([ScaleImg[0].unsqueeze(0), ScaleImg[i].unsqueeze(0)], dim=0).unsqueeze(0))
        FinalFeats = torch.concat(FinalFeats, dim=0)
        FinalLabel = torch.tensor(LabelList, device=self.device)
        return FinalFeats, FinalLabel, VCAllRes, self.index[index]

    def __len__(self):
        return len(self.index)

    def GetPlotInfo(self, index):
        DataDict = LoadSingleData(self.SegyDict, self.H5Dict, self.LabelDict,
                                  index)
        return DataDict
        