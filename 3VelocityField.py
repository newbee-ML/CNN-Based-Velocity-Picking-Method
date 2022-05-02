import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.evaluate import GetResult
from utils.PlotTools import *
from utils.LoadData import interpolation


class BuildVelField:
    def __init__(self, EpName, line, RootPath, DataRoot):
        self.PickPath = os.path.join(RootPath, EpName)
        ParaDict = pd.read_csv(os.path.join(self.PickPath, 'TrainPara.csv')).to_dict()
        self.DataSet = ParaDict['DataSet'][0]
        self.line = line
        self.PickDict = np.load(os.path.join(self.PickPath, 'predict', ParaDict['DataSet'][0], '0-PickDict.npy'), allow_pickle=True).item()
        DataPath = os.path.join(DataRoot, ParaDict['DataSet'][0])
        self.LabelDict = np.load(os.path.join(DataPath, 't_v_labels.npy'), allow_pickle=True).item()
        APDict = self.GetAP(self.PickDict)
        self.GetVelocityField(APDict)

    ################################################################
    # get the AP with different predict threshold
    ################################################################
    def GetAP(self, SegDict):
        APDict = {}
        PtList = np.linspace(0.1, 0.3, 10)
        bar = tqdm(total=len(list(SegDict.keys())), file=sys.stdout)
        for name, PickDict in SegDict.items():
            line, cdp = map(int, name.split('_'))
            if line != int(self.line):
                bar.set_description(name)
                bar.update(1)
                continue
            # get the picking results under different prediction threshold
            for pt in PtList:
                AP, _ = GetResult(PickDict['Seg'][np.newaxis, ...], PickDict['TInt'], [PickDict['VInt']], threshold=pt)
                APDict.setdefault(line, {})
                APDict[line].setdefault(pt, {})
                APDict[line][pt].setdefault(cdp, AP)
            # load manual picking result
            if type(list(self.LabelDict.keys())[0]) == str:
                lineM, cdpM = str(line), str(line)
            else:
                lineM, cdpM = line, cdp
            if lineM in list(self.LabelDict.keys()):
                if cdpM in list(self.LabelDict[lineM].keys()):
                    APDict[line].setdefault('MP', {})
                    APDict[line]['MP'].setdefault(cdp, interpolation(self.LabelDict[lineM][cdpM], PickDict['TInt'], PickDict['VInt']))
            bar.set_description(name)
            bar.update(1)
        bar.close()
        return APDict

    ################################################################
    # plot the velocity field
    ################################################################
    def GetVelocityField(self, APDict):
        for line, ResultDict in APDict.items():
            SavePath = os.path.join(self.PickPath, 'predict', self.DataSet, 'VelocityField', str(line))
            if not os.path.exists(SavePath):
                os.makedirs(SavePath)
            for pt, APResult in ResultDict.items():
                cdpList = sorted(list(APResult.keys()))
                SegDict = self.PickDict['%d_%d'%(line, cdpList[0])]
                VelField = np.array([np.squeeze(APResult[cdp][..., 1]) for cdp in cdpList]).T
                try:
                    PlotVelField(VelField, cdpList, SegDict['TInt'], SegDict['VInt'], str(line), os.path.join(SavePath, 'pt-%.2f.pdf'% float(pt)))
                except ValueError:
                    PlotVelField(VelField, cdpList, SegDict['TInt'], SegDict['VInt'], str(line), os.path.join(SavePath, 'pt-%s.pdf'% pt))


if __name__ == '__main__':
    ################################################################
    # predict experiment list
    ################################################################
    RootPath = 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest'
    DataRoot = 'E:\\Spectrum'
    EpList = {'A': {'Index': [144, 223, 146, 203], 'Line': 3200}}#,
              #'B': {'Index': [140, 141, 142, 213], 'Line': 940}}
    # BuildVelField('Ep-20', 940, RootPath, DataRoot)
    for data, InfoDict in EpList.items():
        for EpNum in InfoDict['Index']:
            BuildVelField('Ep-%d' % EpNum, InfoDict['Line'], RootPath, DataRoot)

