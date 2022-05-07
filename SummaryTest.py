import os
import pandas as pd
import numpy as np


def SummaryResult(root, SavePath=None):
    EpFolderList = [os.path.join(root, file) for file in os.listdir(root)]
    ResultList = []
    for path in EpFolderList:
        ParaCsv = pd.read_csv(os.path.join(path, 'TrainPara.csv')).to_dict()
        PredSet = str(ParaCsv['DataSet'][0])
        VMAEPath = os.path.join(path, 'predict', PredSet, '%s-VMAE.npy'%PredSet)
        if not os.path.exists(VMAEPath):
            continue
        VMAEList = np.load(VMAEPath)
        ParaValues = [elm[0] for elm in ParaCsv.values()]
        ResultList.append(ParaValues+[np.mean(VMAEList[:, 1].astype(np.float32))])
    ColName = list(ParaCsv.keys()) + ['VMAE']
    ResultDF = pd.DataFrame(ResultList, columns=ColName)
    if SavePath is None:
        SavePath = 'summary\\VMAE.csv'
    ResultDF.to_csv(SavePath)


if __name__ == '__main__':
    RootPath = r'F:\\VelocityPicking\\VGG16'
    SummaryResult(RootPath)
    
