from VGGTrainFunc import GetTrainPara, train
from Tuning.tuning import ListPara, ParaStr2Dict, UpdateOpt
import os

#################################
# experiment setting
#################################
# Ep 1-48
ParaDict1 = {'DataSet': ['str', ['hade', 'dq8']], 
            'OutputPath': ['str', ['F:\\VelocityPicking\\VGG16']], 
            'SeedRate': ['float', [1.0]], 
            'InputSize': ['str', ['128,128', '64,64', '128,64']], 
            'VRange': ['str', ['1000,7000']], 
            'LabelLen': ['int', [40, 65]], 
            'CropNum': ['int', [45]], 
            'trainBS': ['int', [16, 32, 64, 128]],
            'StopMax': ['int', [30]], 
            'lrStart': ['float', [0.001]],
            'optimizer': ['str', ['adam', 'sgd']]}


#################################
# training
#################################
# get the experiment (ep) list
EpList = ListPara(ParaDict1)
# get default training parameters
OptDefault = GetTrainPara()
for ind, EpName in enumerate(EpList):
    # try:
    start = 1
    # get the ep para dict
    EpDict = ParaStr2Dict(EpName, ParaDict1)
    EpDict.setdefault('EpName', 'Ep-%d' % (ind+start))
    # judge whether done before
    if os.path.exists(os.path.join(EpDict['OutputPath'], 'Ep-%d' % (ind+start), 'Result.csv')):
        continue
    if os.path.exists(os.path.join(EpDict['OutputPath'], 'Ep-%d' % (ind+start), 'model', 'Best.pth')):
        EpDict.setdefault('ReTrain', 0)
    else:
        EpDict.setdefault('ReTrain', 1)
    # update the para
    EpOpt = UpdateOpt(EpDict, OptDefault)
    # start this experiment
    train(EpOpt)
    # except:
    #     print(EpName)
    #     continue
