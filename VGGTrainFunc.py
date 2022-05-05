

"""
The main file for train VGG16 for velocity picking
Author: Hongtao Wang | stolzpi@163.com
"""
import sys
from ast import Raise

sys.path.append('..')
import argparse
import copy
import os
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model.VGG16 import VGG16
from utils.LoadData import DLSpec, LoadSource
from utils.logger import MyLog
from utils.metrics import MetricsMain
from utils.PlotTools import PlotTensor

warnings.filterwarnings("ignore")


"""
Initialize the folder
"""

def CheckSavePath(opt, BaseName):
    basicFile = ['log', 'model', 'TBLog']
    SavePath = os.path.join(opt.OutputPath, BaseName)
    if opt.ReTrain:
        if os.path.exists(SavePath):
            shutil.rmtree(SavePath)
        if not os.path.exists(SavePath):
            for file in basicFile:
                Path = os.path.join(SavePath, file)
                os.makedirs(Path)

"""
Save the training parameters
"""
def SaveParameters(opt, BaseName):
    ParaDict = opt.__dict__
    ParaDict = {key: [value] for key, value in ParaDict.items()}
    ParaDF = pd.DataFrame(ParaDict)
    ParaDF.to_csv(os.path.join(opt.OutputPath, BaseName, 'TrainPara.csv'))


"""
Get the hyper parameters
"""
def GetTrainPara():
    parser = argparse.ArgumentParser()
    ###########################################################################
    # path setting
    ###########################################################################
    parser.add_argument('--EpName', type=str, default='Ep-1', help='The index of the experiment')
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum')
    parser.add_argument('--OutputPath', type=str, default='F:\\VelocityPicking\\VGG16', help='Path of Output')
    parser.add_argument('--ReTrain', type=int, default=1)
    ###########################################################################
    # load data setting
    ###########################################################################
    parser.add_argument('--DataSet', type=str, default='dq8', help='Dataset List')
    parser.add_argument('--SeedRate', type=float, default=1)
    parser.add_argument('--InputSize', type=str, default='64,64', help='The size of input image')
    parser.add_argument('--VRange', type=str, default='1000,7000', help='The range of velocity domain')
    parser.add_argument('--LabelLen', type=int, default=40, help='The length of label vector')
    parser.add_argument('--CropNum', type=int, default=45, help='The number of sub-image')
    ###########################################################################
    # training setting
    ###########################################################################
    parser.add_argument('--StopMax', type=int, default=10)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--MaxIter', type=int, default=20000, help='max iteration')
    parser.add_argument('--SaveIter', type=int, default=50, help='checkpoint each SaveIter')
    parser.add_argument('--MsgIter', type=int, default=10, help='log the loss each MsgIter')
    parser.add_argument('--lrStart', type=float, default=0.01, help='the beginning learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help=r"the optimizer of training, 'adam' or 'sgd'")
    parser.add_argument('--PretrainModel', type=str, help='The path of pretrain model to train (Path)')
    parser.add_argument('--trainBS', type=int, default=32, help='The batchsize of train')
    parser.add_argument('--valBS', type=int, default=32, help='The batchsize of valid')
    opt = parser.parse_args()
    return opt
    

"""
Main train function
"""

def train(opt):
    ####################
    # base setting
    ####################
    BaseName = opt.EpName
    # check output folder and check path
    CheckSavePath(opt, BaseName)
    DataSetPath = os.path.join(opt.DataSetRoot, opt.DataSet)
    TBPath = os.path.join(opt.OutputPath, BaseName, 'TBLog')
    writer = SummaryWriter(TBPath)
    BestPath = os.path.join(opt.OutputPath, BaseName, 'model', 'Best.pth')
    LogPath = os.path.join(opt.OutputPath, BaseName, 'log')
    logger = MyLog(BaseName, LogPath)
    logger.info('%s start to train ...' % BaseName)
    # save the train parameters to csv
    SaveParameters(opt, BaseName)
    InputSize = tuple(map(int, list(opt.InputSize.split(','))))
    VRange = tuple(map(int, list(opt.VRange.split(','))))

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
    LastSplit1, LastSplit2 = int(len(LineIndex)*0.6), int(len(LineIndex)*0.8)
    # use the first sr% (seed rate) for train set and the other for valid set
    MedSplit = int(LastSplit1*opt.SeedRate)
    trainLine, validLine, testLine = LineIndex[:MedSplit], LineIndex[LastSplit1: LastSplit2], LineIndex[LastSplit2:]
    logger.info('There are %d lines, using for training: ' % len(trainLine) + ','.join(map(str, trainLine)))
    logger.info('There are %d lines, using for valid: ' % len(validLine) + ','.join(map(str, validLine)))
    logger.info('There are %d lines, using for test: ' % len(testLine) + ','.join(map(str, testLine)))
    trainIndex, validIndex = [], []
    for line in trainLine:
        for cdp in IndexDict[line]:
            trainIndex.append('%d_%d' % (line, cdp))
    for line in validLine:
        for cdp in IndexDict[line]:
            validIndex.append('%d_%d' % (line, cdp))
    print('Train Num %d, Valid Num %d' % (len(trainIndex), len(validIndex)))
    random.seed(456)
    VisualSample = random.sample(trainIndex, 16)

    ##################################
    # build the data loader
    ##################################
    # check gpu is available
    if torch.cuda.device_count() > 0:
        device = opt.GPUNO
    else:
        device = 'cpu'
    # build data loader
    ds = DLSpec(SegyDict, H5Dict, LabelDict, trainIndex, t0Int, 
                VRange=VRange, resize=InputSize, CropNum=opt.CropNum, LabelLen=opt.LabelLen, mode='train', device=device)
    dsval = DLSpec(SegyDict, H5Dict, LabelDict, validIndex, t0Int, 
                   VRange=VRange, resize=InputSize, CropNum=opt.CropNum, LabelLen=opt.LabelLen, mode='train', device=device)
    dl = DataLoader(ds,
                    batch_size=opt.trainBS,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True)
    dlval = DataLoader(dsval,
                       batch_size=opt.valBS,
                       shuffle=True,
                       pin_memory=False,
                       num_workers=0,
                       drop_last=True)

    ###################################
    # load the network
    ###################################
    # load network
    net = VGG16(opt.LabelLen, InputSize=InputSize)

    if device is not 'cpu':
        net = net.cuda(device)
    net.train()

    # load pretrain model or last model
    if opt.PretrainModel is None:
        if os.path.exists(BestPath):
            print("Load Last Model Successfully!")
            LoadModelDict = torch.load(BestPath)
            net.load_state_dict(LoadModelDict['Weights'])
            TrainParaDict = LoadModelDict['TrainParas']
            countIter, epoch = TrainParaDict['it'], TrainParaDict['epoch']
            BestValidLoss, lrStart = TrainParaDict['bestLoss'], TrainParaDict['lr']
        else:
            print("Start a new training!")
            countIter, epoch, lrStart, BestValidLoss = 0, 1, opt.lrStart, 1e10
    else:
        print("Load Pretrain Model Successfully!")
        LoadModelDict = torch.load(opt.PretrainModel)
        net.load_state_dict(LoadModelDict['Weights'])
        countIter, epoch, lrStart, BestValidLoss = 0, 1, opt.lrStart, 1e10
    
    # loss setting
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # define the optimizer
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lrStart)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lrStart, momentum=0.9)
    else:
        Raise("Error: invalid optimizer") 

    # define the lr_scheduler of the optimizer
    scheduler = MultiStepLR(optimizer, [10], 0.1)

    ####################################
    # training iteration 
    ####################################

    # initialize
    LossList, EarlyStopCount, PredList, LabelList = [], 0, [], []

    # start the iteration
    diter = iter(dl)
    for _ in range(opt.MaxIter):
        if countIter % len(dl) == 0 and countIter > 0:
            epoch += 1
            scheduler.step()
        countIter += 1
        try:
            FuseImg, label, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            FuseImg, label, _ = next(diter)
        label = label.cuda(device)
        optimizer.zero_grad()
        out = net(FuseImg)
        out = out.squeeze()
        # compute loss
        loss = criterion(out, label)
        # update parameters
        loss.backward()
        optimizer.step()
        LossList.append(loss.item())
        # save loss lr & seg map
        writer.add_scalar('Train-Loss', loss.item(), global_step=countIter)
        writer.add_scalar('Train-Lr', optimizer.param_groups[0]['lr'], global_step=countIter)
        # print the log per opt.MsgIter
        if countIter % opt.MsgIter == 0:
            # compute metrics: acc, recall, precision
            out, label = out.detach().cpu().numpy(), label.detach().cpu().numpy()
            pred = np.array([np.argmax(out[i]) for i in range(out.shape[0])])
            PredList += list(pred)
            LabelList += list(label)
            metrics = MetricsMain(PredList, LabelList)
            for name, value in metrics.items():
                writer.add_scalar('Train-%s'%name, value, global_step=countIter)
            lr = optimizer.param_groups[0]['lr']
            msg = 'it: %d/%d, epoch: %d, lr: %.6f, train-loss: %.7f' % (countIter, opt.MaxIter, epoch, lr, sum(LossList) / len(LossList))
            logger.info(msg)
            LossList = []

        
        # check points
        if countIter % opt.SaveIter == 0:  
            net.eval()
            # evaluator
            with torch.no_grad():
                ValidLoss, Metrics = EvaluateValid(net, dlval, criterion)
                for name, value in Metrics.items():
                    writer.add_scalar('Valid-%s'%name, value, global_step=countIter)
                writer.add_scalar('Valid-Loss', ValidLoss, global_step=countIter)

            if ValidLoss < BestValidLoss:
                BestValidLoss = copy.deepcopy(ValidLoss)
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                StateDict = {
                    'TrainParas': {'lr': optimizer.param_groups[0]['lr'], 
                                   'it': countIter,
                                   'epoch': epoch,
                                   'bestLoss': BestValidLoss},
                    'Weights': state}
                torch.save(StateDict, BestPath)
                EarlyStopCount = 0
            else:
                # count 1 time
                EarlyStopCount += 1
                # reload checkpoint pth
                if os.path.exists(BestPath):
                    net.load_state_dict(torch.load(BestPath)['Weights'])
                # if do not decreate for 10 times then early stop
                if EarlyStopCount > opt.StopMax:
                    break
            
            # write the valid log
            try:
                logger.info('it: %d/%d, epoch: %d, BestLoss: %.6f, Loss: %.6f' %
                        (countIter, opt.MaxIter, epoch, BestValidLoss, ValidLoss))
            except TypeError:
                logger.info('it: %d/%d, epoch: %d, TypeError')
            net.train()

    # save the finish csv
    ResultDF = pd.DataFrame({'BestValidLoss': [BestValidLoss]})
    ResultDF.to_csv(os.path.join(opt.OutputPath, BaseName, 'Result.csv'))


# main function for valid processing
def EvaluateValid(net, DataLoader, criterion):
    """
    main function for valid processing in trainSingle.py

    Params:
    - net: the network, type=class
    - DataLoader: data loader, type=class
    - criterion: loss function
    ---

    Return:
    - mean valid loss
    - mean accurate rate
    """
    # init save list and path
    LossAvg, PredList, LabelList = [], [], []
    Count = 0
    # valid iteration
    for _, (FuseImg, label, _) in enumerate(DataLoader):
        out = net(FuseImg)
        label = label.cuda(FuseImg.device)
        # compute loss
        loss = criterion(out.squeeze(), label.squeeze())
        LossAvg.append(loss.item())
        out, label = out.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()
        pred = np.array([np.argmax(out[i]) for i in range(out.shape[0])])
        PredList += list(pred)
        LabelList += list(label)
    # calculate the mean of metrics
    metrics = MetricsMain(PredList, LabelList)
    return sum(LossAvg) / len(LossAvg), metrics

