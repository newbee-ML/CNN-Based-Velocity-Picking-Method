from VGGPredFunc import GetPredPara, PredFunc
from Tuning.tuning import UpdateOpt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    for i in range(5, 39):
        # get test parameters
        OptDefault = GetPredPara()
        OptDefault.EpName = i
        # start this experiment
        PredFunc(OptDefault)

