#!/usr/bin/env python
# Import necessary libraries
import numpy as np
import pandas as pd
from decompositionalModel import DecompositionalModel

nInst = 50
currentPos = np.zeros(nInst)
# project defined parameters
minLag = 250


def getMyPosition(prcSoFar):
    (nins, nt) = prcSoFar.shape
    if (nt < minLag):
        return np.zeros(nins)
    # currentPos = defaultDecisioning(prcSoFar)
    currentPos = teamDecisioning(prcSoFar)
    return currentPos


def defaultDecisioning(prcSoFar):
    global currentPos
    # log of previous day price / two days price
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    rpos = np.array([int(x) for x in 2000000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos


def teamDecisioning(prcSoFar):
    # Generating Signals if predicted increase > 1%, sell if predicted decrease > 2%)
    global currentPos
    (nins, nt) = prcSoFar.shape  # instruments and time
    decompositionalModel = DecompositionalModel(prcSoFar, 1)
    posLimits = np.array([int(x) for x in 10000 / prcSoFar[:, -1]])
    threshold = 0.04
    r2_threshold = 0.1
    mape_threshold = 0.01
    maxPosition_threshold = 0.3
    rollingWindow = 10
    # calculate moving average
    for ins in range(0, nins):
        predictedPrice = decompositionalModel.insPredictedPrice[ins]
        ins_prc_pd = pd.DataFrame(prcSoFar[ins])
        rollingMean = ins_prc_pd.rolling(
            window=rollingWindow).mean()
        std = rollingMean.std().values[0]
        mean = rollingMean.mean().values[0]
        # First if for validation of the model accuracy
        if (decompositionalModel.r2AllIns[ins] > r2_threshold and decompositionalModel.mapeAllIns[ins] < mape_threshold):
            # 'buy' if predicted price will hike up
            if predictedPrice > rollingMean.iloc[-1].values[0] * (1 + threshold):
                currentPos[ins] = currentPos[ins] + posLimits[ins] * \
                    (maxPosition_threshold + (predictedPrice - rollingMean.iloc[-1].values[0]) /
                     rollingMean.iloc[-1].values[0])
            # 'sell' if predicted price will go down
            elif predictedPrice < rollingMean.iloc[-1].values[0] * (1 + threshold):
                currentPos[ins] = currentPos[ins] - posLimits[ins] * \
                    (maxPosition_threshold + (rollingMean.iloc[-1].values[0] - predictedPrice) /
                     rollingMean.iloc[-1].values[0])
    return currentPos
