#!/usr/bin/env python
# Import necessary libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
# Time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.signal import periodogram

nInst = 50
currentPos = np.zeros(nInst)

# project defined parameters
minLag = 10
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


def getMyPosition(prcSoFar):
    (nins, nt) = prcSoFar.shape
    if (nt < minLag):
        return np.zeros(nins)
    insPredictedPrice = decompositionalModel(prcSoFar)
    # currentPos = defaultModel(prcSoFar)
    currentPos2 = teamDecisioning(prcSoFar, insPredictedPrice)
    # currentPos3 = lagLinearModel(prcSoFar)
    return currentPos2


def defaultModel(prcSoFar):
    global currentPos
    # log of previous day price / two days price
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    rpos = np.array([int(x) for x in 2000000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos


def teamDecisioning(prcSoFar, insPredictedPrice):
    (nins, nt) = prcSoFar.shape  # instruments and time
    currentPos = np.zeros(nins)
    curPrices = prcSoFar[:, -1]
    posLimits = np.array([int(x) for x in 10000 / curPrices])
    # Generating Signals if predicted increase > 1%, sell if predicted decrease > 2%)
    threshold = 0.03
    # calculate moving average
    for ins in range(0, nins):
        predictedPrice = insPredictedPrice[ins]
        ins_prc_pd = pd.DataFrame(prcSoFar[ins])
        rollingMean_5 = ins_prc_pd.rolling(
            window=5).mean().mean().values[0]
        overallMean = ins_prc_pd.rolling(
            window=nt).mean().mean().values[0]
        metric = predictedPrice - rollingMean_5
        # buy if predicted price will hike up
        if predictedPrice > curPrices[ins] and metric > 0 and metric / rollingMean_5 >= threshold:
            currentPos[ins] = posLimits[ins]
        elif predictedPrice < curPrices[ins] and metric < 0 and metric / rollingMean_5 >= threshold:
            currentPos[ins] = -posLimits[ins]
        # print(
        #     f'predicted is `{predictedPrice} and rollingMean_5 is `{rollingMean_5}` and overall mean is `{overallMean}`')

    return currentPos


def decompositionalModel(prcSoFar):
    # series = trend + seasons + residuals
    (nins, nt) = prcSoFar.shape  # instruments and time
    insPredictedPrice = np.zeros(nins)
    for ins in range(0, nins):
        y = pd.DataFrame({'price': prcSoFar[ins]})
        idx = pd.date_range("2023-01-01", periods=len(y))
        idx = idx.to_period()
        y.index = idx

        # deterministic
        fourier = CalendarFourier(freq='D', order=6)
        dp = DeterministicProcess(
            constant=True,
            index=y.index,
            order=2,
            seasonal=True,
            drop=True,
            additional_terms=[fourier],
        )
        X = dp.in_sample()
        # break out trend (assuming linear trend)
        y_trend, trendModel = trend(X[['const', 'trend']], y)
        y_detrend = y - y_trend
        # break out season (assuming linear trend)
        y_season, seasonModel = season(X, y_detrend)
        y_deseason = y_detrend - y_season
        # break out residuals via xgboost -> non linear models
        y_residuals, residualModel = residuals(X, y_deseason)
        y_deresid = y_deseason - y_residuals

        y_pred = decompositPredict(
            dp, 1, trendModel, seasonModel, residualModel)
        insPredictedPrice[ins] = y_pred.price.values[0]
    return insPredictedPrice


def decompositPredict(dp, steps, trendModel, seasonModel, residualModel):
    X = dp.out_of_sample(steps=steps)
    y_trend_pred = trendModel.predict(X[['const', 'trend']])
    y_season_pred = seasonModel.predict(X)
    y_residual_pred = pd.DataFrame(residualModel.predict(X), columns=[
                                   "price"], index=X.index)
    y_pred = y_trend_pred + y_season_pred + y_residual_pred
    return y_pred


def trend(X, y):
    trendModel = LinearRegression(fit_intercept=False)
    trendModel.fit(X, y)
    y_pred = trendModel.predict(X)
    return y_pred, trendModel


def season(X, y_detrend):
    seasonModel = LinearRegression()
    seasonModel.fit(X, y_detrend)
    y_pred = seasonModel.predict(X)
    return y_pred, seasonModel


def residuals(X, y_deseason):
    xgb = XGBRegressor()
    xgb.fit(X, y_deseason)
    y_residuals = pd.DataFrame(
        xgb.predict(X), columns=["price"], index=y_deseason.index)
    return y_residuals, xgb


def lagLinearModel(prcSoFar):
    (nins, nt) = prcSoFar.shape  # instruments and time
    insPredictedPrice = np.zeros(nins)
    for ins in range(0, nins):
        y = pd.DataFrame({'price': prcSoFar[ins]})
        idx = pd.date_range("2023-01-01", periods=len(y))
        idx = idx.to_period()
        y.index = idx
        all_data_lag = make_lags(y.price, lags=6)
        all_data_lag['y'] = y
        all_data_lag = all_data_lag.dropna()
        y_lag = all_data_lag.pop('y')
        X_lag = all_data_lag
        X_train, X_test, y_train, y_test = train_test_split(
            X_lag, y_lag, shuffle=False)
        # # Fit and predict
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(
            X_train).flatten(), index=y_train.index)
        y_fore = pd.Series(model.predict(
            X_test).flatten(), index=X_test.index)
        mse = mean_squared_error(y_test, y_fore, squared=False)
        # print("mse is: ", mse)
        y_fore = pd.Series(model.predict(X_test).flatten(), index=y_test.index)
        ax = y_train.plot(**plot_params)
        ax = y_test.plot(ax=ax)
        ax = y_pred.plot(ax=ax)
        _ = y_fore.plot(ax=ax, color='C3')

    return


def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


# Commented out graphing functions:
# trend
    # X_fore = dp.out_of_sample(steps=10)
    # y_fore = trendModel.predict(X_fore)
    # ax = plt.plot(X.trend, y)
    # ax = plt.plot(X.trend, y_train)
    # ax = plt.plot(X.trend, y_deseason)

# season
    # y_fore = pd.Series(seasonModel.predict(
    #     X_test).flatten(), index=X_test.index)
    # ax = y_pred.plot(**plot_params)
    # ax = y_detrend.plot(ax=ax)
    # _ = y_fore.plot(ax=ax, color='C3')

# lagmodel
    # y_fore = pd.Series(model.predict(X_test).flatten(), index=y_test.index)
    # ax = y_train.plot(**plot_params)
    # ax = y_test.plot(ax=ax)
    # ax = y_pred.plot(ax=ax)
    # _ = y_fore.plot(ax=ax, color='C3')
###
