
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
# Time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.signal import periodogram

# project defined parameters
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


class DecompositionalModel:
    # series = trend + seasons + residuals
    def __init__(self, prcSoFar: np.ndarray, forcastHorizon: int):
        self.prcSoFar = prcSoFar
        (self.nins, self.nt) = self.prcSoFar.shape
        self.trendModel: list[LinearRegression] = []
        self.seasonModel: list[LinearRegression] = []
        self.residualModel: list[XGBRegressor] = []
        # Configure Input
        self.periodIndex = pd.date_range(
            "2023-01-01", periods=self.nt).to_period()
        # Train model on init
        self.fitAllIns()
        # Initialise prediction output
        self.insPredictedPrice = np.zeros(self.nins)
        self.predictAllIns(forcastHorizon=forcastHorizon)

        # Backtest
        self.mseAllIns, self.r2AllIns, self.mapeAllIns = self.backTestAllIns()

    def getInsInputOutput(self, ins: int, out_of_sample=0):
        # Generating Instrument Input required for model training
        # Restructure the price into dataframe
        y = pd.DataFrame({'price': self.prcSoFar[ins]})
        y.index = self.periodIndex
        # configure Input
        fourier = CalendarFourier(freq='D', order=6)
        dp = DeterministicProcess(
            constant=True,
            index=y.index,
            order=2,
            seasonal=True,
            drop=True,
            additional_terms=[fourier],
        )
        if out_of_sample == 0:
            X = dp.in_sample()
        else:
            X = dp.out_of_sample(out_of_sample)
        return X, y

    def backTestAllIns(self):
        mseAllIns = np.zeros(self.nins)
        r2AllIns = np.zeros(self.nins)
        mapeAllIns = np.zeros(self.nins)
        for ins in range(0, self.nins):
            X, y = self.getInsInputOutput(ins)
            mseAllIns[ins], r2AllIns[ins], mapeAllIns[ins] = self.backTest(
                X, y)
        return mseAllIns, r2AllIns, mapeAllIns

    def backTest(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False)
        trendModel, seasonModel, residualModel = self.fit(X_train, y_train)
        y_pred = self.predict(X_test, trendModel, seasonModel, residualModel)
        # Now obtain the KPI's
        mse_kpi = mean_squared_error(y_pred, y_test, squared=False)
        r2_kpi = r2_score(y_pred, y_test)
        mape_kpi = mean_absolute_percentage_error(y_pred, y_test)
        return mse_kpi, r2_kpi, mape_kpi

    def fitAllIns(self):
        # Training a model for each Instrument
        for ins in range(0, self.nins):
            X, y = self.getInsInputOutput(ins)
            trendModel, seasonModel, residualModel = self.fit(X, y)
            self.trendModel.append(trendModel)
            self.seasonModel.append(seasonModel)
            self.residualModel.append(residualModel)

    def fit(self, X, y):
        # break out trend (assuming linear trend)
        trendModel = self.trend(X[['const', 'trend']], y)
        y_trend = trendModel.predict(X[['const', 'trend']])
        y_detrend = y - y_trend
        # break out season (assuming linear trend)
        seasonModel = self.season(X, y_detrend)
        y_season = seasonModel.predict(X)
        y_deseason = y_detrend - y_season
        # break out residuals via xgboost -> non linear models
        residualModel = self.residuals(X, y_deseason)
        return trendModel, seasonModel, residualModel

    def predict(self, X_pred, trendModel, seasonModel, residualModel):
        y_trend_pred = pd.DataFrame(trendModel.predict(X_pred[['const', 'trend']]), columns=[
            "price"], index=X_pred.index)
        y_season_pred = pd.DataFrame(seasonModel.predict(X_pred), columns=[
            "price"], index=X_pred.index)
        y_residual_pred = pd.DataFrame(residualModel.predict(X_pred), columns=[
            "price"], index=X_pred.index)
        y_pred = y_trend_pred + y_season_pred + y_residual_pred
        return y_pred

    def predictIns(self, ins: int, steps: int):
        X_pred, _ = self.getInsInputOutput(ins, steps)
        y_pred = self.predict(
            X_pred, self.trendModel[ins], self.seasonModel[ins], self.residualModel[ins])
        # TODO mutlistep horizon not taken into account of here
        self.insPredictedPrice[ins] = y_pred.price.values[0]
        return y_pred

    def predictAllIns(self, forcastHorizon: int):
        for ins in range(0, self.nins):
            self.predictIns(ins, forcastHorizon)
        return self.insPredictedPrice

    def trend(self, X, y):
        trendModel = LinearRegression(fit_intercept=False)
        trendModel.fit(X, y)
        return trendModel

    def season(self, X, y_detrend):
        seasonModel = LinearRegression()
        seasonModel.fit(X, y_detrend)
        return seasonModel

    def residuals(self, X, y_deseason):
        residualModel = XGBRegressor()
        residualModel.fit(X, y_deseason)
        return residualModel


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
        y_fore = pd.Series(model.predict(X_test).flatten(), index=y_test.index)
        ax = y_train.plot(**plot_params)
        ax = y_test.plot(ax=ax)
        ax = y_pred.plot(ax=ax)
        _ = y_fore.plot(ax=ax, color='C3')


def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


if __name__ == "__main__":
    print("(TODO) Decompositional Model running on sample data via prices.txt file")

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
