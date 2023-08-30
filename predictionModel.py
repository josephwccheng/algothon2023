
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
