import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# project defined parameters
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)

''' 
    AutoRegressive Integrated Moving Average(ARIMA)
    ARIMA(p,d,q)
        p - number of lagged (or past) observations to consider for autoregression
        d - the number of times the raw observations are differenced
        q - the size of the moving average window respectively
    reference: https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
'''


class ArimaModel:
    def __init__(self, prcSoFar):
        self.prcSoFar = prcSoFar
        (self.nins, self.nt) = self.prcSoFar.shape
        self.model: list[ARIMA] = []
        # Configure Input
        self.periodIndex = pd.date_range(
            "2023-01-01", periods=self.nt).to_period()
        # Train model on init
        self.fitAllIns()

    def getInsInputOutput(self, ins: int, out_of_sample=0):
        # Generating Instrument Input required for model training
        # Restructure the price into dataframe
        y = pd.DataFrame({'price': self.prcSoFar[ins]})
        y.index = self.periodIndex
        return y

    def fitAllIns(self):
        # Training a model for each Instrument
        for ins in range(0, self.nins):
            y = self.getInsInputOutput(ins)
            self.visualise(ins)
            model = self.fit(y)
            self.model.append(model)

    def fit(self, y):
        order = (2, 1, 1)  # ARIMA(p, d, q) order
        model = ARIMA(y, order=order)
        fit_model = model.fit()
        return fit_model

    def predict(self, ins: int, forecast_steps: int):
        forecast, stderr, conf_int = self.model[ins].forecast(
            steps=forecast_steps)
        return forecast, stderr, conf_int

    def visualise(self, ins: int):
        # determining the parameters in arima, p, d , and q
        y = pd.DataFrame(self.prcSoFar[ins], columns=['price'])
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title('0 Order Differencing')
        axs[0, 0].plot(y)
        plot_acf(y.diff().dropna(), ax=axs[0, 1])
        # order of difference d
        axs[1, 0].set_title('1st Order Differencing')
        axs[1, 0].plot(y.diff())
        plot_acf(y.diff().dropna(), ax=axs[1, 1])
        axs[2, 0].set_title('2nd Order Differencing')
        axs[2, 0].plot(y.diff())
        plot_acf(y.diff().dropna(), ax=axs[2, 1])


if __name__ == "__main__":
    print("(TODO) Arima Model running on sample data via prices.txt file")
