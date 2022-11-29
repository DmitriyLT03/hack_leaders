import numpy as np
import pandas as pd
from scipy import interpolate
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

class AR_predict:
    def __init__(self, num_interpolation: float, lag=None):
        self.num_interpolation = num_interpolation

    def create_data_sample(self, data_full):
        x_line = data_full.period.unique().astype('datetime64[M]').astype(int) % 12
        y_line = np.array(data_full.groupby(['period']).sum()['Stoim'])
        return x_line, y_line
    
    def interpolate(self, xn_interp, yn_interp):
        f = interpolate.interp1d(xn_interp, yn_interp, kind='cubic')
        xnew = np.linspace(1, len(xn_interp), len(xn_interp) * self.num_interpolation - 1)
        ynew = f(xnew)
        return xnew, ynew
    
    def forward(self, xnew, ynew, length):
        mod = ar_select_order(ynew, maxlag=5, old_names=False)
        forecaster = AutoReg(ynew, lags=mod.ar_lags).fit()
        last_number = length * self.num_interpolation - 1
        y_pred = forecaster.model.predict(forecaster.params, start=len(xnew), end=len(xnew) + last_number - 1)
        return y_pred
    
    def predict(self, data, count_predict):
        x_before_interp, y_before_interp = self.create_data_sample(data)
        x_after_interp, y_after_interp = self.interpolate((x_before_interp + 1), y_before_interp)
        predict_cost = self.forward(x_after_interp, y_after_interp, count_predict)
        return predict_cost[1::self.num_interpolation]