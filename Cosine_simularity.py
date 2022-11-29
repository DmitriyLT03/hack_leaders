import numpy as np
from sklearn.linear_model import LinearRegression

class Cosine_simularity:
    def __init__(self):
        self.linreg = LinearRegression()
    
    def count_arctg(self, y_train):
        X_train = np.array(range(len(y_train))).reshape(-1, 1)
        trained_model = self.linreg.fit(X_train, y_train)
        return trained_model.coef_
    
    def count_average_linreg(self, array_first, array_second, array_third):
        arctg_first = self.count_arctg(array_first)
        arctg_second = self.count_arctg(array_second)
        arctg_third = self.count_arctg(array_third)
        cosine_second = np.cos(arctg_first - arctg_second)
        if cosine_second < 0:
            return None
        cosine_third = np.cos(arctg_first - arctg_third)
        if cosine_third < 0:
            return None
        result = (cosine_second * array_second + cosine_third * array_third)/(cosine_second + cosine_third)
        return result