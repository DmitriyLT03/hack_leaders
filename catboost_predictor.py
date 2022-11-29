import numpy as np
import pandas as pd
import catboost as cb




class catboost_predictor:
    def __init__(self, model_path: str):
        self.model = cb.CatBoostRegressor()
        self.model.load_model(model_path)
        self.frame_size = (len(self.model.feature_names_) - 1) // 2

    def prepare_data(self, data_orig: pd.DataFrame):
        # переводим дату в месяц
        data = data_orig.copy()
        data.period = data_orig.period.values.astype('datetime64[M]').astype(int)
        row = {'tnved': data.tnved[0]}
        for j in range(self.frame_size):
            row['period_'+str(j)] = data.iloc[j]['period']
            row['stoim_'+str(j)] = data.iloc[j]['Stoim']
        pred_in = pd.DataFrame([row])
        return pred_in
        
    def predict(self, data_orig: pd.DataFrame, npoints):
        if len(data_orig) != self.frame_size:
            raise Exception(f'Incorrect number of points given: got {len(data_orig)}, expected {self.frame_size}')

        data = data_orig.copy()
        pred_in = self.prepare_data(data)
        predictions = np.empty(npoints, dtype=float)

        for i in range(npoints):
            predictions[i] = self.model.predict(pred_in)
            # add new prediction to data
            for j in range(self.frame_size-1):
                pred_in[f'period_{j}'] = pred_in[f'period_{j+1}']
                pred_in[f'stoim_{j}'] = pred_in[f'stoim_{j+1}']
            pred_in[f'stoim_{self.frame_size-1}'] = predictions[i]
            pred_in[f'period_{self.frame_size-1}'] = pred_in[f'period_{self.frame_size-2}'] + 1
        
        return predictions

