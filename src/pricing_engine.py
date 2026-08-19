import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pricing_pipeline.joblib')

class PricingEngine:
    def __init__(self):
        self.pipeline_dict = joblib.load(MODEL_PATH)
        self.preprocessor = self.pipeline_dict['preprocessor']
        self.model_freq = self.pipeline_dict['model_freq']
        self.model_sev = self.pipeline_dict['model_sev']
        self.model_tweedie = self.pipeline_dict['model_tweedie']

    def predict_pure_premium(self, input_df: pd.DataFrame, method: str = 'two_part') -> np.ndarray:
        if 'LogDensity' not in input_df.columns and 'Density' in input_df.columns:
            input_df = input_df.copy()
            input_df['LogDensity'] = np.log(input_df['Density'])
            
        feature_cols = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Area', 'Region', 'LogDensity']
        X_trans = self.preprocessor.transform(input_df[feature_cols])
        
        if method == 'two_part':
            pred_freq = self.model_freq.predict(X_trans)
            pred_sev = self.model_sev.predict(X_trans)
            return pred_freq * pred_sev
        elif method == 'tweedie':
            return self.model_tweedie.predict(X_trans)
        else:
            raise ValueError("Method must be either 'two_part' or 'tweedie'")
