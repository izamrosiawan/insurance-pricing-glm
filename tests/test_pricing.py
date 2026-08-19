import pytest
import pandas as pd
import numpy as np
from src.pricing_engine import PricingEngine

@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        'VehPower': 5,
        'VehAge': 2,
        'DrivAge': 30,
        'BonusMalus': 50,
        'VehBrand': 'B12',
        'VehGas': 'Regular',
        'Area': 'C',
        'Region': 'R24',
        'Density': 100
    }])

def test_pricing_engine_positive_prediction(sample_input):
    engine = PricingEngine()
    pred_twopart = engine.predict_pure_premium(sample_input, method='two_part')
    pred_tweedie = engine.predict_pure_premium(sample_input, method='tweedie')
    
    assert len(pred_twopart) == 1
    assert pred_twopart[0] > 0
    assert pred_tweedie[0] > 0

def test_pricing_engine_risk_monotonicity(sample_input):
    engine = PricingEngine()
    high_risk_input = sample_input.copy()
    high_risk_input['BonusMalus'] = 150 # Risk multiplier tinggi
    
    pred_low = engine.predict_pure_premium(sample_input, method='two_part')[0]
    pred_high = engine.predict_pure_premium(high_risk_input, method='two_part')[0]
    
    assert pred_high > pred_low
