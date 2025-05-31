import pytest
import pandas as pd
import numpy as np
from src.analysis_modules.decomposition import decompose_time_series
from statsmodels.tsa.seasonal import DecomposeResult

@pytest.fixture
def sample_series_decomposition():
    idx = pd.date_range(start='2023-01-01', periods=30, freq='D') # Enough for period 7
    data = [i % 7 + (i/10) + np.random.rand() for i in range(30)]
    return pd.Series(data, index=idx, name="DecompTest")

def test_decompose_time_series_additive(sample_series_decomposition):
    result, error = decompose_time_series(sample_series_decomposition, model='additive', period=7)
    assert error is None
    assert isinstance(result, DecomposeResult)
    assert not result.trend.empty
    assert not result.seasonal.empty
    assert not result.resid.empty

def test_decompose_time_series_multiplicative(sample_series_decomposition):
    positive_series = sample_series_decomposition + abs(sample_series_decomposition.min()) + 1
    result, error = decompose_time_series(positive_series, model='multiplicative', period=7)
    assert error is None
    assert isinstance(result, DecomposeResult)

def test_decompose_time_series_no_datetimeindex():
    s = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    result, error = decompose_time_series(s, period=7)
    assert result is None
    assert "Series index must be a DatetimeIndex" in error

def test_decompose_time_series_insufficient_data(sample_series_decomposition):
    short_series = sample_series_decomposition.head(10)
    result, error = decompose_time_series(short_series, period=7)
    assert result is None
    assert "Not enough data points" in error

def test_decompose_time_series_no_period_no_freq():
    idx = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-10'])
    s = pd.Series([1,2,3,4], index=idx)
    s.index.freq = None
    result, error = decompose_time_series(s)
    assert result is None
    assert "Series frequency is not set. Please specify a 'period'" in error
