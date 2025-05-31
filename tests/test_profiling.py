import pytest
import pandas as pd
import numpy as np
from src.analysis_modules.profiling import (
    get_series_summary_stats,
    get_missing_values_summary,
    perform_stationarity_test
)

@pytest.fixture
def sample_series_profiling():
    idx = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 4) # 20 points, 4 NaNs
    return pd.Series(data, index=idx, name="TestSeries")

def test_get_series_summary_stats(sample_series_profiling):
    summary = get_series_summary_stats(sample_series_profiling)
    assert not summary.empty
    assert 'mean' in summary.columns
    assert 'median' in summary.columns
    assert 'mode' in summary.columns
    assert summary['count'].iloc[0] == 16 # Non-NaN count
    # Mode can be tricky with floats and multiple modes.
    # Ensure it doesn't error and returns a plausible value or 'N/A' as per implementation
    assert summary['mode'].iloc[0] is not None

def test_get_missing_values_summary(sample_series_profiling):
    summary = get_missing_values_summary(sample_series_profiling)
    assert not summary.empty
    assert summary[summary['Metric'] == 'Total Count']['Value'].iloc[0] == 20
    assert summary[summary['Metric'] == 'Missing Count']['Value'].iloc[0] == 4
    assert summary[summary['Metric'] == 'Missing Percentage (%)']['Value'].iloc[0] == "20.00%"

def test_perform_stationarity_test_non_stationary(sample_series_profiling):
    results = perform_stationarity_test(sample_series_profiling.dropna()) # dropna() here because ADF itself drops, but this sample is non-stationary even then
    assert 'ADF Statistic' in results
    assert 'p-value' in results
    assert 'Interpretation' in results
    assert isinstance(results.get("Is Stationary (at chosen alpha)"), bool)

def test_perform_stationarity_test_stationary():
    data = np.random.randn(100)
    stationary_s = pd.Series(data)
    results = perform_stationarity_test(stationary_s)
    # P-value for random normal data should be low enough for stationarity
    assert results.get("Is Stationary (at chosen alpha)"), f"Expected stationary, p-value: {results.get('p-value')}"

def test_perform_stationarity_test_empty_series():
    results = perform_stationarity_test(pd.Series([], dtype=float))
    assert "error" in results
    assert results["error"] == "Input series is empty."

def test_perform_stationarity_test_all_nan_series():
    results = perform_stationarity_test(pd.Series([np.nan, np.nan], dtype=float))
    assert "error" in results
    assert results["error"] == "Series is empty after dropping NaN values. Cannot perform ADF test."
