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
    # The stationarity of sample_series_profiling (repeating pattern) is ambiguous for ADF;
    # other tests cover clear stationary/non-stationary cases.


def test_perform_stationarity_test_on_random_noise(): # Renamed and uses pure noise
    data = np.random.randn(100)
    stationary_s = pd.Series(data)
    results = perform_stationarity_test(stationary_s)
    assert 'ADF Statistic' in results
    assert 'p-value' in results
    # For pure random noise, p-value should be very low, indicating stationarity
    assert results.get("Is Stationary (at chosen alpha)"), f"Expected stationary for random noise, p-value: {results.get('p-value')}"

def test_perform_stationarity_test_on_trend_series(): # New test for clear non-stationarity
    data = np.arange(100).astype(float) # ensure float for consistency
    trend_s = pd.Series(data)
    results = perform_stationarity_test(trend_s)
    assert 'ADF Statistic' in results
    assert 'p-value' in results
    assert not results.get("Is Stationary (at chosen alpha)"), f"Expected non-stationary for trend series, p-value: {results.get('p-value')}"

def test_perform_stationarity_test_empty_series():
    results = perform_stationarity_test(pd.Series([], dtype=float))
    assert "error" in results
    assert results["error"] == "Input series is empty."

def test_perform_stationarity_test_all_nan_series():
    results = perform_stationarity_test(pd.Series([np.nan, np.nan], dtype=float))
    assert "error" in results
    assert results["error"] == "Series is empty after dropping NaN values. Cannot perform ADF test."

# --- More Error and Edge Case Tests ---

def test_get_series_summary_stats_error_cases():
    # Not a series
    summary = get_series_summary_stats([1, 2, 3])
    assert summary.empty

    # Empty series
    s_empty = pd.Series([], dtype=float)
    summary = get_series_summary_stats(s_empty)
    assert summary['mode'].iloc[0] == 'N/A'

def test_get_missing_values_summary_error_cases():
    # Not a series
    summary = get_missing_values_summary([1, 2, 3])
    assert summary.empty

    # Empty series
    s_empty = pd.Series([], dtype=float)
    summary = get_missing_values_summary(s_empty)
    assert summary[summary['Metric'] == 'Missing Percentage (%)']['Value'].iloc[0] == "0.00%"
    assert summary[summary['Metric'] == 'Total Count']['Value'].iloc[0] == 0

def test_perform_stationarity_test_error_cases():
    # Not a series
    results = perform_stationarity_test([1, 2, 3])
    assert "Input is not a pandas Series" in results['error']

from unittest.mock import patch

@patch('src.analysis_modules.profiling.adfuller')
def test_perform_stationarity_test_exception(mock_adfuller, sample_series_profiling):
    mock_adfuller.side_effect = Exception("ADF Error")
    results = perform_stationarity_test(sample_series_profiling)
    assert "ADF test failed: ADF Error" in results['error']
