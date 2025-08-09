import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, call # Import call for checking arguments
from src.analysis_modules.feature_engineering import (
    extract_basic_stats,
    extract_trend_features,
    extract_volatility_features,
    extract_autocorrelation_features,
    extract_rolling_stats_features,
    generate_all_features_for_series,
    extract_event_features_for_series,
    run_feature_engineering_for_all_devices # Make sure this is imported
)

@pytest.fixture
def normal_series():
    idx = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = [i + (i % 7) * 2 + np.random.randn() * 2 + (i/10)**1.5 for i in range(50)]
    return pd.Series(data, index=idx, name="NormalTS")

@pytest.fixture
def series_with_nans(normal_series):
    series_copy = normal_series.copy()
    series_copy.iloc[[5, 15, 25, 35]] = np.nan
    return series_copy

@pytest.fixture
def short_series(): # Length 8
    idx = pd.date_range(start='2023-01-01', periods=8, freq='D')
    return pd.Series(np.arange(8), index=idx, name="ShortTS")

@pytest.fixture
def very_short_series(): # Length 3
    idx = pd.date_range(start='2023-01-01', periods=3, freq='D')
    return pd.Series(np.arange(3), index=idx, name="VeryShortTS")

@pytest.fixture
def single_value_series():
    idx = pd.date_range(start='2023-01-01', periods=1, freq='D')
    return pd.Series([10.0], index=idx, name="SingleValueTS")

@pytest.fixture
def two_value_series():
    idx = pd.date_range(start='2023-01-01', periods=2, freq='D')
    return pd.Series([10.0, 12.0], index=idx, name="TwoValueTS")


@pytest.fixture
def empty_series():
    return pd.Series([], dtype=float, name="EmptyTS")

@pytest.fixture
def all_nan_series():
    idx = pd.date_range(start='2023-01-01', periods=10, freq='D')
    return pd.Series([np.nan]*10, index=idx, dtype=float, name="AllNaNTS")

@pytest.fixture
def constant_series():
    idx = pd.date_range(start='2023-01-01', periods=20, freq='D')
    return pd.Series([5.0]*20, index=idx, name="ConstantTS")

# --- Fixtures for Event Feature Testing ---
@pytest.fixture
def sample_device_events_fe():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-03', '2023-01-05', '2023-01-08', '2023-01-08', '2023-01-15', '2023-02-05']),
        'event_type': ['Error_A', 'Warning_B', 'Error_A', 'Info_C', 'Error_A', 'Error_A']
    })

@pytest.fixture
def normal_series_for_events():
    idx = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    data = np.arange(len(idx))
    return pd.Series(data, index=idx, name="EventTestSeries")


# --- Test individual feature extractors ---
def test_extract_basic_stats_normal(normal_series):
    features = extract_basic_stats(normal_series, "test_")
    assert 'test_mean' in features and not pd.isna(features['test_mean'])

def test_extract_basic_stats_constant(constant_series):
    features = extract_basic_stats(constant_series, "test_")
    assert features['test_mean'] == 5.0
    assert features['test_std'] == 0.0
    assert pd.isna(features['test_skewness'])
    assert pd.isna(features['test_kurtosis_val'])

def test_extract_trend_features_normal(normal_series):
    features = extract_trend_features(normal_series, "test_")
    assert not pd.isna(features['test_slope'])

def test_extract_volatility_features_normal(normal_series):
    features = extract_volatility_features(normal_series, "test_")
    assert not pd.isna(features['test_mean_abs_diff'])

def test_extract_autocorrelation_features_normal(normal_series):
    features = extract_autocorrelation_features(normal_series, lags=[1,2,5], prefix="test_")
    assert not pd.isna(features['test_acf_lag_1'])


# --- Tests for extract_event_features_for_series (existing, ensure they pass with new logic if any) ---
# These tests implicitly test the new `all_possible_event_types_cleaned_names` parameter when called from `generate_all_features_for_series`
# Direct tests for `extract_event_features_for_series` with this new parameter can be added if needed for more isolation.

# --- Tests for generate_all_features_for_series configuration ---
@patch('src.analysis_modules.feature_engineering.extract_autocorrelation_features')
def test_generate_all_features_uses_custom_acf_lags(mock_extract_acf, normal_series):
    custom_lags = [2, 4, 6]
    # Define a return value for the mock that includes the expected feature names
    mock_return_data = {f"ts_acf_acf_lag_{lag}": np.random.rand() for lag in custom_lags}
    mock_extract_acf.return_value = pd.Series(mock_return_data)

    features_series, _ = generate_all_features_for_series(normal_series, acf_lags_list=custom_lags)
    mock_extract_acf.assert_called_with(normal_series, lags=custom_lags, prefix="ts_acf_")
    for lag in custom_lags:
        assert f"ts_acf_acf_lag_{lag}" in features_series.index

@patch('src.analysis_modules.feature_engineering.extract_autocorrelation_features')
def test_generate_all_features_uses_default_acf_lags(mock_extract_acf, normal_series):
    default_lags = [1, 5, 10] # Default defined in generate_all_features
    # Define a return value for the mock
    mock_return_data = {f"ts_acf_acf_lag_{lag}": np.random.rand() for lag in default_lags}
    mock_extract_acf.return_value = pd.Series(mock_return_data)

    features_series, _ = generate_all_features_for_series(normal_series, acf_lags_list=None)
    mock_extract_acf.assert_called_with(normal_series, lags=default_lags, prefix="ts_acf_")
    # Optionally, assert that features from the mock are present
    for lag in default_lags:
        assert f"ts_acf_acf_lag_{lag}" in features_series.index

@patch('src.analysis_modules.feature_engineering.extract_rolling_stats_features')
def test_generate_all_features_uses_custom_rolling_windows(mock_extract_rolling, normal_series):
    custom_windows = [3, 7, 14]
    # Define a return value for the mock that includes one expected feature name per window
    mock_return_data = {}
    for w in custom_windows:
        mock_return_data[f"ts_roll_mean_of_means_w{w}"] = np.random.rand()
    mock_extract_rolling.return_value = pd.Series(mock_return_data)

    features_series, _ = generate_all_features_for_series(normal_series, rolling_windows_list=custom_windows)
    mock_extract_rolling.assert_called_with(normal_series, windows=custom_windows, prefix="ts_roll_")
    for w in custom_windows: # Check one stat type for each window
        assert f"ts_roll_mean_of_means_w{w}" in features_series.index

@patch('src.analysis_modules.feature_engineering.extract_rolling_stats_features')
def test_generate_all_features_uses_default_rolling_windows(mock_extract_rolling, normal_series):
    default_windows = [1, 5, 10, 20] # Default defined in generate_all_features
    # Define a return value for the mock
    mock_return_data = {}
    for w in default_windows:
        mock_return_data[f"ts_roll_mean_of_means_w{w}"] = np.random.rand()
    mock_extract_rolling.return_value = pd.Series(mock_return_data)

    features_series, _ = generate_all_features_for_series(normal_series, rolling_windows_list=None)
    mock_extract_rolling.assert_called_with(normal_series, windows=default_windows, prefix="ts_roll_")
    # Optionally, assert that features from the mock are present
    for w in default_windows:
        assert f"ts_roll_mean_of_means_w{w}" in features_series.index


# --- Test Refined extract_rolling_stats_features Logic ---
def test_rolling_stats_empty_series_input(empty_series):
    features = extract_rolling_stats_features(empty_series, windows=[1, 5], prefix="test_")
    assert pd.isna(features['test_mean_of_means_w1'])
    assert pd.isna(features['test_mean_of_stds_w5'])

def test_rolling_stats_all_nan_series_input(all_nan_series):
    features = extract_rolling_stats_features(all_nan_series, windows=[1, 5], prefix="test_")
    assert pd.isna(features['test_mean_of_means_w1'])
    assert pd.isna(features['test_mean_of_stds_w5'])

def test_rolling_stats_window_one(normal_series):
    features = extract_rolling_stats_features(normal_series, windows=[1], prefix="w1_")
    assert np.isclose(features['w1_mean_of_means_w1'], normal_series.dropna().mean())
    assert np.isclose(features['w1_std_of_means_w1'], normal_series.dropna().std())
    assert features['w1_mean_of_stds_w1'] == 0.0
    assert features['w1_std_of_stds_w1'] == 0.0

def test_rolling_stats_window_larger_than_series(short_series): # len 8
    L = len(short_series.dropna())
    features = extract_rolling_stats_features(short_series, windows=[L + 1, L + 5], prefix="test_")
    # With min_periods=1, even a window larger than the series will produce one aggregate value.
    for w in [L+1, L+5]:
        assert pd.notna(features[f'test_mean_of_means_w{w}'])
        assert pd.notna(features[f'test_std_of_means_w{w}'])
        assert pd.notna(features[f'test_mean_of_stds_w{w}'])
        assert pd.notna(features[f'test_std_of_stds_w{w}'])

def test_rolling_stats_constant_series(constant_series): # len 20, value 5.0
    features = extract_rolling_stats_features(constant_series, windows=[2, 5], prefix="const_")
    for w in [2, 5]:
        assert features[f'const_mean_of_means_w{w}'] == 5.0
        assert features[f'const_std_of_means_w{w}'] == 0.0
        assert features[f'const_mean_of_stds_w{w}'] == 0.0
        assert features[f'const_std_of_stds_w{w}'] == 0.0

def test_rolling_stats_short_series_for_std_min_periods(very_short_series): # len 3
    # Window 3, min_periods for std = max(2, 3//2=1) = 2
    # rolling_std_intermediate will have one value (std of [0,1,2])
    features = extract_rolling_stats_features(very_short_series, windows=[3], prefix="short3_")
    assert not pd.isna(features['short3_mean_of_means_w3'])
    assert not pd.isna(features['short3_std_of_means_w3']) # std of (mean of [0,1,2]) is 0
    assert not pd.isna(features['short3_mean_of_stds_w3'])
    assert features['short3_std_of_stds_w3'] == pytest.approx(0.514, abs=1e-3)

    # Window 2, min_periods for std = 2
    # rolling_std_intermediate will have two values (std of [0,1], std of [1,2])
    features_w2 = extract_rolling_stats_features(very_short_series, windows=[2], prefix="short2_")
    assert not pd.isna(features_w2['short2_mean_of_stds_w2'])
    assert not pd.isna(features_w2['short2_std_of_stds_w2'])


# --- Test Consistent NaN Output for extract_basic_stats ---
def test_basic_stats_single_value_series(single_value_series): # Series([10.0])
    features = extract_basic_stats(single_value_series)
    assert features['mean'] == 10.0
    assert pd.isna(features['std']) # Pandas std of single value is NaN (ddof=1)
    assert pd.isna(features['variance'])
    assert pd.isna(features['skewness']) # Scipy skew of single value is NaN
    assert pd.isna(features['kurtosis_val']) # Scipy kurtosis of single value is NaN
    assert features['iqr'] == 0.0 # q75(10) - q25(10) = 0

def test_basic_stats_two_value_series(two_value_series): # Series([10.0, 12.0])
    features = extract_basic_stats(two_value_series)
    assert features['mean'] == 11.0
    assert not pd.isna(features['std'])
    assert not pd.isna(features['variance'])
    assert pd.notna(features['skewness']) # Scipy skew for 2 values is 0
    assert pd.notna(features['kurtosis_val']) # Scipy kurtosis for 2 values is -2 (or NaN if fisher=False)
    assert features['iqr'] == 1.0 # (11.5 - 10.5)


# --- Tests for run_feature_engineering_for_all_devices (existing, ensure they pass) ---
# (Copied from previous state, assuming they are relevant and up-to-date)
@pytest.fixture
def sample_data_for_orchestration():
    data = {
        'device_id': ['dev1', 'dev1', 'dev1', 'dev2', 'dev2', 'dev2', 'dev3', 'dev3'],
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00',
            '2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00',
            '2023-01-01 00:00:00', '2023-01-01 00:01:00'
        ]),
        'value': [10, 12, 11, 20, 22, 21, 30, 31]
    }
    data_df_original = pd.DataFrame(data)
    ts_specs = {
        "id_cols": ["device_id"], "timestamp_col": "timestamp",
        "selected_value_col_for_analysis": "value",
        "event_device_id_col": "device_id", "event_timestamp_col": "event_timestamp",
        "event_event_type_col": "event_type",
        "top_n_event_types_for_series_features": 2,
        "acf_lags": [1, 2], "rolling_windows": [2, 3] # Add new configurable params
    }
    event_data = {
        'device_id': ['dev1', 'dev1', 'dev2'],
        'event_timestamp': pd.to_datetime(['2023-01-01 00:00:30', '2023-01-01 00:01:30', '2023-01-01 00:00:30']),
        'event_type': ['Error_X', 'Warning_Y', 'Error_X']
    }
    event_df = pd.DataFrame(event_data)
    global_top_event_types_cleaned = ["Error_X", "Warning_Y", "Info_Z"]
    return data_df_original, ts_specs, event_df, global_top_event_types_cleaned

@patch('src.analysis_modules.feature_engineering.generate_all_features_for_series')
def test_run_all_devices_success_with_configs(mock_generate_features, sample_data_for_orchestration):
    data_df, ts_specs, event_df, global_events = sample_data_for_orchestration
    mock_feature_series = pd.Series({'value_basic_mean': 10.5})
    mock_generate_features.return_value = (mock_feature_series, None)

    result_df, error_list = run_feature_engineering_for_all_devices(
        data_df, ts_specs, event_df, global_events
    )
    assert not result_df.empty
    assert len(result_df) == 3
    assert mock_generate_features.call_count == 3
    assert not error_list

    # Check if acf_lags and rolling_windows from ts_specs were passed through
    for call_args in mock_generate_features.call_args_list:
        _, kwargs = call_args
        assert kwargs.get('acf_lags_list') == ts_specs['acf_lags']
        assert kwargs.get('rolling_windows_list') == ts_specs['rolling_windows']

def test_run_all_devices_empty_input_orchestration(sample_data_for_orchestration):
    _, ts_specs, event_df, global_events = sample_data_for_orchestration
    empty_df = pd.DataFrame(columns=['device_id', 'timestamp', 'value'])
    result_df, error_list = run_feature_engineering_for_all_devices(
        empty_df, ts_specs, event_df, global_events
    )
    assert result_df.empty
    assert not error_list # Should not produce errors, just empty result

@patch('src.analysis_modules.feature_engineering.generate_all_features_for_series')
def test_run_all_devices_generate_features_returns_error_orchestration(mock_generate_features, sample_data_for_orchestration):
    data_df, ts_specs, event_df, global_events = sample_data_for_orchestration
    mock_series_dev1 = pd.Series({'value_basic_mean': 10})

    def side_effect_func(series, name, device_event_df, all_possible_event_types, event_type_col, event_ts_col, top_n_event_types_to_focus, acf_lags_list, rolling_windows_list):
        # Using series name (derived from value col name + processed suffix) to distinguish calls for different devices
        if name.startswith("value_"): # crude check
            if series.iloc[0] == 10: # dev1
                 return (mock_series_dev1, None)
            elif series.iloc[0] == 20: # dev2
                 return (None, "Simulated processing error for dev2")
            elif series.iloc[0] == 30: # dev3
                 return (pd.Series(dtype=float), None) # Empty series
        return (None, "Unknown series for side effect")


    mock_generate_features.side_effect = side_effect_func

    result_df, error_list = run_feature_engineering_for_all_devices(
        data_df, ts_specs, event_df, global_events
    )
    assert not result_df.empty
    assert len(result_df) == 1
    assert result_df.index[0] == 'dev1'
    assert 'value_basic_mean' in result_df.columns
    assert mock_generate_features.call_count == 3
    assert len(error_list) == 2 # This will fail if the "No features generated" case isn't added
    assert any(item[0] == 'dev2' and "Simulated processing error for dev2" in item[1] for item in error_list)
    assert any(item[0] == 'dev3' and "No features generated, skipping." in item[1] for item in error_list)

# --- Tests for generate_all_features_for_series with corrected rolling stats ---

def test_generate_all_features_normal_rolling(normal_series):
    features, _ = generate_all_features_for_series(normal_series, name="ts_", rolling_windows_list=[5, 10])
    assert 'ts_roll_mean_of_means_w5' in features and not pd.isna(features['ts_roll_mean_of_means_w5'])
    assert 'ts_roll_mean_of_stds_w5' in features and not pd.isna(features['ts_roll_mean_of_stds_w5'])


def test_generate_all_features_short_series_rolling(short_series): # len 8
    features, _ = generate_all_features_for_series(short_series, name="ts_", rolling_windows_list=[1, 4, 8, 10])
    assert 'ts_roll_mean_of_means_w4' in features and not pd.isna(features['ts_roll_mean_of_means_w4'])
    assert 'ts_roll_mean_of_means_w1' in features and not pd.isna(features['ts_roll_mean_of_means_w1'])
    # Window of 8 on series of length 8 should be fine
    assert 'ts_roll_mean_of_means_w8' in features and not pd.isna(features['ts_roll_mean_of_means_w8'])
    # Window of 10 on series of length 8 should also produce a result with min_periods=1
    assert 'ts_roll_mean_of_means_w10' in features and not pd.isna(features['ts_roll_mean_of_means_w10'])


def test_generate_all_features_constant_series_rolling(constant_series):
    features, _ = generate_all_features_for_series(constant_series, name="ts_", rolling_windows_list=[5])
    assert features['ts_roll_mean_of_stds_w5'] == 0.0
    assert features['ts_roll_std_of_stds_w5'] == 0.0
    assert features['ts_roll_std_of_means_w5'] == 0.0
    assert pd.isna(features['ts_acf_acf_lag_1'])
