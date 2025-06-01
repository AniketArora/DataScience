import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
# For more advanced features like entropy, you might need libraries like 'antropy' or 'nolds'
# For now, keeping it to pandas, numpy, scipy, statsmodels to avoid adding too many new deps
# unless explicitly requested and confirmed.

def extract_basic_stats(series: pd.Series, prefix=""):
    """Extracts basic statistical features from a series."""
    features = {}
    if series.empty:
        # Return NaNs or default values for all expected features if series is empty
        for stat in ['mean', 'std', 'median', 'min', 'max', 'skewness', 'kurtosis_val', 'sum', 'variance', 'iqr']:
            features[f'{prefix}{stat}'] = np.nan
        return features

    features[f'{prefix}mean'] = series.mean()
    features[f'{prefix}std'] = series.std()
    features[f'{prefix}median'] = series.median()
    features[f'{prefix}min'] = series.min()
    features[f'{prefix}max'] = series.max()
    features[f'{prefix}skewness'] = skew(series.dropna()) if not series.dropna().empty else np.nan
    features[f'{prefix}kurtosis_val'] = kurtosis(series.dropna()) if not series.dropna().empty else np.nan # Fisher's kurtosis (normal ==> 0)
    features[f'{prefix}sum'] = series.sum()
    features[f'{prefix}variance'] = series.var()
    features[f'{prefix}iqr'] = series.quantile(0.75) - series.quantile(0.25)
    return features

def extract_trend_features(series: pd.Series, prefix=""):
    """Extracts features related to the trend of the series."""
    features = {}
    if series.empty or len(series.dropna()) < 2: # Need at least 2 points for polyfit
        features[f'{prefix}slope'] = np.nan
        return features

    # Linear trend (slope)
    # Using series.values assumes numeric index if not reset, or just values if index is datetime
    # For robustness with DatetimeIndex, ensure x is numeric (e.g., seconds from start or simple sequence)
    series_cleaned = series.dropna()
    x = np.arange(len(series_cleaned))
    try:
        coeffs = np.polyfit(x, series_cleaned.values, 1)
        features[f'{prefix}slope'] = coeffs[0]
    except (np.linalg.LinAlgError, TypeError): # Adding TypeError for safety
        features[f'{prefix}slope'] = np.nan
    return features

def extract_volatility_features(series: pd.Series, prefix=""):
    """Extracts volatility features from the series."""
    features = {}
    if series.empty or len(series.dropna()) < 2: # Need at least 2 points for diff
        features[f'{prefix}mean_abs_diff'] = np.nan
        features[f'{prefix}std_diff'] = np.nan
        return features

    diff_series = series.dropna().diff().dropna()
    if diff_series.empty:
        features[f'{prefix}mean_abs_diff'] = 0.0 # If only one non-NaN point originally, diff is empty
        features[f'{prefix}std_diff'] = 0.0
        return features

    features[f'{prefix}mean_abs_diff'] = diff_series.abs().mean()
    features[f'{prefix}std_diff'] = diff_series.std()
    return features

def extract_autocorrelation_features(series: pd.Series, lags=[1, 5, 10], prefix=""):
    """Extracts autocorrelation features from the series."""
    features = {}
    series_cleaned = series.dropna()

    if series_cleaned.empty or len(series_cleaned) < max(lags) + 1 if lags else 1 : # Check if series is too short for max lag
        for lag in lags:
            features[f'{prefix}acf_lag_{lag}'] = np.nan
        return features

    try:
        # nlags should be less than len(series_cleaned)
        # acf function itself handles some short series issues, but good to be cautious
        max_nlags = len(series_cleaned) -1
        if max_nlags < 1: # Cannot compute ACF if less than 2 data points
             for lag in lags:
                features[f'{prefix}acf_lag_{lag}'] = np.nan
             return features

        acf_values = acf(series_cleaned, nlags=min(max(lags) if lags else 0, max_nlags), fft=False) # fft=False for robustness with NaNs handled by dropna
        for lag in lags:
            if lag < len(acf_values):
                features[f'{prefix}acf_lag_{lag}'] = acf_values[lag]
            else:
                features[f'{prefix}acf_lag_{lag}'] = np.nan
    except Exception: # Catch any error from acf function
        for lag in lags:
            features[f'{prefix}acf_lag_{lag}'] = np.nan
    return features

def extract_rolling_stats_features(series: pd.Series, windows, prefix=""):
    """Extracts features from rolling window statistics."""
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        for window in windows:
            for stat in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']: # Corrected key names
                 features[f'{prefix}rolling_{stat}_w{window}'] = np.nan
        return features

    for window in windows:
        if window > len(series_cleaned): # Window too large
            features[f'{prefix}rolling_mean_of_means_w{window}'] = np.nan # Corrected key names
            features[f'{prefix}rolling_mean_of_stds_w{window}'] = np.nan
            features[f'{prefix}rolling_std_of_means_w{window}'] = np.nan
            features[f'{prefix}rolling_std_of_stds_w{window}'] = np.nan
            continue

        # Taking the mean of the rolling stat over the series as a feature
        rolling_mean = series_cleaned.rolling(window=window, min_periods=1).mean()
        rolling_std = series_cleaned.rolling(window=window, min_periods=1).std()

        features[f'{prefix}rolling_mean_of_means_w{window}'] = rolling_mean.mean()
        features[f'{prefix}rolling_mean_of_stds_w{window}'] = rolling_std.mean()
        features[f'{prefix}rolling_std_of_means_w{window}'] = rolling_mean.std()
        features[f'{prefix}rolling_std_of_stds_w{window}'] = rolling_std.std()

    return features


def generate_all_features_for_series(series: pd.Series, name="ts_"):
    """
    Generates a comprehensive feature vector for a single time series.
    Args:
        series (pd.Series): The input time series.
        name (str): A prefix for all feature names.
    Returns:
        pd.Series: A pandas Series containing all extracted features.
    """
    all_features = {}

    # Ensure series is not all NaN, which would make most stats NaN or error
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        # If series is empty or all NaN, return a structure with NaNs for expected features
        # This helps in creating a consistent DataFrame structure when processing multiple series
        all_features.update(extract_basic_stats(series_cleaned, prefix=f"{name}basic_"))
        all_features.update(extract_trend_features(series_cleaned, prefix=f"{name}trend_"))
        all_features.update(extract_volatility_features(series_cleaned, prefix=f"{name}vol_"))
        all_features.update(extract_autocorrelation_features(series_cleaned, lags=[1, 5, 10], prefix=f"{name}acf_"))
        all_features.update(extract_rolling_stats_features(series_cleaned, windows=[5, 10, 20], prefix=f"{name}roll_"))
        # Add more feature categories here if needed
        return pd.Series(all_features)

    all_features.update(extract_basic_stats(series, prefix=f"{name}basic_"))
    all_features.update(extract_trend_features(series, prefix=f"{name}trend_"))
    all_features.update(extract_volatility_features(series, prefix=f"{name}vol_"))
    all_features.update(extract_autocorrelation_features(series, lags=[1, 5, 10], prefix=f"{name}acf_"))

    # Define windows for rolling stats based on series length to avoid issues
    len_s = len(series_cleaned)
    rolling_windows = []
    if len_s >= 10: rolling_windows.append(5)
    if len_s >= 20: rolling_windows.append(10)
    if len_s >= 40: rolling_windows.append(20)
    if not rolling_windows and len_s > 1: # at least one small window if series is very short but not empty
        rolling_windows.append(max(1, len_s // 2))

    if rolling_windows: # only call if there are valid windows
        all_features.update(extract_rolling_stats_features(series, windows=rolling_windows, prefix=f"{name}roll_"))
    else: # if no valid windows, fill with NaNs for consistency
        # This part ensures feature vector has same keys even for short series
        dummy_windows = [5,10,20] # These are the windows we *expect* features for
        for window in dummy_windows:
            for stat_type in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
                 all_features[f'{name}roll_{stat_type}_w{window}'] = np.nan


    # Placeholder for more advanced features:
    # features.update(extract_entropy_features(series, prefix=f"{name}entropy_"))
    # features.update(extract_peak_features(series, prefix=f"{name}peaks_"))

    return pd.Series(all_features)

if __name__ == '__main__':
    # Example Usage
    idx = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = [i + (i % 7) * 2 + np.random.randn() * 5 + (i/10)**1.5 for i in range(50)]
    sample_ts = pd.Series(data, index=idx, name="MySensor")
    sample_ts.iloc[[3, 10, 20, 35]] = np.nan # Add some NaNs

    print("--- Features for Sample Time Series ---")
    ts_features = generate_all_features_for_series(sample_ts, name="sensor_X_")
    print(ts_features)

    print("\n--- Features for a Short Time Series ---")
    short_ts = sample_ts.head(8) # Very short
    short_ts_features = generate_all_features_for_series(short_ts, name="short_")
    print(short_ts_features)

    print("\n--- Features for an Empty Time Series ---")
    empty_ts = pd.Series([], dtype=float)
    empty_ts_features = generate_all_features_for_series(empty_ts, name="empty_")
    print(empty_ts_features)

    print("\n--- Features for an All NaN Time Series ---")
    all_nan_ts = pd.Series([np.nan, np.nan, np.nan], dtype=float)
    all_nan_ts_features = generate_all_features_for_series(all_nan_ts, name="allnan_")
    print(all_nan_ts_features)
