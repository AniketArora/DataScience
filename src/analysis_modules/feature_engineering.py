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
    if series.empty: # Actually checks if series itself is empty, not just after dropna
        series_cleaned = series.dropna() # this will also be empty
        for stat in ['mean', 'std', 'median', 'min', 'max', 'skewness', 'kurtosis_val', 'sum', 'variance', 'iqr']:
            features[f'{prefix}{stat}'] = np.nan
        return features

    # For non-empty series, proceed with calculations
    series_cleaned_for_stats = series.dropna() # Use a cleaned version for stats that need it
    if series_cleaned_for_stats.empty: # If all values were NaN
        for stat in ['mean', 'std', 'median', 'min', 'max', 'skewness', 'kurtosis_val', 'sum', 'variance', 'iqr']:
            features[f'{prefix}{stat}'] = np.nan
        # sum of an all-NaN series is 0 by default in pandas, but let's make it NaN for consistency
        features[f'{prefix}sum'] = np.nan
        return features

    features[f'{prefix}mean'] = series_cleaned_for_stats.mean()
    features[f'{prefix}std'] = series_cleaned_for_stats.std()
    features[f'{prefix}median'] = series_cleaned_for_stats.median()
    features[f'{prefix}min'] = series_cleaned_for_stats.min()
    features[f'{prefix}max'] = series_cleaned_for_stats.max()
    features[f'{prefix}skewness'] = skew(series_cleaned_for_stats) # scipy.stats.skew handles NaNs by ignoring them if not empty
    features[f'{prefix}kurtosis_val'] = kurtosis(series_cleaned_for_stats)
    features[f'{prefix}sum'] = series_cleaned_for_stats.sum()
    features[f'{prefix}variance'] = series_cleaned_for_stats.var()
    features[f'{prefix}iqr'] = series_cleaned_for_stats.quantile(0.75) - series_cleaned_for_stats.quantile(0.25)
    return features

def extract_trend_features(series: pd.Series, prefix=""):
    """Extracts features related to the trend of the series."""
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty or len(series_cleaned) < 2:
        features[f'{prefix}slope'] = np.nan
        return features

    x = np.arange(len(series_cleaned))
    try:
        coeffs = np.polyfit(x, series_cleaned.values, 1)
        features[f'{prefix}slope'] = coeffs[0]
    except (np.linalg.LinAlgError, TypeError):
        features[f'{prefix}slope'] = np.nan
    return features

def extract_volatility_features(series: pd.Series, prefix=""):
    """Extracts volatility features from the series."""
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty or len(series_cleaned) < 2:
        features[f'{prefix}mean_abs_diff'] = np.nan
        features[f'{prefix}std_diff'] = np.nan
        return features

    diff_series = series_cleaned.diff().dropna()
    if diff_series.empty: # Happens if series_cleaned had only 1 point
        features[f'{prefix}mean_abs_diff'] = 0.0
        features[f'{prefix}std_diff'] = 0.0
        return features

    features[f'{prefix}mean_abs_diff'] = diff_series.abs().mean()
    features[f'{prefix}std_diff'] = diff_series.std()
    return features

def extract_autocorrelation_features(series: pd.Series, lags=[1, 5, 10], prefix=""):
    """Extracts autocorrelation features from the series."""
    features = {}
    series_cleaned = series.dropna()

    if series_cleaned.empty:
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
        return features

    max_possible_nlags = len(series_cleaned) - 1
    if max_possible_nlags < 1:
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
        return features

    nlags_to_compute = 0
    if lags: nlags_to_compute = min(max(lags), max_possible_nlags)

    acf_values = []
    if nlags_to_compute >= 1 :
        try:
            acf_values = acf(series_cleaned, nlags=nlags_to_compute, fft=False)
        except Exception:
            for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
            return features

    for lag in lags:
        if lag > 0 and lag < len(acf_values): # acf_values[0] is lag 0
            features[f'{prefix}acf_lag_{lag}'] = acf_values[lag]
        else:
            features[f'{prefix}acf_lag_{lag}'] = np.nan
    return features

def extract_rolling_stats_features(series: pd.Series, windows, prefix=""):
    features = {}
    series_cleaned = series.dropna() # Key: operate on cleaned series

    if series_cleaned.empty:
        for window in windows:
            for stat_type in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
                 features[f'{prefix}rolling_{stat_type}_w{window}'] = np.nan
        return features

    for window in windows:
        if window > len(series_cleaned):
            for stat_type in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
                 features[f'{prefix}rolling_{stat_type}_w{window}'] = np.nan
            continue

        current_mean_of_means, current_mean_of_stds = np.nan, np.nan
        current_std_of_means, current_std_of_stds = np.nan, np.nan

        if window == 1:
            current_mean_of_means = series_cleaned.mean()
            current_mean_of_stds = 0.0
            current_std_of_means = series_cleaned.std() if len(series_cleaned) >= 2 else 0.0
            current_std_of_stds = 0.0
        else: # window > 1
            # min_periods=1 for mean is okay as it will just be the value itself.
            rolling_mean_intermediate = series_cleaned.rolling(window=window, min_periods=1).mean()
            # min_periods=2 for std ensures we have at least two points to calculate std.
            rolling_std_intermediate = series_cleaned.rolling(window=window, min_periods=2).std()

            rm_agg = rolling_mean_intermediate.dropna()
            rs_agg = rolling_std_intermediate.dropna()

            # print(f"DEBUG: Window {window}, Series (len {len(series_cleaned)}): {series_cleaned.to_dict() if not series_cleaned.empty else 'Empty'}")
            # print(f"DEBUG: Window {window}, rolling_mean_intermediate (len {len(rolling_mean_intermediate)}): {rolling_mean_intermediate.to_dict() if not rolling_mean_intermediate.empty else 'Empty'}")
            # print(f"DEBUG: Window {window}, rolling_std_intermediate (len {len(rolling_std_intermediate)}): {rolling_std_intermediate.to_dict() if not rolling_std_intermediate.empty else 'Empty'}")
            # print(f"DEBUG: Window {window}, rm_agg (len {len(rm_agg)}): {rm_agg.to_dict() if not rm_agg.empty else 'Empty'}")
            # print(f"DEBUG: Window {window}, rs_agg (len {len(rs_agg)}): {rs_agg.to_dict() if not rs_agg.empty else 'Empty'}")


            if not rm_agg.empty:
                current_mean_of_means = rm_agg.mean()
                if len(rm_agg) >= 2:
                    current_std_of_means = rm_agg.std()

            if not rs_agg.empty: # If rs_agg is empty (all NaNs from rolling_std_intermediate), mean/std will be NaN
                current_mean_of_stds = rs_agg.mean()
                if len(rs_agg) >= 2:
                    current_std_of_stds = rs_agg.std()

            # If the original series_cleaned was constant, std related features should be 0
            if series_cleaned.nunique() == 1:
                current_mean_of_stds = 0.0
                current_std_of_stds = 0.0

        features[f'{prefix}rolling_mean_of_means_w{window}'] = current_mean_of_means
        features[f'{prefix}rolling_mean_of_stds_w{window}'] = current_mean_of_stds
        features[f'{prefix}rolling_std_of_means_w{window}'] = current_std_of_means
        features[f'{prefix}rolling_std_of_stds_w{window}'] = current_std_of_stds

    return features

def generate_all_features_for_series(series: pd.Series, name="ts_"):
    all_features = {}
    all_features.update(extract_basic_stats(series, prefix=f"{name}basic_"))
    all_features.update(extract_trend_features(series, prefix=f"{name}trend_"))
    all_features.update(extract_volatility_features(series, prefix=f"{name}vol_"))
    all_features.update(extract_autocorrelation_features(series, lags=[1, 5, 10], prefix=f"{name}acf_"))

    series_cleaned_for_len = series.dropna()
    len_s = len(series_cleaned_for_len)

    dynamic_rolling_windows = []
    if len_s >= 40: dynamic_rolling_windows.extend([5,10,20])
    elif len_s >= 20: dynamic_rolling_windows.extend([5,10])
    elif len_s >= 10: dynamic_rolling_windows.append(5)

    if len_s > 0 :
        short_window_candidate = max(1, len_s // 2)
        is_standard_dynamic = short_window_candidate in [5,10,20]
        if not is_standard_dynamic and short_window_candidate not in dynamic_rolling_windows:
            if len_s >= short_window_candidate :
                dynamic_rolling_windows.append(short_window_candidate)
        if 1 not in dynamic_rolling_windows and len_s >=1 :
             dynamic_rolling_windows.append(1)

    dynamic_rolling_windows = sorted(list(set(w for w in dynamic_rolling_windows if 0 < w <= len_s)))

    if not series_cleaned_for_len.empty and dynamic_rolling_windows:
        computed_rolling_features = extract_rolling_stats_features(series, windows=dynamic_rolling_windows, prefix=f"{name}roll_")
        all_features.update(computed_rolling_features)

    fixed_expected_windows_for_output_keys = [1, 4, 5, 10, 20]
    for window_val in fixed_expected_windows_for_output_keys:
        for stat_type_val in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
            key_name = f'{name}roll_{stat_type_val}_w{window_val}'
            if key_name not in all_features:
                all_features[key_name] = np.nan

    return pd.Series(all_features)

if __name__ == '__main__':
    # ... (main block unchanged) ...
    idx = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = [i + (i % 7) * 2 + np.random.randn() * 5 + (i/10)**1.5 for i in range(50)]
    sample_ts = pd.Series(data, index=idx, name="MySensor")
    sample_ts.iloc[[3, 10, 20, 35]] = np.nan

    print("--- Features for Sample Time Series ---")
    ts_features = generate_all_features_for_series(sample_ts, name="sensor_X_")
    print(ts_features)

    print("\n--- Features for a Short Time Series ---")
    short_ts = sample_ts.head(8)
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
