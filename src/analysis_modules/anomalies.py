import pandas as pd
import numpy as np # Import numpy
import streamlit as st # Not used in functions but often included

def detect_anomalies_zscore(series: pd.Series, threshold=3, window=None):
    """
    Detects anomalies in a time series using the Z-score method.
    Anomalies are points where the Z-score exceeds the threshold.
    Can use a rolling window for calculating mean and std dev if window is specified.

    Args:
        series (pd.Series): The time series to analyze.
        threshold (float): The Z-score threshold for identifying anomalies.
        window (int, optional): The rolling window size. If None, uses global mean/std.

    Returns:
        pd.Series: A boolean Series of the same index as input, True where anomalies are detected.
        pd.Series: The Z-scores.
        str or None: An error message if detection fails.
    """
    if not isinstance(series, pd.Series):
        return None, None, "Input is not a pandas Series."
    if series.empty:
        return None, None, "Input series is empty."

    series_cleaned = series.dropna()
    if series_cleaned.empty:
        return None, None, "Series is empty after dropping NaN values."

    try:
        if window:
            if not isinstance(window, int) or window <= 0:
                return None, None, "Window must be a positive integer."
            if window >= len(series_cleaned):
                 return None, None, f"Window size ({window}) is too large for the series length ({len(series_cleaned)})."

            # Rolling mean and std
            rolling_mean = series_cleaned.rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = series_cleaned.rolling(window=window, center=True, min_periods=1).std()

            # Fill NaNs at the beginning/end of rolling calculations (e.g., if center=True)
            # For Z-score, we need to ensure we can calculate it for all points in series_cleaned
            # Reindex to align with series_cleaned and then backfill/ffill
            rolling_mean = rolling_mean.reindex(series_cleaned.index).bfill().ffill() # Updated fillna
            rolling_std = rolling_std.reindex(series_cleaned.index).bfill().ffill() # Updated fillna

            # Avoid division by zero if std is 0 for some windows
            rolling_std[rolling_std == 0] = np.nan # Replaced pd.np.nan with np.nan

            z_scores = (series_cleaned - rolling_mean) / rolling_std
        else:
            # Global mean and std
            mean = series_cleaned.mean()
            std = series_cleaned.std()
            if std == 0: # Avoid division by zero if series is constant
                z_scores = pd.Series(np.nan, index=series_cleaned.index) # Replaced pd.np.nan with np.nan
            else:
                z_scores = (series_cleaned - mean) / std

        anomalies = (z_scores.abs() > threshold)

        # Reindex anomalies and z_scores to original series index, filling non-evaluated points (original NaNs) as False/NaN
        anomalies_reindexed = anomalies.reindex(series.index, fill_value=False)
        z_scores_reindexed = z_scores.reindex(series.index)

        return anomalies_reindexed, z_scores_reindexed, None
    except Exception as e:
        return None, None, f"Z-score anomaly detection failed: {e}"


def detect_anomalies_iqr(series: pd.Series, multiplier=1.5):
    """
    Detects anomalies in a time series using the Interquartile Range (IQR) method.
    Anomalies are points outside Q1 - multiplier*IQR and Q3 + multiplier*IQR.

    Args:
        series (pd.Series): The time series to analyze.
        multiplier (float): The IQR multiplier to define outlier bounds.

    Returns:
        pd.Series: A boolean Series of the same index as input, True where anomalies are detected.
        pd.DataFrame or None: A DataFrame containing Q1, Q3, IQR, Lower Bound, Upper Bound.
        str or None: An error message if detection fails.
    """
    if not isinstance(series, pd.Series):
        return None, None, "Input is not a pandas Series."
    if series.empty:
        return None, None, "Input series is empty."

    series_cleaned = series.dropna()
    if series_cleaned.empty:
        return None, None, "Series is empty after dropping NaN values."

    try:
        Q1 = series_cleaned.quantile(0.25)
        Q3 = series_cleaned.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)

        anomalies = (series_cleaned < lower_bound) | (series_cleaned > upper_bound)

        bounds_info = pd.DataFrame({
            'Metric': ['Q1', 'Q3', 'IQR', 'Lower Bound', 'Upper Bound'],
            'Value': [Q1, Q3, IQR, lower_bound, upper_bound]
        })

        # Reindex anomalies to original series index, filling non-evaluated points (original NaNs) as False
        anomalies_reindexed = anomalies.reindex(series.index, fill_value=False)

        return anomalies_reindexed, bounds_info, None
    except Exception as e:
        return None, None, f"IQR anomaly detection failed: {e}"

if __name__ == '__main__':
    # Example Usage
    idx = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = [i % 20 + pd.np.random.randn() for i in range(100)] # Some periodic data
    sample_series = pd.Series(data, index=idx)
    sample_series.iloc[10] = 50  # Add a clear anomaly
    sample_series.iloc[20] = -30 # Add another clear anomaly
    sample_series.iloc[[5, 15, 25]] = pd.NA # Add some missing values

    print("--- Z-score Anomalies (Global) ---")
    anomalies_z, z_scores, error_z = detect_anomalies_zscore(sample_series, threshold=2.5)
    if error_z:
        print(f"Error: {error_z}")
    else:
        print("Anomalies detected at:", anomalies_z[anomalies_z].index.tolist())
        # print("Z-scores of anomalies:\n", z_scores[anomalies_z])


    print("\n--- Z-score Anomalies (Rolling Window=10) ---")
    anomalies_z_roll, z_scores_roll, error_z_roll = detect_anomalies_zscore(sample_series, threshold=2.0, window=10)
    if error_z_roll:
        print(f"Error: {error_z_roll}")
    else:
        print("Anomalies detected at (rolling):", anomalies_z_roll[anomalies_z_roll].index.tolist())
        # print("Z-scores of anomalies (rolling):\n", z_scores_roll[anomalies_z_roll])


    print("\n--- IQR Anomalies ---")
    anomalies_iqr, bounds, error_iqr = detect_anomalies_iqr(sample_series, multiplier=1.5)
    if error_iqr:
        print(f"Error: {error_iqr}")
    else:
        print("Anomalies detected at:", anomalies_iqr[anomalies_iqr].index.tolist())
        print("Bounds Info:\n", bounds)

    print("\n--- Error Handling Examples ---")
    empty_s = pd.Series([], dtype=float)
    all_nan_s = pd.Series([pd.NA, pd.NA], dtype=float)

    _, _, err_z_empty = detect_anomalies_zscore(empty_s)
    print(f"Z-score on empty: {err_z_empty}")
    _, _, err_z_nan = detect_anomalies_zscore(all_nan_s)
    print(f"Z-score on all_nan: {err_z_nan}")

    _, _, err_iqr_empty = detect_anomalies_iqr(empty_s)
    print(f"IQR on empty: {err_iqr_empty}")
    _, _, err_iqr_nan = detect_anomalies_iqr(all_nan_s)
    print(f"IQR on all_nan: {err_iqr_nan}")

    constant_series = pd.Series([5.0] * 20)
    an_z_const, zs_const, err_z_const = detect_anomalies_zscore(constant_series)
    if err_z_const: print(f"Z-score on constant series error: {err_z_const}")
    else: print(f"Z-score on constant series anomalies: {an_z_const.sum()}, Z-scores NaN: {zs_const.isna().all()}")

    an_z_const_w, zs_const_w, err_z_const_w = detect_anomalies_zscore(constant_series, window=5)
    if err_z_const_w: print(f"Z-score (window) on constant series error: {err_z_const_w}")
    else: print(f"Z-score (window) on constant series anomalies: {an_z_const_w.sum()}, Z-scores NaN: {zs_const_w.isna().all()}")
