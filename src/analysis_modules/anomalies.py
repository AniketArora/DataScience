import pandas as pd
import numpy as np # Ensure numpy is imported
import streamlit as st # Not used in functions but often included
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler # Good practice for One-Class SVM


@st.cache_data
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

            rolling_mean = series_cleaned.rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = series_cleaned.rolling(window=window, center=True, min_periods=1).std()

            rolling_mean = rolling_mean.reindex(series_cleaned.index).bfill().ffill()
            rolling_std = rolling_std.reindex(series_cleaned.index).bfill().ffill()

            rolling_std[rolling_std == 0] = np.nan

            z_scores = (series_cleaned - rolling_mean) / rolling_std
        else:
            mean = series_cleaned.mean()
            std = series_cleaned.std()
            if std == 0:
                z_scores = pd.Series(np.nan, index=series_cleaned.index)
            else:
                z_scores = (series_cleaned - mean) / std

        anomalies = (z_scores.abs() > threshold)

        anomalies_reindexed = anomalies.reindex(series.index, fill_value=False)
        z_scores_reindexed = z_scores.reindex(series.index)

        return anomalies_reindexed, z_scores_reindexed, None
    except Exception as e:
        return None, None, f"Z-score anomaly detection failed: {e}"


@st.cache_data
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

        anomalies_reindexed = anomalies.reindex(series.index, fill_value=False)

        return anomalies_reindexed, bounds_info, None
    except Exception as e:
        return None, None, f"IQR anomaly detection failed: {e}"

# --- New ML-based Anomaly Detection Functions ---

@st.cache_data
def detect_anomalies_isolation_forest(feature_df: pd.DataFrame, contamination='auto', random_state=42, **kwargs):
    """
    Detects anomalies in a DataFrame of device features using Isolation Forest.

    Args:
        feature_df (pd.DataFrame): DataFrame where rows are devices and columns are features.
                                   Must not contain NaN values.
        contamination (float or 'auto'): The proportion of outliers in the data set.
                                        Passed to IsolationForest.
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional keyword arguments for IsolationForest.

    Returns:
        pd.Series: Anomaly labels (-1 for outliers, 1 for inliers) for each device.
        pd.Series: Anomaly scores for each device.
        str or None: An error message if detection fails.
    """
    if not isinstance(feature_df, pd.DataFrame):
        return None, None, "Input is not a pandas DataFrame."
    if feature_df.empty:
        return None, None, "Input DataFrame is empty."
    if feature_df.isnull().any().any(): # Check for any NaNs in the entire DataFrame
        return None, None, "Input DataFrame contains NaN values. Please handle them before anomaly detection."

    try:
        model = IsolationForest(contamination=contamination, random_state=random_state, **kwargs)
        model.fit(feature_df)

        labels = pd.Series(model.predict(feature_df), index=feature_df.index, name="anomaly_label_iforest")
        scores = pd.Series(model.decision_function(feature_df), index=feature_df.index, name="anomaly_score_iforest")

        return labels, scores, None
    except Exception as e:
        return None, None, f"Isolation Forest anomaly detection failed: {e}"

@st.cache_data
def detect_anomalies_one_class_svm(feature_df: pd.DataFrame, nu=0.05, kernel="rbf", gamma='scale', **kwargs):
    """
    Detects anomalies in a DataFrame of device features using One-Class SVM.
    It's recommended to scale features before using One-Class SVM with RBF kernel.

    Args:
        feature_df (pd.DataFrame): DataFrame where rows are devices and columns are features.
                                   Must not contain NaN values.
        nu (float): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        kernel (str): Specifies the kernel type to be used in the algorithm.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        **kwargs: Additional keyword arguments for OneClassSVM.

    Returns:
        pd.Series: Anomaly labels (-1 for outliers, 1 for inliers) for each device.
        pd.Series: Anomaly scores (signed distance to the separating hyperplane) for each device.
        str or None: An error message if detection fails.
    """
    if not isinstance(feature_df, pd.DataFrame):
        return None, None, "Input is not a pandas DataFrame."
    if feature_df.empty:
        return None, None, "Input DataFrame is empty."
    if feature_df.isnull().any().any():
        return None, None, "Input DataFrame contains NaN values. Please handle them before anomaly detection."

    try:
        # Scale features - important for SVM
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        scaled_feature_df = pd.DataFrame(scaled_features, index=feature_df.index, columns=feature_df.columns)

        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, **kwargs)
        model.fit(scaled_feature_df)

        labels = pd.Series(model.predict(scaled_feature_df), index=feature_df.index, name="anomaly_label_ocsvm")
        scores = pd.Series(model.decision_function(scaled_feature_df), index=feature_df.index, name="anomaly_score_ocsvm")

        return labels, scores, None
    except Exception as e:
        return None, None, f"One-Class SVM anomaly detection failed: {e}"


if __name__ == '__main__':
    # Example Usage for existing functions
    idx = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = [i % 20 + np.random.randn() for i in range(100)] # Corrected pd.np to np
    sample_series = pd.Series(data, index=idx)
    sample_series.iloc[10] = 50
    sample_series.iloc[20] = -30
    sample_series.iloc[[5, 15, 25]] = pd.NA

    print("--- Z-score Anomalies (Global) ---")
    anomalies_z, z_scores, error_z = detect_anomalies_zscore(sample_series, threshold=2.5)
    if error_z: print(f"Error: {error_z}")
    else: print("Anomalies detected at:", anomalies_z[anomalies_z].index.tolist())

    print("\n--- Z-score Anomalies (Rolling Window=10) ---")
    anomalies_z_roll, z_scores_roll, error_z_roll = detect_anomalies_zscore(sample_series, threshold=2.0, window=10)
    if error_z_roll: print(f"Error: {error_z_roll}")
    else: print("Anomalies detected at (rolling):", anomalies_z_roll[anomalies_z_roll].index.tolist())

    print("\n--- IQR Anomalies ---")
    anomalies_iqr, bounds, error_iqr = detect_anomalies_iqr(sample_series, multiplier=1.5)
    if error_iqr: print(f"Error: {error_iqr}")
    else:
        print("Anomalies detected at:", anomalies_iqr[anomalies_iqr].index.tolist())
        print("Bounds Info:\n", bounds)

    print("\n--- Error Handling Examples (Existing) ---")
    empty_s = pd.Series([], dtype=float)
    all_nan_s = pd.Series([pd.NA, pd.NA], dtype=float) # Using pd.NA which is fine

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

    # --- Example for ML-based Anomaly Detection ---
    print("\n--- ML-based Anomaly Detection Examples ---")
    rng = np.random.RandomState(42)
    n_samples = 50
    n_features = 5
    X_train = pd.DataFrame(rng.rand(n_samples, n_features), columns=[f'feature_{j}' for j in range(n_features)])
    X_outliers = pd.DataFrame(rng.uniform(low=-4, high=4, size=(10, n_features)), columns=X_train.columns)
    sample_feature_df = pd.concat([X_train, X_outliers], ignore_index=True)
    sample_feature_df.index = [f"device_{i}" for i in range(len(sample_feature_df))]

    print("\n--- Isolation Forest ---")
    if_labels, if_scores, if_error = detect_anomalies_isolation_forest(sample_feature_df, contamination=0.15)
    if if_error:
        print(f"Error: {if_error}")
    else:
        print("Labels (-1 is outlier):\n", if_labels[if_labels == -1].head())
        print("Scores (lower is more anomalous):\n", if_scores.sort_values().head())

    print("\n--- One-Class SVM ---")
    ocsvm_labels, ocsvm_scores, ocsvm_error = detect_anomalies_one_class_svm(sample_feature_df, nu=0.15)
    if ocsvm_error:
        print(f"Error: {ocsvm_error}")
    else:
        print("Labels (-1 is outlier):\n", ocsvm_labels[ocsvm_labels == -1].head())
        print("Scores (lower is more anomalous):\n", ocsvm_scores.sort_values().head())

    print("\n--- Error Handling for ML Anomaly Detectors ---")
    empty_df = pd.DataFrame()
    nan_df = pd.DataFrame({'A': [1, np.nan], 'B': [2,3]})

    _, _, err_if_empty = detect_anomalies_isolation_forest(empty_df)
    print(f"Isolation Forest on empty: {err_if_empty}")
    _, _, err_if_nan = detect_anomalies_isolation_forest(nan_df)
    print(f"Isolation Forest on NaN df: {err_if_nan}")

    _, _, err_ocsvm_empty = detect_anomalies_one_class_svm(empty_df)
    print(f"One-Class SVM on empty: {err_ocsvm_empty}")
    _, _, err_ocsvm_nan = detect_anomalies_one_class_svm(nan_df)
    print(f"One-Class SVM on NaN df: {err_ocsvm_nan}")
