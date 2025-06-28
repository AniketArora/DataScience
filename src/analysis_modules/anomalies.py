import pandas as pd
import numpy as np # Ensure numpy is imported
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler # Good practice for One-Class SVM
import logging

logger = logging.getLogger(__name__)


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
        msg = "Input is not a pandas Series."
        logger.warning(msg)
        return None, None, msg
    if series.empty:
        msg = "Input series is empty."
        logger.warning(msg)
        return None, None, msg

    series_cleaned = series.dropna()
    if series_cleaned.empty:
        msg = "Series is empty after dropping NaN values."
        logger.warning(msg)
        return None, None, msg

    try:
        if window:
            if not isinstance(window, int) or window <= 0:
                msg = "Window must be a positive integer."
                logger.warning(msg)
                return None, None, msg
            if window >= len(series_cleaned):
                 msg = f"Window size ({window}) is too large for the series length ({len(series_cleaned)})."
                 logger.warning(msg)
                 return None, None, msg

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
        logger.error("Z-score anomaly detection failed for series %s: %s", series.name if series.name else "Unnamed", e, exc_info=True)
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
        msg = "Input is not a pandas Series."
        logger.warning(msg)
        return None, None, msg
    if series.empty:
        msg = "Input series is empty."
        logger.warning(msg)
        return None, None, msg

    series_cleaned = series.dropna()
    if series_cleaned.empty:
        msg = "Series is empty after dropping NaN values."
        logger.warning(msg)
        return None, None, msg

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
        logger.error("IQR anomaly detection failed for series %s: %s", series.name if series.name else "Unnamed", e, exc_info=True)
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
        msg = "Input is not a pandas DataFrame."
        logger.warning(msg)
        return None, None, msg
    if feature_df.empty:
        msg = "Input DataFrame is empty."
        logger.warning(msg)
        return None, None, msg
    if feature_df.isnull().any().any(): # Check for any NaNs in the entire DataFrame
        msg = "Input DataFrame contains NaN values. Please handle them before anomaly detection."
        logger.warning(msg)
        return None, None, msg

    try:
        model = IsolationForest(contamination=contamination, random_state=random_state, **kwargs)
        model.fit(feature_df)

        labels = pd.Series(model.predict(feature_df), index=feature_df.index, name="anomaly_label_iforest")
        scores = pd.Series(model.decision_function(feature_df), index=feature_df.index, name="anomaly_score_iforest")

        return labels, scores, None
    except Exception as e:
        logger.error("Isolation Forest anomaly detection failed: %s", e, exc_info=True)
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
        msg = "Input is not a pandas DataFrame."
        logger.warning(msg)
        return None, None, msg
    if feature_df.empty:
        msg = "Input DataFrame is empty."
        logger.warning(msg)
        return None, None, msg
    if feature_df.isnull().any().any():
        msg = "Input DataFrame contains NaN values. Please handle them before anomaly detection."
        logger.warning(msg)
        return None, None, msg

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
        logger.error("One-Class SVM anomaly detection failed: %s", e, exc_info=True)
        return None, None, f"One-Class SVM anomaly detection failed: {e}"

# --- Analysis Module Interface Implementation ---
from src.interfaces import AnalysisModuleInterface
from typing import Any, Dict, Tuple, Optional # For type hinting
import matplotlib.pyplot as plt


class AnomalyDetectionAnalysisModule(AnalysisModuleInterface):
    """
    Analysis module for detecting anomalous devices from a population
    based on their engineered feature fingerprints.
    """

    def get_name(self) -> str:
        return "Population Anomaly Detection"

    def get_description(self) -> str:
        return "Detects anomalous devices from a population based on their engineered feature fingerprints using unsupervised learning algorithms."

    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "selected_method": {
                "type": "selectbox",
                "options": ["Isolation Forest", "One-Class SVM"],
                "default": "Isolation Forest",
                "label": "Anomaly Detection Method",
                "help": "Choose the algorithm for detecting population-level anomalies."
            },
            # Isolation Forest specific
            "iforest_contamination": {
                "type": "slider",
                "default": 0.1,
                "min_value": 0.001, # Contamination cannot be 0
                "max_value": 0.5,
                "step": 0.01,
                "label": "Contamination (Isolation Forest)",
                "help": "The expected proportion of outliers in the data set. Adjust based on domain knowledge."
            },
            # One-Class SVM specific
            "ocsvm_nu": {
                "type": "slider",
                "default": 0.05,
                "min_value": 0.001, # Nu should be > 0
                "max_value": 0.5, # And typically <= 0.5
                "step": 0.01,
                "label": "Nu (One-Class SVM)",
                "help": "An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors."
            },
            "ocsvm_kernel": {
                "type": "selectbox",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "default": "rbf",
                "label": "Kernel (One-Class SVM)",
                "help": "Specifies the kernel type to be used in the algorithm."
            },
            "ocsvm_gamma": {
                "type": "text_input", # Using text_input to allow 'scale', 'auto' or float
                "default": "scale",
                "label": "Gamma (One-Class SVM)",
                "help": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Use 'scale', 'auto', or a specific float value."
            }
        }

    def render_parameters_ui(self, st_object: Any, current_values: Dict[str, Any], module_key: str) -> Dict[str, Any]:
        updated_values = {}
        param_defs = self.get_parameter_definitions()

        updated_values["selected_method"] = st_object.selectbox(
            param_defs["selected_method"]["label"],
            options=param_defs["selected_method"]["options"],
            index=param_defs["selected_method"]["options"].index(current_values.get("selected_method", param_defs["selected_method"]["default"])),
            help=param_defs["selected_method"]["help"],
            key=f"{module_key}_selected_method"
        )

        st_object.markdown("---")

        if updated_values["selected_method"] == "Isolation Forest":
            st_object.subheader("Isolation Forest Parameters")
            updated_values["iforest_contamination"] = st_object.slider(
                param_defs["iforest_contamination"]["label"],
                min_value=param_defs["iforest_contamination"]["min_value"],
                max_value=param_defs["iforest_contamination"]["max_value"],
                value=current_values.get("iforest_contamination", param_defs["iforest_contamination"]["default"]),
                step=param_defs["iforest_contamination"]["step"],
                help=param_defs["iforest_contamination"]["help"],
                key=f"{module_key}_iforest_contamination"
            )
        elif updated_values["selected_method"] == "One-Class SVM":
            st_object.subheader("One-Class SVM Parameters")
            updated_values["ocsvm_nu"] = st_object.slider(
                param_defs["ocsvm_nu"]["label"],
                min_value=param_defs["ocsvm_nu"]["min_value"],
                max_value=param_defs["ocsvm_nu"]["max_value"],
                value=current_values.get("ocsvm_nu", param_defs["ocsvm_nu"]["default"]),
                step=param_defs["ocsvm_nu"]["step"],
                help=param_defs["ocsvm_nu"]["help"],
                key=f"{module_key}_ocsvm_nu"
            )
            updated_values["ocsvm_kernel"] = st_object.selectbox(
                param_defs["ocsvm_kernel"]["label"],
                options=param_defs["ocsvm_kernel"]["options"],
                index=param_defs["ocsvm_kernel"]["options"].index(current_values.get("ocsvm_kernel", param_defs["ocsvm_kernel"]["default"])),
                help=param_defs["ocsvm_kernel"]["help"],
                key=f"{module_key}_ocsvm_kernel"
            )
            updated_values["ocsvm_gamma"] = st_object.text_input(
                param_defs["ocsvm_gamma"]["label"],
                value=str(current_values.get("ocsvm_gamma", param_defs["ocsvm_gamma"]["default"])), # Ensure value is str for text_input
                help=param_defs["ocsvm_gamma"]["help"],
                key=f"{module_key}_ocsvm_gamma"
            )

        # Ensure all params are in updated_values, even if not currently displayed
        for param_name, definition in param_defs.items():
            if param_name not in updated_values:
                updated_values[param_name] = current_values.get(param_name, definition["default"])

        # The run button is handled by main.py which calls run_analysis.
        # This UI rendering function only returns the selected parameters.
        # If a button were here, its state would be part of updated_values or handled via session_state by main.py
        # For example: if st_object.button("Run Anomaly Detection", key=f"{module_key}_run_button"):
        #    updated_values["action"] = "run_anomaly_detection"

        return updated_values

    def run_analysis(self, data_df: pd.DataFrame, params: Dict[str, Any], session_state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if data_df.empty:
            return None, "Input data (features_df_cleaned) is empty."
        if not all(dtype.kind in 'biufc' for dtype in data_df.dtypes): # Check if all columns are numeric
             return None, "All columns in input data must be numeric for these anomaly detection methods."
        if data_df.isnull().values.any(): # Check for any NaNs
             return None, "Input data contains NaN values. Please ensure data is cleaned (e.g., features_df_cleaned.dropna())."


        method = params.get("selected_method", "Isolation Forest")
        results_payload: Dict[str, Any] = {"method": method, "status": "pending"}
        error_msg: Optional[str] = None
        labels: Optional[pd.Series] = None
        scores: Optional[pd.Series] = None

        try:
            if method == "Isolation Forest":
                contamination = params.get("iforest_contamination", 0.1)
                labels, scores, error_msg = detect_anomalies_isolation_forest(data_df, contamination=contamination)
            elif method == "One-Class SVM":
                nu = params.get("ocsvm_nu", 0.05)
                kernel = params.get("ocsvm_kernel", "rbf")
                gamma_str = str(params.get("ocsvm_gamma", "scale"))
                try: # Attempt to convert gamma to float if it's a number, else keep as string ('scale', 'auto')
                    gamma_val = float(gamma_str)
                except ValueError:
                    gamma_val = gamma_str

                labels, scores, error_msg = detect_anomalies_one_class_svm(data_df, nu=nu, kernel=kernel, gamma=gamma_val)
            else:
                error_msg = f"Unknown anomaly detection method: {method}"

            if error_msg:
                return None, error_msg

            if labels is not None and scores is not None:
                 results_payload.update({"labels": labels, "scores": scores, "status": "detection_done"})
                 return results_payload, None
            else: # Should be caught by error_msg from detection functions, but as a safeguard
                return None, f"Anomaly detection for {method} failed to return labels or scores."

        except Exception as e:
            return None, f"An unexpected error occurred during {method} analysis: {e}"


    def render_results(self, st_object: Any, results: Optional[Dict[str, Any]], session_state: Dict[str, Any]) -> None:
        if results is None or results.get("status") != "detection_done":
            st_object.info("Anomaly detection not yet run or no results to display.")
            return

        method = results.get("method", "N/A")
        labels: Optional[pd.Series] = results.get("labels")
        scores: Optional[pd.Series] = results.get("scores")

        st_object.subheader(f"Results: {method}")

        if labels is not None and scores is not None:
            num_anomalies = (labels == -1).sum()
            st_object.write(f"Found **{num_anomalies}** potential anomalies out of {len(labels)} devices.")

            res_df = pd.DataFrame({'label': labels, 'score': scores}).sort_values(by='score')
            st_object.write("Top potentially anomalous devices (lower scores are more anomalous):")
            st_object.dataframe(res_df[res_df['label'] == -1].head())

            st_object.write("Top potentially normal devices (higher scores are less anomalous):")
            st_object.dataframe(res_df[res_df['label'] == 1].sort_values(by='score', ascending=False).head())


            # Optional: Bar chart of scores
            try:
                fig, ax = plt.subplots(figsize=(10, 4))
                # Plot a sample of scores if too many, e.g., 1000 max
                scores_to_plot = scores.sort_values()
                if len(scores_to_plot) > 1000:
                    scores_to_plot = scores_to_plot.iloc[np.linspace(0, len(scores_to_plot)-1, 1000, dtype=int)] # Sample

                scores_to_plot.plot(kind='bar', ax=ax)
                ax.set_xticks([]) # Hide device ID ticks for clarity if too many
                ax.set_ylabel("Anomaly Score")
                ax.set_title(f"Anomaly Scores ({method}) - Lower is more anomalous")
                st_object.pyplot(fig)
            except Exception as e:
                st_object.warning(f"Could not generate scores plot: {e}")
        else:
            st_object.info("No labels or scores found in results.")


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
