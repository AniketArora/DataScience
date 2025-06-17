import json
import streamlit as st
import numpy as np # For np.nan if needed for default values, though not used in current version

# Define keys for different settings groups
TS_SETTINGS_KEYS = {
    "id_cols": "time_series_specs.id_cols",
    "timestamp_col": "time_series_specs.timestamp_col",
    "value_cols": "time_series_specs.value_cols",
    "selected_id": "time_series_specs.selected_id",
    "selected_value_col_for_analysis": "time_series_specs.selected_value_col_for_analysis"
}

IFOREST_PARAMS_KEYS = {
    "contamination": "if_contam_general"
}
OCSVM_PARAMS_KEYS = {
    "nu": "ocsvm_nu_general",
    "kernel": "ocsvm_kernel_general",
    "gamma": "ocsvm_gamma_general"
}
KMEANS_PARAMS_KEYS = {
    "k_min_stats": "k_min_stats_general",
    "k_max_stats": "k_max_stats_general",
    "k_final": "kmeans_k_final_general",
    "scale_data": "scale_data_clustering_kmeans_general"
}
DBSCAN_PARAMS_KEYS = {
    "eps": "dbscan_eps_general",
    "min_samples": "dbscan_min_samples_general",
    "scale_data": "scale_data_clustering_dbscan_general"
}

FEATURE_ENG_PARAMS_KEYS = {
    "acf_lags": "fe_acf_lags_general",
    "rolling_windows": "fe_rolling_windows_general"
}

# Helper to get nested session_state values
def _get_nested_session_state(key_path_str, default=None):
    keys = key_path_str.split('.')
    val = st.session_state
    try:
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError, AttributeError): # Added AttributeError for safety
        return default

# Helper to set nested session_state values
def _set_nested_session_state(key_path_str, value):
    keys = key_path_str.split('.')
    obj = st.session_state
    try:
        for key in keys[:-1]:
            if key not in obj or not isinstance(obj.get(key), dict): # Use .get for safety
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value
        return True
    except (KeyError, TypeError, AttributeError): # Added AttributeError for safety
        return False


def gather_settings_for_save():
    """Gathers all relevant analysis settings from st.session_state."""
    settings = {
        "time_series_settings": {},
        "anomaly_detection_settings": {
            "IsolationForest": {},
            "OneClassSVM": {}
        },
        "clustering_settings": {
            "KMeans": {},
            "DBSCAN": {}
        },
        "feature_engineering_settings": {} # New section
    }

    # Time Series Settings
    for key, ss_key_path in TS_SETTINGS_KEYS.items():
        default_val = [] if "cols" in key else "None" # Default for multiselect/select
        if key == "selected_id" or key == "selected_value_col_for_analysis": # these might be None or a string
             default_val = st.session_state.time_series_specs.get(key, "None") # Keep current if exists, else None

        settings["time_series_settings"][key] = _get_nested_session_state(ss_key_path, default_val)

    # Anomaly Detection - Isolation Forest
    for key, widget_key in IFOREST_PARAMS_KEYS.items():
        settings["anomaly_detection_settings"]["IsolationForest"][key] = st.session_state.get(widget_key)

    # Anomaly Detection - One-Class SVM
    for key, widget_key in OCSVM_PARAMS_KEYS.items():
        settings["anomaly_detection_settings"]["OneClassSVM"][key] = st.session_state.get(widget_key)

    # Clustering - K-Means
    for key, widget_key in KMEANS_PARAMS_KEYS.items():
        settings["clustering_settings"]["KMeans"][key] = st.session_state.get(widget_key)

    # Clustering - DBSCAN
    for key, widget_key in DBSCAN_PARAMS_KEYS.items():
        settings["clustering_settings"]["DBSCAN"][key] = st.session_state.get(widget_key)

    # Feature Engineering Settings
    # Using sensible defaults if keys are not in session_state, though UI should ideally set them first.
    settings["feature_engineering_settings"]["acf_lags"] = st.session_state.get(
        FEATURE_ENG_PARAMS_KEYS["acf_lags"], [1, 5, 10]
    )
    settings["feature_engineering_settings"]["rolling_windows"] = st.session_state.get(
        FEATURE_ENG_PARAMS_KEYS["rolling_windows"], [1, 5, 10, 20]
    )

    return settings

def apply_loaded_settings_to_session_state(settings_dict):
    """Applies loaded settings to st.session_state, which widgets read from."""
    applied_keys_log = []
    errors_log = []
    try:
        # Time Series Settings
        ts_settings = settings_dict.get("time_series_settings", {})
        for key, ss_key_path in TS_SETTINGS_KEYS.items():
            if key in ts_settings:
                if _set_nested_session_state(ss_key_path, ts_settings[key]):
                    applied_keys_log.append(ss_key_path)
                else:
                    errors_log.append(f"Could not set {ss_key_path}")


        # Anomaly Detection - Isolation Forest
        if_settings = settings_dict.get("anomaly_detection_settings", {}).get("IsolationForest", {})
        for key, widget_key in IFOREST_PARAMS_KEYS.items():
            if key in if_settings and if_settings[key] is not None:
                st.session_state[widget_key] = if_settings[key]
                applied_keys_log.append(widget_key)

        # Anomaly Detection - One-Class SVM
        ocsvm_settings = settings_dict.get("anomaly_detection_settings", {}).get("OneClassSVM", {})
        for key, widget_key in OCSVM_PARAMS_KEYS.items():
            if key in ocsvm_settings and ocsvm_settings[key] is not None:
                st.session_state[widget_key] = ocsvm_settings[key]
                applied_keys_log.append(widget_key)

        # Clustering - K-Means
        kmeans_settings = settings_dict.get("clustering_settings", {}).get("KMeans", {})
        for key, widget_key in KMEANS_PARAMS_KEYS.items():
            if key in kmeans_settings and kmeans_settings[key] is not None:
                st.session_state[widget_key] = kmeans_settings[key]
                applied_keys_log.append(widget_key)

        # Clustering - DBSCAN
        dbscan_settings = settings_dict.get("clustering_settings", {}).get("DBSCAN", {})
        for key, widget_key in DBSCAN_PARAMS_KEYS.items():
            if key in dbscan_settings and dbscan_settings[key] is not None:
                st.session_state[widget_key] = dbscan_settings[key]
                applied_keys_log.append(widget_key)

        # Feature Engineering Settings
        fe_settings = settings_dict.get("feature_engineering_settings", {})
        for key, widget_key in FEATURE_ENG_PARAMS_KEYS.items():
            if key in fe_settings and fe_settings[key] is not None:
                # Ensure lists are lists (e.g. from JSON)
                if isinstance(fe_settings[key], list):
                    st.session_state[widget_key] = fe_settings[key]
                    applied_keys_log.append(widget_key)
                else:
                    errors_log.append(f"Invalid type for {widget_key}, expected list, got {type(fe_settings[key])}")


        if errors_log:
            return False, f"Errors applying settings for: {', '.join(errors_log)}. Applied: {', '.join(applied_keys_log)}"
        return True, f"Successfully applied settings for keys: {', '.join(applied_keys_log) if applied_keys_log else 'None'}."
    except Exception as e:
        return False, f"Error applying loaded settings: {e}"
