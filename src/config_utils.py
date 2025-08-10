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
            sub_obj = obj.get(key)
            if sub_obj is None:
                sub_obj = {}
                obj[key] = sub_obj
            elif not isinstance(sub_obj, dict):
                return False # Path is blocked by a non-dict item
            obj = sub_obj
        obj[keys[-1]] = value
        return True
    except (KeyError, TypeError, AttributeError):
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

    # Clustering - K-Means (Old - to be removed or adapted)
    # for key, widget_key in KMEANS_PARAMS_KEYS.items():
    #     settings["clustering_settings"]["KMeans"][key] = st.session_state.get(widget_key)

    # Clustering - DBSCAN (Old - to be removed or adapted)
    # for key, widget_key in DBSCAN_PARAMS_KEYS.items():
    #     settings["clustering_settings"]["DBSCAN"][key] = st.session_state.get(widget_key)

    # New: Save clustering module parameters
    # These are stored under a single key in session_state, e.g., 'clustering_module_params'
    # The module_key used in main.py was "clustering_module" for its params.
    # So, st.session_state.clustering_module_params should exist.
    # For consistency with other settings, we might want to use a key like 'clustering_module_params_general'
    # if the actual key in session_state is 'clustering_module_params'.
    # Assuming st.session_state.clustering_module_params holds the dictionary from the module.

    # Let's simplify and assume the key in session_state is 'clustering_module_params'
    # and we save it directly.
    settings["clustering_module_params"] = st.session_state.get('clustering_module_params', {})
    # Clean up old clustering_settings if it's now empty or redundant
    if not settings["clustering_settings"]["KMeans"] and not settings["clustering_settings"]["DBSCAN"]:
        del settings["clustering_settings"]


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

        # Clustering - K-Means (Old - to be removed or adapted)
        # kmeans_settings = settings_dict.get("clustering_settings", {}).get("KMeans", {})
        # for key, widget_key in KMEANS_PARAMS_KEYS.items():
        #     if key in kmeans_settings and kmeans_settings[key] is not None:
        #         st.session_state[widget_key] = kmeans_settings[key]
        #         applied_keys_log.append(widget_key)

        # Clustering - DBSCAN (Old - to be removed or adapted)
        # dbscan_settings = settings_dict.get("clustering_settings", {}).get("DBSCAN", {})
        # for key, widget_key in DBSCAN_PARAMS_KEYS.items():
        #     if key in dbscan_settings and dbscan_settings[key] is not None:
        #         st.session_state[widget_key] = dbscan_settings[key]
        #         applied_keys_log.append(widget_key)

        # New: Load clustering module parameters
        # The key in the saved file is 'clustering_module_params'
        loaded_clustering_params = settings_dict.get("clustering_module_params")
        if loaded_clustering_params is not None: # Check if it exists in the loaded file
            # Ensure 'clustering_module_params' exists in session_state before updating
            if 'clustering_module_params' not in st.session_state:
                st.session_state.clustering_module_params = {} # Initialize if not present

            # Update existing keys, don't discard new keys added to the module since last save
            for key, value in loaded_clustering_params.items():
                if value is not None: # Avoid setting None for params that might have valid defaults
                    st.session_state.clustering_module_params[key] = value
            applied_keys_log.append('clustering_module_params')

        # Note: The old "clustering_settings" with KMeans/DBSCAN sub-keys will now be ignored
        # if they are present in an old config file, as we are not explicitly loading them.

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
