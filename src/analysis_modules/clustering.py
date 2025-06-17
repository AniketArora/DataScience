import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.interfaces import AnalysisModuleInterface
from typing import Any, Dict, Tuple, Optional, List


# --- Existing Clustering Functions (ensure they are here or imported) ---
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales numerical features using StandardScaler."""
    if df.empty:
        return df
    # Only scale numeric columns, keep others as is (though feature DFs are usually all numeric)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaled_values
        return df_scaled
    return df

def get_kmeans_elbow_silhouette_data(
    feature_df: pd.DataFrame, k_range: range, scale_data: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Calculates inertia and silhouette scores for a range of K in K-Means."""
    if feature_df.empty:
        return None, "Feature DataFrame is empty."
    if len(feature_df) < max(k_range, default=2): # Ensure enough samples for max K
         return None, f"Not enough samples ({len(feature_df)}) for max K value ({max(k_range, default=2)})."

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df_processed = scale_features(feature_df) if scale_data else feature_df.copy()

    results = []
    for k_val in k_range:
        if k_val > len(df_processed):
            # st.warning(f"K={k_val} is greater than number of samples. Skipping.")
            continue
        try:
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(df_processed)
            inertia = kmeans.inertia_
            if k_val > 1 and len(np.unique(labels)) > 1: # Silhouette score requires at least 2 labels
                silhouette = silhouette_score(df_processed, labels)
            else:
                silhouette = np.nan # Or None, or 0, depending on how you want to handle
            results.append({'K': k_val, 'Inertia': inertia, 'Silhouette Score': silhouette})
        except Exception as e:
            return None, f"Error during K-Means for K={k_val}: {e}"

    if not results:
        return None, "No K-Means statistics were calculated. Check K range and data."
    return pd.DataFrame(results), None

def perform_kmeans_clustering(
    feature_df: pd.DataFrame, n_clusters: int, scale_data: bool = True
) -> Tuple[Optional[pd.Series], Optional[Any], Optional[str]]:
    """Performs K-Means clustering."""
    if feature_df.empty:
        return None, None, "Feature DataFrame is empty."
    if len(feature_df) < n_clusters:
        return None, None, f"Number of samples ({len(feature_df)}) is less than n_clusters ({n_clusters})."
    if n_clusters <= 0:
        return None, None, "Number of clusters must be positive."

    from sklearn.cluster import KMeans

    df_processed = scale_features(feature_df) if scale_data else feature_df.copy()

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(df_processed)
        return pd.Series(labels, index=feature_df.index, name="cluster_labels"), kmeans, None
    except Exception as e:
        return None, None, f"Error during K-Means clustering: {e}"

def perform_dbscan_clustering(
    feature_df: pd.DataFrame, eps: float, min_samples: int, scale_data: bool = True
) -> Tuple[Optional[pd.Series], Optional[Any], Optional[str]]:
    """Performs DBSCAN clustering."""
    if feature_df.empty:
        return None, None, "Feature DataFrame is empty."
    if eps <= 0:
        return None, None, "DBSCAN eps must be positive."
    if min_samples <= 0:
        return None, None, "DBSCAN min_samples must be positive."

    from sklearn.cluster import DBSCAN

    df_processed = scale_features(feature_df) if scale_data else feature_df.copy()

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(df_processed)
        return pd.Series(labels, index=feature_df.index, name="cluster_labels"), dbscan, None
    except Exception as e:
        return None, None, f"Error during DBSCAN clustering: {e}"

# --- Analysis Module Implementation ---

class ClusteringAnalysisModule(AnalysisModuleInterface):
    """
    Analysis module for performing device behavior clustering.
    Supports K-Means and DBSCAN algorithms.
    """

    def get_name(self) -> str:
        return "Device Behavior Clustering"

    def get_description(self) -> str:
        return "Groups devices based on their feature fingerprints using clustering algorithms (K-Means, DBSCAN)."

    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            "selected_method": {
                "type": "selectbox",
                "options": ["K-Means", "DBSCAN"],
                "default": "K-Means",
                "label": "Clustering Method",
                "help": "Choose the clustering algorithm to use."
            },
            # K-Means specific
            "kmeans_scale_data": {
                "type": "checkbox",
                "default": True,
                "label": "Scale data before K-Means",
                "help": "Apply StandardScaler to features before K-Means."
            },
            "kmeans_k_min": {
                "type": "number_input",
                "default": 2,
                "min_value": 2,
                "label": "Min K (for Elbow/Silhouette)",
                "help": "Minimum number of clusters for statistical evaluation (Elbow method, Silhouette score)."
            },
            "kmeans_k_max": {
                "type": "number_input",
                "default": 8,
                "min_value": 2, # Should be >= k_min ideally, handle in UI logic or validation
                "label": "Max K (for Elbow/Silhouette)",
                "help": "Maximum number of clusters for statistical evaluation."
            },
            "kmeans_k_final": {
                "type": "number_input",
                "default": 3,
                "min_value": 1, # K=1 can be valid if you want to see "all data" as one group stats
                "label": "Number of Clusters (K)",
                "help": "The final number of clusters (K) to use for K-Means."
            },
            # DBSCAN specific
            "dbscan_scale_data": {
                "type": "checkbox",
                "default": True,
                "label": "Scale data before DBSCAN",
                "help": "Apply StandardScaler to features before DBSCAN."
            },
            "dbscan_eps": {
                "type": "number_input",
                "default": 0.5,
                "min_value": 0.001, # eps must be > 0
                "format": "%.3f",
                "label": "Epsilon (DBSCAN)",
                "help": "The maximum distance between two samples for one to be considered as in the neighborhood of the other."
            },
            "dbscan_min_samples": {
                "type": "number_input",
                "default": 5,
                "min_value": 1,
                "label": "Min Samples (DBSCAN)",
                "help": "The number of samples in a neighborhood for a point to be considered as a core point."
            },
        }

    def render_parameters_ui(self, st_object: Any, current_values: Dict[str, Any], module_key: str) -> Dict[str, Any]:
        updated_values = {}
        param_defs = self.get_parameter_definitions()

        # General method selection
        updated_values["selected_method"] = st_object.selectbox(
            param_defs["selected_method"]["label"],
            options=param_defs["selected_method"]["options"],
            index=param_defs["selected_method"]["options"].index(current_values.get("selected_method", param_defs["selected_method"]["default"])),
            help=param_defs["selected_method"]["help"],
            key=f"{module_key}_selected_method"
        )

        st_object.markdown("---") # Visual separator

        if updated_values["selected_method"] == "K-Means":
            st_object.subheader("K-Means Parameters")
            updated_values["kmeans_scale_data"] = st_object.checkbox(
                param_defs["kmeans_scale_data"]["label"],
                value=current_values.get("kmeans_scale_data", param_defs["kmeans_scale_data"]["default"]),
                help=param_defs["kmeans_scale_data"]["help"],
                key=f"{module_key}_kmeans_scale_data"
            )

            k_min_val = current_values.get("kmeans_k_min", param_defs["kmeans_k_min"]["default"])
            updated_values["kmeans_k_min"] = st_object.number_input(
                param_defs["kmeans_k_min"]["label"],
                min_value=param_defs["kmeans_k_min"]["min_value"],
                value=k_min_val,
                help=param_defs["kmeans_k_min"]["help"],
                key=f"{module_key}_kmeans_k_min"
            )

            # Ensure k_max is not less than k_min
            k_max_default = param_defs["kmeans_k_max"]["default"]
            if k_max_default < updated_values["kmeans_k_min"]:
                k_max_default = updated_values["kmeans_k_min"]
            k_max_val = current_values.get("kmeans_k_max", k_max_default)
            if k_max_val < updated_values["kmeans_k_min"]: # Adjust if current value is too low due to k_min change
                k_max_val = updated_values["kmeans_k_min"]

            updated_values["kmeans_k_max"] = st_object.number_input(
                param_defs["kmeans_k_max"]["label"],
                min_value=updated_values["kmeans_k_min"], # Dynamic min_value based on k_min
                value=k_max_val,
                help=param_defs["kmeans_k_max"]["help"],
                key=f"{module_key}_kmeans_k_max"
            )

            # Button to trigger K-Means stats calculation
            # The action of this button needs to be handled by the calling code (main.py)
            # by checking its value in st.session_state or passing a specific flag.
            # For this interface, we assume the click results in a parameter change that run_analysis can check.
            if st_object.button("Calculate K-Means Statistics (Elbow/Silhouette)", key=f"{module_key}_calc_k_stats_button"):
                updated_values["action"] = "calculate_k_stats" # Flag for run_analysis

            updated_values["kmeans_k_final"] = st_object.number_input(
                param_defs["kmeans_k_final"]["label"],
                min_value=param_defs["kmeans_k_final"]["min_value"],
                value=current_values.get("kmeans_k_final", param_defs["kmeans_k_final"]["default"]),
                help=param_defs["kmeans_k_final"]["help"],
                key=f"{module_key}_kmeans_k_final"
            )

        elif updated_values["selected_method"] == "DBSCAN":
            st_object.subheader("DBSCAN Parameters")
            updated_values["dbscan_scale_data"] = st_object.checkbox(
                param_defs["dbscan_scale_data"]["label"],
                value=current_values.get("dbscan_scale_data", param_defs["dbscan_scale_data"]["default"]),
                help=param_defs["dbscan_scale_data"]["help"],
                key=f"{module_key}_dbscan_scale_data"
            )
            updated_values["dbscan_eps"] = st_object.number_input(
                param_defs["dbscan_eps"]["label"],
                min_value=param_defs["dbscan_eps"]["min_value"],
                value=current_values.get("dbscan_eps", param_defs["dbscan_eps"]["default"]),
                format=param_defs["dbscan_eps"]["format"],
                help=param_defs["dbscan_eps"]["help"],
                key=f"{module_key}_dbscan_eps"
            )
            updated_values["dbscan_min_samples"] = st_object.number_input(
                param_defs["dbscan_min_samples"]["label"],
                min_value=param_defs["dbscan_min_samples"]["min_value"],
                value=current_values.get("dbscan_min_samples", param_defs["dbscan_min_samples"]["default"]),
                help=param_defs["dbscan_min_samples"]["help"],
                key=f"{module_key}_dbscan_min_samples"
            )

        # Ensure all params are in updated_values, even if not currently displayed,
        # to maintain consistent structure, using their current or default values.
        for param_name, definition in param_defs.items():
            if param_name not in updated_values:
                updated_values[param_name] = current_values.get(param_name, definition["default"])

        return updated_values

    def run_analysis(self, data_df: pd.DataFrame, params: Dict[str, Any], session_state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if data_df.empty:
            return None, "Input data is empty."

        # Drop rows with any NaN/inf values from features_df as clustering algorithms typically can't handle them.
        # This should ideally be done before passing to the module, but as a safeguard:
        data_df_cleaned = data_df.replace([np.inf, -np.inf], np.nan).dropna()
        if data_df_cleaned.empty:
            return None, "Data became empty after removing NaN/inf values."


        method = params.get("selected_method", "K-Means")
        results_payload: Dict[str, Any] = {"method": method, "status": "pending"}
        error_msg: Optional[str] = None

        if method == "K-Means":
            if params.get("action") == "calculate_k_stats":
                k_min = params.get("kmeans_k_min", 2)
                k_max = params.get("kmeans_k_max", 8)
                if k_max < k_min: # Basic validation
                    return None, "Max K cannot be less than Min K for K-Means stats."

                scale = params.get("kmeans_scale_data", True)
                stats_df, error_msg_stats = get_kmeans_elbow_silhouette_data(data_df_cleaned, range(k_min, k_max + 1), scale_data=scale)
                if error_msg_stats:
                    error_msg = f"Error calculating K-Means stats: {error_msg_stats}"
                else:
                    results_payload["kmeans_stats_df"] = stats_df
                    results_payload["status"] = "k_stats_calculated"
            else: # Run actual K-Means clustering
                k_final = params.get("kmeans_k_final", 3)
                scale = params.get("kmeans_scale_data", True)
                if len(data_df_cleaned) < k_final :
                     return None, f"Number of samples ({len(data_df_cleaned)}) is less than k_final ({k_final})."

                labels, model, error_msg_run = perform_kmeans_clustering(data_df_cleaned, n_clusters=k_final, scale_data=scale)
                if error_msg_run:
                    error_msg = f"Error during K-Means: {error_msg_run}"
                else:
                    results_payload.update({"labels": labels, "model": model, "k": k_final, "status": "clustering_done"})

        elif method == "DBSCAN":
            eps = params.get("dbscan_eps", 0.5)
            min_samples = params.get("dbscan_min_samples", 5)
            scale = params.get("dbscan_scale_data", True)

            labels, model, error_msg_run = perform_dbscan_clustering(data_df_cleaned, eps=eps, min_samples=min_samples, scale_data=scale)
            if error_msg_run:
                error_msg = f"Error during DBSCAN: {error_msg_run}"
            else:
                results_payload.update({"labels": labels, "model": model, "status": "clustering_done"})

        else:
            error_msg = f"Unknown clustering method: {method}"

        if error_msg:
            return None, error_msg

        return results_payload, None


    def render_results(self, st_object: Any, results: Optional[Dict[str, Any]], session_state: Dict[str, Any]) -> None:
        if results is None:
            st_object.info("No clustering results to display.")
            return

        status = results.get("status")
        method = results.get("method", "N/A")

        if status == "k_stats_calculated":
            st_object.subheader(f"K-Means Statistics (Elbow/Silhouette)")
            stats_df = results.get("kmeans_stats_df")
            if stats_df is not None and not stats_df.empty:
                try:
                    st_object.write("Inertia (Elbow Method):")
                    st_object.line_chart(stats_df.set_index('K')['Inertia'])
                    st_object.write("Silhouette Score:")
                    st_object.line_chart(stats_df.set_index('K')['Silhouette Score'].dropna())
                except Exception as e:
                    st_object.error(f"Error rendering K-Means stats charts: {e}")
            else:
                st_object.info("No K-Means statistics data available to display.")

        elif status == "clustering_done":
            st_object.subheader(f"Clustering Results: {method}")
            labels: Optional[pd.Series] = results.get("labels")
            model: Any = results.get("model")

            if labels is not None:
                st_object.write("Cluster Summary (Device Counts):")
                summary_df = labels.value_counts().rename("device_count").to_frame()
                summary_df.index.name = "cluster_id"
                st_object.dataframe(summary_df)

                if method == "K-Means" and model is not None:
                    try:
                        # Accessing feature names from session_state if features_df_cleaned was stored,
                        # or assuming they are part of the model if available.
                        # For this example, we'll assume data_df (passed to run_analysis) had original column names.
                        # This part might need adjustment based on how `main.py` manages data.
                        # If `session_state['features_df_cleaned_columns']` was available:
                        # feature_names = session_state.get('features_df_cleaned_columns')
                        # A more robust way is to ensure run_analysis also returns feature names if scaling happened.
                        # For now, this is a placeholder.
                        if hasattr(model, "feature_names_in_"):
                             feature_names = model.feature_names_in_
                        elif hasattr(model, "n_features_in_"):
                             feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
                        else: # Fallback, less ideal
                            # This assumes the 'data_df' passed to run_analysis had the correct columns
                            # and scaling (if any) didn't change their order for cluster_centers_.
                            # It's better if the module that did scaling provides column names.
                            # For now, we'll try to get it from session_state as a placeholder concept.
                            feature_names = session_state.get("last_clustering_feature_columns", # This key would be set by main.py
                                                              [f"feature_{i}" for i in range(model.cluster_centers_.shape[1])])


                        st_object.write("Cluster Centers (Mean Feature Values):")
                        centers_df = pd.DataFrame(model.cluster_centers_, columns=feature_names)
                        st_object.dataframe(centers_df)
                    except Exception as e:
                        st_object.warning(f"Could not display K-Means cluster centers: {e}. Feature names might be missing or mismatched.")
            else:
                st_object.info("No clustering labels found in results.")

        elif status == "pending":
            st_object.info("Clustering analysis is pending or parameters not yet run.")

        else:
            st_object.warning(f"Unknown results status: {status}")

# --- Comments for main.py integration ---
# In main.py:
# 1. Initialize in st.session_state:
#    if 'clustering_module_params' not in st.session_state:
#        st.session_state.clustering_module_params = ClusteringAnalysisModule().get_parameter_definitions() # Or just use defaults on first render
#        # More accurately, just let the module handle its defaults internally on first call to render_parameters_ui
#        st.session_state.clustering_module_params = {"selected_method": "K-Means"} # minimal init
#    if 'clustering_module_results' not in st.session_state:
#        st.session_state.clustering_module_results = None
#    if 'clustering_module_k_stats_df' not in st.session_state: # Specific for K-Means stats
#        st.session_state.clustering_module_k_stats_df = None

# 2. Instantiate the module:
#    clustering_module = ClusteringAnalysisModule()
#    module_key = "clustering_analysis" # Unique key for this module instance

# 3. Render parameters UI (e.g., in sidebar or an expander):
#    with st.sidebar.expander(clustering_module.get_name(), expanded=True):
#        current_params = st.session_state.get(f"{module_key}_params", clustering_module.get_parameter_definitions())
#        # Extract default values from definitions for the first run
#        current_params_with_defaults = {k: current_params.get(k, v['default']) for k,v in clustering_module.get_parameter_definitions().items()}
#
#        updated_params = clustering_module.render_parameters_ui(st, current_params_with_defaults, module_key)
#        st.session_state[f"{module_key}_params"] = updated_params
#
#        # Button to run the main analysis (clustering)
#        if st.button("Run Clustering", key=f"{module_key}_run_button"):
#            st.session_state[f"{module_key}_params"]["action"] = "run_clustering" # Signal to run clustering
#            # In a real app, you'd trigger the run here or manage a queue

# 4. Call run_analysis (typically after a button click in main.py):
#    if st.session_state.get(f"{module_key}_params",{}).get("action"): # if an action was set
#        params_to_run = st.session_state[f"{module_key}_params"]
#        features_df_cleaned = st.session_state.get('all_device_features_df', pd.DataFrame()).dropna() # Get data
#
#        if not features_df_cleaned.empty:
#            # Store columns for render_results if scaling happens and feature names are lost
#            st.session_state["last_clustering_feature_columns"] = features_df_cleaned.columns.tolist()
#
#            results, error = clustering_module.run_analysis(features_df_cleaned, params_to_run, st.session_state)
#            if error:
#                st.error(error)
#                st.session_state.clustering_module_results = None
#            else:
#                st.session_state.clustering_module_results = results
#                if results.get("status") == "k_stats_calculated":
#                     st.session_state.clustering_module_k_stats_df = results.get("kmeans_stats_df") # Store separately if needed elsewhere
#        else:
#            st.warning("No features data available for clustering.")
#        st.session_state[f"{module_key}_params"]["action"] = None # Reset action
#        st.experimental_rerun() # To update UI based on new results/state


# 5. Render results (in the main area of the app):
#    results_to_render = st.session_state.get('clustering_module_results')
#    # If K-stats were calculated and stored separately, pass them to render_results or let it access from session_state
#    # For simplicity, current render_results expects k_stats_df within the main 'results' dict.
#    # So, if k_stats were calculated, results_to_render would be like:
#    # {"status": "k_stats_calculated", "kmeans_stats_df": st.session_state.clustering_module_k_stats_df, "method": "K-Means"}
#
#    # This part of the state management needs careful handling in main.py:
#    # if st.session_state.clustering_module_k_stats_df is not None and \
#    #    (results_to_render is None or results_to_render.get("status") != "clustering_done"):
#    #     # Construct a temporary result dict for rendering k-stats if actual clustering hasn't run yet
#    #     temp_k_stats_results = {
#    #         "status": "k_stats_calculated",
#    #         "kmeans_stats_df": st.session_state.clustering_module_k_stats_df,
#    #         "method": "K-Means" # Assume K-Means if stats are present
#    #     }
#    #     clustering_module.render_results(st, temp_k_stats_results, st.session_state)
#    # elif results_to_render and results_to_render.get("status") == "clustering_done":
#    #     clustering_module.render_results(st, results_to_render, st.session_state)
#    # else:
#    #     st.info("Run clustering analysis to see results.") # Or render pending status

```
