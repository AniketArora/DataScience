import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import logging
# For k-need, if we want to use it for Elbow or DBSCAN eps automatically
# from kneed import KneeLocator # Requires 'kneed' package
# For now, will implement manual elbow/silhouette plotting guidance

logger = logging.getLogger(__name__)

def scale_features(feature_df: pd.DataFrame):
    """Scales features using StandardScaler."""
    if not isinstance(feature_df, pd.DataFrame) or feature_df.empty:
        logger.warning("scale_features: Input is not a valid DataFrame or is empty.")
        return feature_df, None

    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(feature_df)
        scaled_feature_df = pd.DataFrame(scaled_features, index=feature_df.index, columns=feature_df.columns)
        return scaled_feature_df, scaler
    except Exception as e:
        logger.error("scale_features: Error during scaling: %s", e, exc_info=True)
        return feature_df, None

@st.cache_data
def perform_kmeans_clustering(feature_df: pd.DataFrame, n_clusters, random_state=42, scale_data=True, **kwargs):
    """
    Performs K-Means clustering on the feature DataFrame.

    Args:
        feature_df (pd.DataFrame): DataFrame where rows are devices and columns are features.
                                   Should not contain NaN values.
        n_clusters (int): The number of clusters to form.
        random_state (int): Random seed for reproducibility.
        scale_data (bool): Whether to scale data before clustering.
        **kwargs: Additional keyword arguments for KMeans.

    Returns:
        pd.Series: Cluster labels for each device.
        sklearn.cluster.KMeans: The fitted KMeans model instance.
        str or None: An error message if clustering fails.
    """
    if not isinstance(feature_df, pd.DataFrame):
        msg = "Input is not a pandas DataFrame."
        logger.warning("perform_kmeans_clustering: %s", msg)
        return None, None, msg
    if feature_df.empty:
        msg = "Input DataFrame is empty."
        logger.warning("perform_kmeans_clustering: %s", msg)
        return None, None, msg
    if feature_df.isnull().any().any():
        msg = "Input DataFrame contains NaN values. Please handle them."
        logger.warning("perform_kmeans_clustering: %s", msg)
        return None, None, msg
    if n_clusters <= 0:
        msg = "Number of clusters (n_clusters) must be positive."
        logger.warning("perform_kmeans_clustering: %s", msg)
        return None, None, msg
    if n_clusters > len(feature_df):
        msg = f"Number of clusters ({n_clusters}) cannot exceed number of samples ({len(feature_df)})."
        logger.warning("perform_kmeans_clustering: %s", msg)
        return None, None, msg


    df_to_cluster = feature_df
    if scale_data:
        df_to_cluster, _ = scale_features(feature_df) # Scaler object not used here but good practice

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', **kwargs)
        kmeans.fit(df_to_cluster)
        labels = pd.Series(kmeans.labels_, index=df_to_cluster.index, name="kmeans_cluster_labels")
        return labels, kmeans, None
    except Exception as e:
        logger.error("K-Means clustering failed: %s", e, exc_info=True)
        return None, None, f"K-Means clustering failed: {e}"

@st.cache_data
def get_kmeans_elbow_silhouette_data(feature_df: pd.DataFrame, k_range=range(2, 11), scale_data=True, random_state=42):
    """
    Calculates inertia (WCSS) and silhouette scores for a range of K values for K-Means.

    Args:
        feature_df (pd.DataFrame): DataFrame of features.
        k_range (range): Range of K values to test (e.g., range(2, 11)).
        scale_data (bool): Whether to scale data before calculations.
        random_state (int): Random seed.

    Returns:
        pd.DataFrame: DataFrame with K, Inertia, and Silhouette Score.
        str or None: Error message.
    """
    if not isinstance(feature_df, pd.DataFrame):
        msg = "Input is not a pandas DataFrame."
        logger.warning("get_kmeans_elbow_silhouette_data: %s", msg)
        return None, msg
    if feature_df.empty:
        msg = "Input DataFrame is empty."
        logger.warning("get_kmeans_elbow_silhouette_data: %s", msg)
        return None, msg
    if feature_df.isnull().any().any():
        msg = "Input DataFrame contains NaN values."
        logger.warning("get_kmeans_elbow_silhouette_data: %s", msg)
        return None, msg

    min_k = min(k_range) if len(k_range) > 0 else 0
    max_k = max(k_range) if len(k_range) > 0 else 0

    if not k_range or min_k < 2 :
        msg = "k_range must start from at least 2."
        logger.warning("get_kmeans_elbow_silhouette_data: %s", msg)
        return None, msg
    if max_k >= len(feature_df):
        msg = f"Max K in k_range ({max_k}) must be less than number of samples ({len(feature_df)})."
        logger.warning("get_kmeans_elbow_silhouette_data: %s", msg)
        return None, msg


    df_to_process = feature_df
    if scale_data:
        df_to_process, _ = scale_features(feature_df)

    inertia_values = []
    silhouette_values = []

    try:
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            kmeans.fit(df_to_process)
            inertia_values.append(kmeans.inertia_)
            # Silhouette score requires at least 2 labels, and all labels shouldn't be the same
            if k > 1 and len(np.unique(kmeans.labels_)) > 1 and len(np.unique(kmeans.labels_)) < len(df_to_process):
                 score = silhouette_score(df_to_process, kmeans.labels_, random_state=random_state)
                 silhouette_values.append(score)
            else: # Cannot compute silhouette score
                 silhouette_values.append(np.nan)

        results_df = pd.DataFrame({'K': list(k_range), 'Inertia': inertia_values, 'Silhouette Score': silhouette_values})
        return results_df, None
    except Exception as e:
        logger.error("Error calculating K-Means stats: %s", e, exc_info=True)
        return None, f"Error calculating K-Means stats: {e}"

@st.cache_data
def perform_dbscan_clustering(feature_df: pd.DataFrame, eps=0.5, min_samples=5, scale_data=True, **kwargs):
    """
    Performs DBSCAN clustering on the feature DataFrame.

    Args:
        feature_df (pd.DataFrame): DataFrame where rows are devices and columns are features.
                                   Should not contain NaN values.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        scale_data (bool): Whether to scale data before clustering.
        **kwargs: Additional keyword arguments for DBSCAN.

    Returns:
        pd.Series: Cluster labels for each device (-1 for noise points).
        sklearn.cluster.DBSCAN: The fitted DBSCAN model instance.
        str or None: An error message if clustering fails.
    """
    if not isinstance(feature_df, pd.DataFrame):
        msg = "Input is not a pandas DataFrame."
        logger.warning("perform_dbscan_clustering: %s", msg)
        return None, None, msg
    if feature_df.empty:
        msg = "Input DataFrame is empty."
        logger.warning("perform_dbscan_clustering: %s", msg)
        return None, None, msg
    if feature_df.isnull().any().any():
        msg = "Input DataFrame contains NaN values. Please handle them."
        logger.warning("perform_dbscan_clustering: %s", msg)
        return None, None, msg

    df_to_cluster = feature_df
    if scale_data:
        df_to_cluster, _ = scale_features(feature_df)

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        dbscan.fit(df_to_cluster)
        labels = pd.Series(dbscan.labels_, index=df_to_cluster.index, name="dbscan_cluster_labels")
        return labels, dbscan, None
    except Exception as e:
        logger.error("DBSCAN clustering failed: %s", e, exc_info=True)
        return None, None, f"DBSCAN clustering failed: {e}"

if __name__ == '__main__':
    # Example Usage
    rng = np.random.RandomState(42)
    n_samples = 100
    n_features = 5
    sample_features = pd.DataFrame(rng.rand(n_samples, n_features), columns=[f'feature_{j}' for j in range(n_features)])
    sample_features.index = [f"device_{i}" for i in range(n_samples)]

    # Add some structure for clustering
    sample_features.iloc[0:30, 0:2] += 2 # Group 1
    sample_features.iloc[30:60, 0:2] -= 2 # Group 2
    # Group 3 is the rest

    print("--- K-Means Elbow/Silhouette Data ---")
    # Adjust k_range if n_samples is small to avoid k >= n_samples
    max_k_for_test = min(8, n_samples -1) if n_samples > 2 else 0
    if max_k_for_test > 1:
        k_stats_df, k_stats_error = get_kmeans_elbow_silhouette_data(sample_features, k_range=range(2,max_k_for_test))
        if k_stats_error:
            print(f"Error: {k_stats_error}")
        else:
            print(k_stats_df)
    else:
        print("Not enough samples for k-means stats range test.")


    print("\n--- K-Means Clustering (k=3) ---")
    if n_samples >= 3:
        kmeans_labels, kmeans_model, kmeans_error = perform_kmeans_clustering(sample_features, n_clusters=3)
        if kmeans_error:
            print(f"Error: {kmeans_error}")
        else:
            print("Cluster counts:\n", kmeans_labels.value_counts())
            # print("Cluster centers:\n", kmeans_model.cluster_centers_)
    else:
        print("Not enough samples for k=3 K-Means test.")


    print("\n--- DBSCAN Clustering ---")
    # DBSCAN parameters (eps, min_samples) are data-dependent.
    # Using StandardScaler output for DBSCAN is common.
    scaled_sample_features, _ = scale_features(sample_features)
    if scaled_sample_features is not None and not scaled_sample_features.empty:
        dbscan_labels, dbscan_model, dbscan_error = perform_dbscan_clustering(scaled_sample_features, eps=1.0, min_samples=5, scale_data=False) # Data already scaled
        if dbscan_error:
            print(f"Error: {dbscan_error}")
        else:
            print("Cluster counts (DBSCAN, -1 is noise):\n", dbscan_labels.value_counts())
    else:
        print("Scaled features are empty or None, skipping DBSCAN.")


    print("\n--- Error Handling for Clustering ---")
    empty_df = pd.DataFrame()
    nan_df = pd.DataFrame({'A': [1, np.nan], 'B': [2,3]})

    _, _, err_km_empty = perform_kmeans_clustering(empty_df, n_clusters=2)
    print(f"K-Means on empty: {err_km_empty}")
    # For nan_df, after dropna (if not handled by caller), it might be very small or empty
    # K-Means requires n_samples > n_clusters. If nan_df becomes too small, it will error.
    # The function now checks for n_clusters > len(feature_df)
    _, _, err_km_nan = perform_kmeans_clustering(nan_df, n_clusters=1)
    print(f"K-Means on NaN df: {err_km_nan}") # Expected to fail due to NaNs or size

    _, _, err_db_empty = perform_dbscan_clustering(empty_df)
    print(f"DBSCAN on empty: {err_db_empty}")
    _, _, err_db_nan = perform_dbscan_clustering(nan_df)
    print(f"DBSCAN on NaN df: {err_db_nan}")


from src.interfaces import AnalysisModuleInterface # Ensure this import is present
from typing import Any, Dict, Tuple, Optional # For type hinting

class ClusteringAnalysisModule(AnalysisModuleInterface):
    """
    Analysis module for performing device behavior clustering.
    """

    def get_name(self) -> str:
        return "Device Behavior Clustering"

    def get_description(self) -> str:
        return "Performs K-Means or DBSCAN clustering on device features to identify groups of similar behavior."

    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        # These would be parameters for K-Means and DBSCAN, selected via UI
        return {
            "method": {"type": "selectbox", "default": "K-Means", "label": "Clustering Method", "options": ["K-Means", "DBSCAN"], "help": "Select the clustering algorithm."},
            "n_clusters": {"type": "int", "default": 3, "label": "Number of Clusters (K-Means)", "min_value": 2, "help": "Number of clusters for K-Means."},
            "scale_data_kmeans": {"type": "checkbox", "default": True, "label": "Scale Data (K-Means)", "help": "Scale data before K-Means clustering."},
            "eps": {"type": "float", "default": 0.5, "label": "Epsilon (DBSCAN)", "min_value": 0.01, "format": "%.2f", "help": "DBSCAN eps parameter."},
            "min_samples": {"type": "int", "default": 5, "label": "Min Samples (DBSCAN)", "min_value": 1, "help": "DBSCAN min_samples parameter."},
            "scale_data_dbscan": {"type": "checkbox", "default": True, "label": "Scale Data (DBSCAN)", "help": "Scale data before DBSCAN clustering."}
            # Potentially add k_range for elbow plot, random_state etc.
        }

    def render_parameters_ui(self, st_object: Any, current_values: Dict[str, Any], module_key: str) -> Dict[str, Any]:
        """
        Renders the UI for selecting clustering parameters.

        Args:
            st_object (Any): The Streamlit object (e.g., st or st.sidebar) to render the UI on.
            current_values (Dict[str, Any]): A dictionary of current parameter values.
            module_key (str): A unique key for the module to prevent widget state issues.

        Returns:
            Dict[str, Any]: A dictionary of updated parameter values.
        """
        # In a real scenario, this would render Streamlit widgets
        # For now, just return current_values or defaults to satisfy interface
        # This part is heavily UI dependent and would be built out in main.py or similar
        st_object.warning(f"{self.get_name()}: UI rendering for parameters is not fully implemented in this module stub.")

        updated_values = {}
        defs = self.get_parameter_definitions()

        # Example of how one might render:
        updated_values["method"] = st_object.selectbox(
            defs["method"]["label"],
            options=defs["method"]["options"],
            index=defs["method"]["options"].index(current_values.get("method", defs["method"]["default"])),
            key=f"{module_key}_method",
            help=defs["method"]["help"]
        )

        if updated_values["method"] == "K-Means":
            updated_values["n_clusters"] = st_object.number_input(
                defs["n_clusters"]["label"],
                min_value=defs["n_clusters"]["min_value"],
                value=current_values.get("n_clusters", defs["n_clusters"]["default"]),
                key=f"{module_key}_n_clusters",
                help=defs["n_clusters"]["help"]
            )
            updated_values["scale_data_kmeans"] = st_object.checkbox(
                defs["scale_data_kmeans"]["label"],
                value=current_values.get("scale_data_kmeans", defs["scale_data_kmeans"]["default"]),
                key=f"{module_key}_scale_data_kmeans",
                help=defs["scale_data_kmeans"]["help"]
            )
        elif updated_values["method"] == "DBSCAN":
            updated_values["eps"] = st_object.number_input(
                defs["eps"]["label"],
                min_value=defs["eps"]["min_value"],
                value=current_values.get("eps", defs["eps"]["default"]),
                format=defs["eps"]["format"],
                key=f"{module_key}_eps",
                help=defs["eps"]["help"]
            )
            updated_values["min_samples"] = st_object.number_input(
                defs["min_samples"]["label"],
                min_value=defs["min_samples"]["min_value"],
                value=current_values.get("min_samples", defs["min_samples"]["default"]),
                key=f"{module_key}_min_samples",
                help=defs["min_samples"]["help"]
            )
            updated_values["scale_data_dbscan"] = st_object.checkbox(
                defs["scale_data_dbscan"]["label"],
                value=current_values.get("scale_data_dbscan", defs["scale_data_dbscan"]["default"]),
                key=f"{module_key}_scale_data_dbscan",
                help=defs["scale_data_dbscan"]["help"]
            )
        return updated_values


    def run_analysis(self, data: pd.DataFrame, params: Dict[str, Any], session_state: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Executes the selected clustering algorithm on the provided data.

        Args:
            data (pd.DataFrame): The feature DataFrame for clustering.
            params (Dict[str, Any]): A dictionary of parameters for the analysis,
                                     including the clustering method and its specific settings.
            session_state (Dict[str, Any]): The Streamlit session state. Not used in this module.

        Returns:
            Tuple[Any, Optional[str]]: A tuple containing the clustering results
                                       (a dictionary with labels, model, etc.) and an
                                       optional error message string.
        """
        logger.info(f"Running {self.get_name()} with method {params.get('method')}")
        method = params.get("method", "K-Means")

        results = {}
        error_msg = None

        if data.empty:
            return None, "Input data is empty."
        if data.isnull().any().any():
            return None, "Input data contains NaN values. Please clean data before clustering."

        try:
            if method == "K-Means":
                n_clusters = params.get("n_clusters", 3)
                scale = params.get("scale_data_kmeans", True)
                labels, model, err = perform_kmeans_clustering(data, n_clusters=n_clusters, scale_data=scale)
                if err:
                    error_msg = f"K-Means error: {err}"
                else:
                    results = {'method': 'K-Means', 'labels': labels, 'model': model, 'k': n_clusters}
            elif method == "DBSCAN":
                eps = params.get("eps", 0.5)
                min_samples = params.get("min_samples", 5)
                scale = params.get("scale_data_dbscan", True)
                labels, model, err = perform_dbscan_clustering(data, eps=eps, min_samples=min_samples, scale_data=scale)
                if err:
                    error_msg = f"DBSCAN error: {err}"
                else:
                    results = {'method': 'DBSCAN', 'labels': labels, 'model': model}
            else:
                error_msg = f"Unsupported clustering method: {method}"

            if error_msg:
                logger.error(f"Clustering analysis failed: {error_msg}")
                return None, error_msg

            logger.info(f"{method} clustering complete. Found {len(np.unique(results.get('labels', [])))} clusters/groups.")
            return results, None

        except Exception as e:
            logger.error(f"Exception in {self.get_name()}: {e}", exc_info=True)
            return None, f"An unexpected error occurred: {str(e)}"


    def render_results(self, st_object: Any, results: Any, session_state: Dict[str, Any]) -> None:
        """
        Renders the results of the clustering analysis.

        Args:
            st_object (Any): The Streamlit object to render the results on.
            results (Any): The results from the run_analysis method.
            session_state (Dict[str, Any]): The Streamlit session state. Not used in this module.
        """
        # In a real scenario, this would render Streamlit charts/tables for the results
        # This part is heavily UI dependent and would be built out in main.py or similar
        st_object.write(f"Results for {self.get_name()}:")
        if results:
            st_object.write(f"Method: {results.get('method')}")
            labels = results.get('labels')
            if labels is not None:
                st_object.write("Cluster counts:")
                st_object.dataframe(labels.value_counts().rename("Device Count").to_frame())
            if results.get('method') == 'K-Means' and results.get('model'):
                try:
                    # features_df_cleaned is needed for cluster centers if data was scaled
                    # This assumes all_device_features_df is available and was the input to run_analysis
                    # This part might be tricky if the 'data' passed to run_analysis was already scaled
                    # or if the columns are not directly available.
                    # For simplicity, this might need to be handled in main.py where features_df_cleaned is defined.
                    # st_object.write("Cluster Centers (if applicable and data was scaled, interpretation needs original feature names):")
                    # centers_df = pd.DataFrame(results['model'].cluster_centers_, columns=session_state.get('all_device_features_df', pd.DataFrame()).columns) # This is a guess
                    # st_object.dataframe(centers_df)
                    st_object.info("Cluster centers visualization would require access to feature names used for clustering.")
                except Exception as e:
                    st_object.warning(f"Could not display cluster centers: {e}")
        else:
            st_object.info("No clustering results to display.")
