import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
# For k-need, if we want to use it for Elbow or DBSCAN eps automatically
# from kneed import KneeLocator # Requires 'kneed' package
# For now, will implement manual elbow/silhouette plotting guidance

def scale_features(feature_df: pd.DataFrame):
    """Scales features using StandardScaler."""
    if not isinstance(feature_df, pd.DataFrame) or feature_df.empty:
        return feature_df, None # Return original if not suitable for scaling, plus an error/None scaler

    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(feature_df)
        scaled_feature_df = pd.DataFrame(scaled_features, index=feature_df.index, columns=feature_df.columns)
        return scaled_feature_df, scaler
    except Exception as e:
        # If scaling fails (e.g., all-NaN column after some upstream processing error)
        # This shouldn't happen if feature_df is validated (e.g. no NaNs, numeric)
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
        return None, None, "Input is not a pandas DataFrame."
    if feature_df.empty:
        return None, None, "Input DataFrame is empty."
    if feature_df.isnull().any().any():
        return None, None, "Input DataFrame contains NaN values. Please handle them."
    if n_clusters <= 0:
        return None, None, "Number of clusters (n_clusters) must be positive."
    if n_clusters > len(feature_df):
        return None, None, f"Number of clusters ({n_clusters}) cannot exceed number of samples ({len(feature_df)})."


    df_to_cluster = feature_df
    if scale_data:
        df_to_cluster, _ = scale_features(feature_df) # Scaler object not used here but good practice

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', **kwargs)
        kmeans.fit(df_to_cluster)
        labels = pd.Series(kmeans.labels_, index=df_to_cluster.index, name="kmeans_cluster_labels")
        return labels, kmeans, None
    except Exception as e:
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
    if not isinstance(feature_df, pd.DataFrame): return None, "Input is not a pandas DataFrame."
    if feature_df.empty: return None, "Input DataFrame is empty."
    if feature_df.isnull().any().any(): return None, "Input DataFrame contains NaN values."
    # Ensure k_range is valid and max K is less than number of samples
    min_k = min(k_range) if len(k_range) > 0 else 0
    max_k = max(k_range) if len(k_range) > 0 else 0

    if not k_range or min_k < 2 : return None, "k_range must start from at least 2."
    if max_k >= len(feature_df): return None, f"Max K in k_range ({max_k}) must be less than number of samples ({len(feature_df)})."


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
        return None, None, "Input is not a pandas DataFrame."
    if feature_df.empty:
        return None, None, "Input DataFrame is empty."
    if feature_df.isnull().any().any():
        return None, None, "Input DataFrame contains NaN values. Please handle them."

    df_to_cluster = feature_df
    if scale_data:
        df_to_cluster, _ = scale_features(feature_df)

    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        dbscan.fit(df_to_cluster)
        labels = pd.Series(dbscan.labels_, index=df_to_cluster.index, name="dbscan_cluster_labels")
        return labels, dbscan, None
    except Exception as e:
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
