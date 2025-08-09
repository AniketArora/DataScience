import pytest
import pandas as pd
import numpy as np

from src.analysis_modules.clustering import ClusteringAnalysisModule
from src.interfaces import AnalysisModuleInterface

def test_module_metadata_simple():
    module = ClusteringAnalysisModule()
    assert module.get_name() == "Device Behavior Clustering"
    assert isinstance(module.get_description(), str)
    assert len(module.get_description()) > 0
    assert isinstance(module, AnalysisModuleInterface)

# Fixture for sample data
@pytest.fixture
def sample_feature_df():
    """Create a sample DataFrame for clustering tests."""
    rng = np.random.RandomState(42)
    n_samples = 30 # Reduced for faster tests
    n_features = 4
    data = rng.rand(n_samples, n_features)
    df = pd.DataFrame(data, columns=[f'feature_{j}' for j in range(n_features)])
    df.index = [f"device_{i}" for i in range(n_samples)]
    # Add some structure for clustering
    df.iloc[0:10, 0:2] += 2  # Group 1
    df.iloc[10:20, 0:2] -= 2 # Group 2
    # Group 3 is the rest
    return df

def test_perform_kmeans_clustering_basic(sample_feature_df):
    from src.analysis_modules.clustering import perform_kmeans_clustering
    n_clusters = 3
    labels, model, error = perform_kmeans_clustering(sample_feature_df, n_clusters=n_clusters, random_state=42)

    assert error is None
    assert labels is not None
    assert model is not None
    assert len(labels) == len(sample_feature_df)
    assert len(np.unique(labels)) == n_clusters
    assert isinstance(labels, pd.Series)
    assert hasattr(model, 'cluster_centers_')

def test_perform_kmeans_clustering_scale_data(sample_feature_df):
    from src.analysis_modules.clustering import perform_kmeans_clustering
    n_clusters = 3
    # Test with scaling
    labels_scaled, _, error_scaled = perform_kmeans_clustering(sample_feature_df, n_clusters=n_clusters, scale_data=True, random_state=42)
    assert error_scaled is None
    assert labels_scaled is not None
    assert len(np.unique(labels_scaled)) <= n_clusters # Scaling might change exact number of clusters found if some become empty

    # Test without scaling
    labels_unscaled, _, error_unscaled = perform_kmeans_clustering(sample_feature_df, n_clusters=n_clusters, scale_data=False, random_state=42)
    assert error_unscaled is None
    assert labels_unscaled is not None

def test_perform_kmeans_clustering_edge_cases(sample_feature_df):
    from src.analysis_modules.clustering import perform_kmeans_clustering
    # Empty DataFrame
    labels, model, error = perform_kmeans_clustering(pd.DataFrame(), n_clusters=2)
    assert "Input DataFrame is empty" in error
    assert labels is None
    assert model is None

    # NaN DataFrame
    nan_df = sample_feature_df.copy()
    nan_df.iloc[0,0] = np.nan
    labels, model, error = perform_kmeans_clustering(nan_df, n_clusters=2)
    assert "Input DataFrame contains NaN values" in error
    assert labels is None
    assert model is None

    # n_clusters too high
    labels, model, error = perform_kmeans_clustering(sample_feature_df, n_clusters=len(sample_feature_df) + 1)
    assert f"Number of clusters ({len(sample_feature_df) + 1}) cannot exceed number of samples ({len(sample_feature_df)})" in error
    assert labels is None
    assert model is None

    # n_clusters zero or negative
    labels, model, error = perform_kmeans_clustering(sample_feature_df, n_clusters=0)
    assert "Number of clusters (n_clusters) must be positive" in error

def test_get_kmeans_elbow_silhouette_data(sample_feature_df):
    from src.analysis_modules.clustering import get_kmeans_elbow_silhouette_data
    # Ensure sample_feature_df has enough samples for the default k_range (2-10)
    # If sample_feature_df has < 10 samples, this test might fail due to k_range check
    # For this fixture, n_samples = 30, so it's fine.

    # Make sure k_range is valid for the number of samples
    max_k_test = min(10, len(sample_feature_df) -1)
    if max_k_test < 2:
        pytest.skip("Not enough samples for k-means stats range test after considering max_k.")

    k_range = range(2, max_k_test + 1)

    results_df, error = get_kmeans_elbow_silhouette_data(sample_feature_df, k_range=k_range, scale_data=True)
    assert error is None
    assert results_df is not None
    assert 'K' in results_df.columns
    assert 'Inertia' in results_df.columns
    assert 'Silhouette Score' in results_df.columns
    assert len(results_df) == len(k_range)

def test_perform_dbscan_clustering_basic(sample_feature_df):
    from src.analysis_modules.clustering import perform_dbscan_clustering
    # DBSCAN parameters are sensitive, these are just for a smoke test
    labels, model, error = perform_dbscan_clustering(sample_feature_df, eps=1.0, min_samples=3, scale_data=True)
    assert error is None
    assert labels is not None
    assert model is not None
    assert len(labels) == len(sample_feature_df)
    assert isinstance(labels, pd.Series)

def test_perform_dbscan_clustering_edge_cases():
    from src.analysis_modules.clustering import perform_dbscan_clustering
    # Empty DataFrame
    labels, model, error = perform_dbscan_clustering(pd.DataFrame())
    assert "Input DataFrame is empty" in error
    assert labels is None
    assert model is None

    # NaN DataFrame
    nan_df = pd.DataFrame({'A': [1, np.nan], 'B': [2,3]})
    labels, model, error = perform_dbscan_clustering(nan_df)
    assert "Input DataFrame contains NaN values" in error
    assert labels is None
    assert model is None

# --- Tests for ClusteringAnalysisModule ---
class TestClusteringAnalysisModule:
    @pytest.fixture
    def module(self):
        return ClusteringAnalysisModule()

    def test_run_analysis_kmeans(self, module, sample_feature_df):
        params = {
            "method": "K-Means",
            "n_clusters": 3,
            "scale_data_kmeans": True
        }
        results, error = module.run_analysis(sample_feature_df, params, {})
        assert error is None
        assert results is not None
        assert results['method'] == "K-Means"
        assert "labels" in results
        assert "model" in results

    def test_run_analysis_dbscan(self, module, sample_feature_df):
        params = {
            "method": "DBSCAN",
            "eps": 1.0, # Adjusted for sample data
            "min_samples": 3,
            "scale_data_dbscan": True
        }
        results, error = module.run_analysis(sample_feature_df, params, {})
        assert error is None
        assert results is not None
        assert results['method'] == "DBSCAN"
        assert "labels" in results
        assert "model" in results

    def test_run_analysis_empty_df(self, module):
        params = {"method": "K-Means"}
        _, error = module.run_analysis(pd.DataFrame(), params, {})
        assert "Input data is empty" in error

    def test_run_analysis_nan_df(self, module):
        params = {"method": "K-Means"}
        nan_df = pd.DataFrame({'A': [1, np.nan], 'B': [2, 3]})
        _, error = module.run_analysis(nan_df, params, {})
        assert "Input data contains NaN values" in error

    def test_run_analysis_unknown_method(self, module, sample_feature_df):
        params = {"method": "Unknown Method"}
        _, error = module.run_analysis(sample_feature_df, params, {})
        assert "Unsupported clustering method" in error
