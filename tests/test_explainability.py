import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.analysis_modules.explainability import (
    get_cluster_feature_summary,
    get_feature_importance_for_clusters_anova,
    compare_anomalous_vs_normal_features,
    generate_cluster_summary_text,
    generate_anomaly_summary_text,
    explain_anomalies_with_surrogate_model # Added new function
)

# --- Fixtures for existing tests (assuming they might be needed or adapted) ---
@pytest.fixture
def sample_features_for_explainability(): # Renamed to avoid conflict if other tests use similar names
    rng = np.random.RandomState(42)
    n_dev = 60 # Divisible by 3 for cluster tests
    n_feat = 3
    features = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    features.iloc[0:n_dev//3, 0] += 2  # Group A for feat_0
    features.iloc[n_dev//3:2*n_dev//3, 0] -= 2 # Group B for feat_0
    features.iloc[0:n_dev//2, 1] += 1.5 # Group C for feat_1 (overlaps with A)
    return features

@pytest.fixture
def sample_cluster_labels(sample_features_for_explainability):
    n_dev = len(sample_features_for_explainability)
    return pd.Series([0]*(n_dev//3) + [1]*(n_dev//3) + [2]*(n_dev - 2*(n_dev//3)), index=sample_features_for_explainability.index)

@pytest.fixture
def sample_anomaly_labels(sample_features_for_explainability):
    n_dev = len(sample_features_for_explainability)
    labels = pd.Series([1]*n_dev, index=sample_features_for_explainability.index)
    anomalous_indices = sample_features_for_explainability.sample(n=n_dev//5, random_state=1).index # 20% anomalies
    labels.loc[anomalous_indices] = -1
    return labels

# --- Tests for existing functions (condensed for brevity, assuming they pass from previous steps) ---
def test_get_cluster_feature_summary_runs(sample_features_for_explainability, sample_cluster_labels):
    summary_df, error = get_cluster_feature_summary(sample_features_for_explainability, sample_cluster_labels)
    assert error is None
    assert summary_df is not None
    assert len(summary_df) == sample_cluster_labels.nunique()

def test_get_feature_importance_for_clusters_anova_runs(sample_features_for_explainability, sample_cluster_labels):
    importance_df, error = get_feature_importance_for_clusters_anova(sample_features_for_explainability, sample_cluster_labels)
    assert error is None
    assert importance_df is not None
    assert not importance_df.empty

def test_compare_anomalous_vs_normal_features_runs(sample_features_for_explainability, sample_anomaly_labels):
    comparison_df, error = compare_anomalous_vs_normal_features(sample_features_for_explainability, sample_anomaly_labels)
    assert error is None
    assert comparison_df is not None
    # Add more specific assertions if needed based on fixture data behavior

def test_generate_cluster_summary_text_runs(sample_features_for_explainability, sample_cluster_labels):
    importance_df, _ = get_feature_importance_for_clusters_anova(sample_features_for_explainability, sample_cluster_labels)
    text = generate_cluster_summary_text(0, (sample_cluster_labels == 0).sum(), len(sample_features_for_explainability), importance_df)
    assert isinstance(text, str)
    assert "Cluster 0" in text

def test_generate_anomaly_summary_text_runs(sample_features_for_explainability, sample_anomaly_labels):
    comparison_df, _ = compare_anomalous_vs_normal_features(sample_features_for_explainability, sample_anomaly_labels)
    anomalous_device_id = sample_anomaly_labels[sample_anomaly_labels == -1].index[0]
    text = generate_anomaly_summary_text(anomalous_device_id, -2.5, comparison_df) # Example score
    assert isinstance(text, str)
    assert f"Device {anomalous_device_id}" in text


# --- Fixtures for Surrogate Model Anomaly Explanation ---
@pytest.fixture
def sample_features_for_surrogate():
    rng = np.random.RandomState(42)
    n_dev = 50
    n_feat = 4
    X = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    X.iloc[0:n_dev//2, 0] += 0.5
    X.iloc[0:n_dev//4, 1] -= 0.5
    return X

@pytest.fixture
def sample_anomaly_labels_for_surrogate(sample_features_for_surrogate):
    n_dev = len(sample_features_for_surrogate)
    labels = pd.Series([1] * n_dev, index=sample_features_for_surrogate.index)
    labels.iloc[0:n_dev//4] = -1
    return labels

# --- Tests for Surrogate Model Anomaly Explanation ---

def test_explain_anomalies_with_surrogate_model_basic_run(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate):
    tree_model, importances, report, error = explain_anomalies_with_surrogate_model(
        sample_features_for_surrogate,
        sample_anomaly_labels_for_surrogate,
        test_size=0
    )
    assert error is None
    assert isinstance(tree_model, DecisionTreeClassifier)
    assert isinstance(importances, pd.Series)
    assert not importances.empty
    assert report is None
    assert set(importances.index) == set(sample_features_for_surrogate.columns)

def test_explain_anomalies_with_surrogate_model_with_test_set(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate):
    tree_model, importances, report, error = explain_anomalies_with_surrogate_model(
        sample_features_for_surrogate,
        sample_anomaly_labels_for_surrogate,
        test_size=0.25
    )
    assert error is None
    assert isinstance(tree_model, DecisionTreeClassifier)
    assert isinstance(importances, pd.Series)
    assert isinstance(report, dict)
    assert 'accuracy' in report

def test_explain_anomalies_surrogate_input_validation():
    tree, imp, rep, err = explain_anomalies_with_surrogate_model(pd.DataFrame(), pd.Series([], dtype=int))
    assert "Input DataFrame or labels Series is empty" in err
    assert tree is None

    feat_df = pd.DataFrame({'A': [1,2,3]}, index=['d1','d2','d3'])
    labels_mismatch_idx = pd.Series([-1, 1], index=['d10','d20'], dtype=int)
    tree, imp, rep, err = explain_anomalies_with_surrogate_model(feat_df, labels_mismatch_idx)
    assert "must have the same index" in err
    assert tree is None

    labels_single_class = pd.Series([1,1,1], index=['d1','d2','d3'], dtype=int)
    tree, imp, rep, err = explain_anomalies_with_surrogate_model(feat_df, labels_single_class)
    assert "must have at least two distinct classes" in err
    assert tree is None

def test_explain_anomalies_surrogate_model_importances_check(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate):
    _, importances, _, error = explain_anomalies_with_surrogate_model(
        sample_features_for_surrogate,
        sample_anomaly_labels_for_surrogate,
        max_depth=3,
        test_size=0
    )
    assert error is None
    assert isinstance(importances, pd.Series)
    assert not importances.empty
    assert np.isclose(importances.sum(), 1.0) # Standard check for feature importances
    # Check that the most important feature is one of the ones we designed to be discriminative
    assert importances.index[0] in ['feat_0', 'feat_1']
