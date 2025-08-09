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
    explain_anomalies_with_surrogate_model,
    analyze_event_correlations # Added import
)

# --- Fixtures for existing tests ---
@pytest.fixture
def sample_features_for_explainability():
    rng = np.random.RandomState(42)
    n_dev = 60
    n_feat = 3
    features = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    features.iloc[0:n_dev//3, 0] += 2
    features.iloc[n_dev//3:2*n_dev//3, 0] -= 2
    features.iloc[0:n_dev//2, 1] += 1.5
    return features

@pytest.fixture
def sample_cluster_labels(sample_features_for_explainability):
    n_dev = len(sample_features_for_explainability)
    return pd.Series([0]*(n_dev//3) + [1]*(n_dev//3) + [2]*(n_dev - 2*(n_dev//3)), index=sample_features_for_explainability.index)

@pytest.fixture
def sample_anomaly_labels(sample_features_for_explainability):
    n_dev = len(sample_features_for_explainability)
    labels = pd.Series([1]*n_dev, index=sample_features_for_explainability.index)
    anomalous_indices = sample_features_for_explainability.sample(n=n_dev//5, random_state=1).index
    labels.loc[anomalous_indices] = -1
    return labels

# --- Tests for existing functions (condensed) ---
def test_get_cluster_feature_summary_runs(sample_features_for_explainability, sample_cluster_labels):
    summary_df, error = get_cluster_feature_summary(sample_features_for_explainability, sample_cluster_labels)
    assert error is None; assert summary_df is not None; assert len(summary_df) == sample_cluster_labels.nunique()

def test_get_feature_importance_for_clusters_anova_runs(sample_features_for_explainability, sample_cluster_labels):
    importance_df, error = get_feature_importance_for_clusters_anova(sample_features_for_explainability, sample_cluster_labels)
    assert error is None; assert importance_df is not None; assert not importance_df.empty

def test_compare_anomalous_vs_normal_features_runs(sample_features_for_explainability, sample_anomaly_labels):
    comparison_df, error = compare_anomalous_vs_normal_features(sample_features_for_explainability, sample_anomaly_labels)
    assert error is None; assert comparison_df is not None

def test_generate_cluster_summary_text_runs(sample_features_for_explainability, sample_cluster_labels):
    summary_df, _ = get_cluster_feature_summary(sample_features_for_explainability, sample_cluster_labels)
    overall_means = sample_features_for_explainability.mean()
    text = generate_cluster_summary_text(0, (sample_cluster_labels == 0).sum(), len(sample_features_for_explainability), summary_df.loc[0], overall_means)
    assert isinstance(text, str); assert "Cluster 0" in text

def test_generate_anomaly_summary_text_runs(sample_features_for_explainability, sample_anomaly_labels):
    comparison_df, _ = compare_anomalous_vs_normal_features(sample_features_for_explainability, sample_anomaly_labels)
    anomalous_device_id = sample_anomaly_labels[sample_anomaly_labels == -1].index[0]
    text = generate_anomaly_summary_text(anomalous_device_id, -2.5, comparison_df)
    assert isinstance(text, str); assert f"Device {anomalous_device_id}" in text

# --- Fixtures for Surrogate Model Anomaly Explanation ---
@pytest.fixture
def sample_features_for_surrogate():
    rng = np.random.RandomState(42); n_dev = 50; n_feat = 4
    X = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    X.iloc[0:n_dev//2, 0] += 0.5; X.iloc[0:n_dev//4, 1] -= 0.5
    return X

@pytest.fixture
def sample_anomaly_labels_for_surrogate(sample_features_for_surrogate):
    n_dev = len(sample_features_for_surrogate)
    labels = pd.Series([1] * n_dev, index=sample_features_for_surrogate.index)
    labels.iloc[0:n_dev//4] = -1
    return labels

# --- Tests for Surrogate Model Anomaly Explanation ---
def test_explain_anomalies_with_surrogate_model_basic_run(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate):
    tree_model, importances, report, error = explain_anomalies_with_surrogate_model(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate, test_size=0)
    assert error is None; assert isinstance(tree_model, DecisionTreeClassifier); assert isinstance(importances, pd.Series)
    assert not importances.empty; assert report is None; assert set(importances.index) == set(sample_features_for_surrogate.columns)

def test_explain_anomalies_with_surrogate_model_with_test_set(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate):
    tree_model, importances, report, error = explain_anomalies_with_surrogate_model(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate, test_size=0.25)
    assert error is None; assert isinstance(tree_model, DecisionTreeClassifier); assert isinstance(importances, pd.Series)
    assert isinstance(report, dict); assert 'accuracy' in report

def test_explain_anomalies_surrogate_input_validation():
    tree, imp, rep, err = explain_anomalies_with_surrogate_model(pd.DataFrame(), pd.Series([], dtype=int))
    assert "Input DataFrame or labels Series is empty" in err; assert tree is None
    feat_df = pd.DataFrame({'A': [1,2,3]}, index=['d1','d2','d3'])
    labels_mismatch_idx = pd.Series([-1, 1], index=['d10','d20'], dtype=int)
    tree, imp, rep, err = explain_anomalies_with_surrogate_model(feat_df, labels_mismatch_idx)
    assert "Indices must match." in err; assert tree is None # Updated error message check
    labels_single_class = pd.Series([1,1,1], index=['d1','d2','d3'], dtype=int)
    tree, imp, rep, err = explain_anomalies_with_surrogate_model(feat_df, labels_single_class)
    assert "Need at least two distinct classes for surrogate model training." in err; assert tree is None

def test_explain_anomalies_surrogate_model_importances_check(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate):
    _, importances, _, error = explain_anomalies_with_surrogate_model(sample_features_for_surrogate, sample_anomaly_labels_for_surrogate, max_depth=3, test_size=0)
    assert error is None; assert isinstance(importances, pd.Series); assert not importances.empty
    assert np.isclose(importances.sum(), 1.0)
    assert importances.index[0] in ['feat_0', 'feat_1']

# --- Fixtures for Event Correlation Tests ---
@pytest.fixture
def features_with_events_for_corr(sample_features_for_explainability): # Using a larger, more diverse base
    df = sample_features_for_explainability.copy()
    rng = np.random.RandomState(42)
    df['evt_count_Error_A'] = rng.randint(0, 3, size=len(df))
    df['evt_count_Warning_B'] = rng.randint(0, 5, size=len(df))
    df['evt_count_Info_C'] = rng.randint(0, 2, size=len(df))
    # Make Error_A more common for anomalous devices (using sample_anomaly_labels fixture)
    # Recreate anomaly labels for this specific fixture to ensure index alignment
    temp_anomaly_labels = pd.Series([1]*len(df), index=df.index)
    anomalous_indices_event_test = df.sample(n=len(df)//5, random_state=1).index
    temp_anomaly_labels.loc[anomalous_indices_event_test] = -1
    df.loc[anomalous_indices_event_test, 'evt_count_Error_A'] = rng.randint(3, 6, size=len(anomalous_indices_event_test))
    return df, temp_anomaly_labels

# --- Tests for analyze_event_correlations ---
def test_analyze_event_correlations_anomalies(features_with_events_for_corr):
    df_with_events, anomaly_labels_for_events = features_with_events_for_corr
    corr_df, error = analyze_event_correlations(df_with_events, anomaly_labels_for_events, event_feature_prefix="evt_count_")
    assert error is None
    assert isinstance(corr_df, pd.DataFrame)
    assert 'evt_count_Error_A' in corr_df.index # Expect prefixed name
    assert 'evt_count_Warning_B' in corr_df.index # Expect prefixed name
    assert 'Anomalous_Mean_Events' in corr_df.columns
    assert 'Normal_Mean_Events' in corr_df.columns
    assert 'Overall_Mean_Events' in corr_df.columns
    assert corr_df.loc['evt_count_Error_A', 'Anomalous_Mean_Events'] > corr_df.loc['evt_count_Error_A', 'Normal_Mean_Events']

def test_analyze_event_correlations_clusters(features_with_events_for_corr, sample_cluster_labels):
    df_with_events, _ = features_with_events_for_corr # We use sample_cluster_labels for this test
    # Ensure sample_cluster_labels aligns with df_with_events (they both use sample_features_for_explainability)

    corr_df, error = analyze_event_correlations(df_with_events, sample_cluster_labels, event_feature_prefix="evt_count_")
    assert error is None
    assert isinstance(corr_df, pd.DataFrame)
    assert 'evt_count_Error_A' in corr_df.index # Expect prefixed name
    assert 'Cluster_0_Mean_Events' in corr_df.columns
    assert 'Cluster_1_Mean_Events' in corr_df.columns
    assert 'Cluster_2_Mean_Events' in corr_df.columns
    assert 'Overall_Mean_Events' in corr_df.columns

def test_analyze_event_correlations_no_event_features(sample_features_for_explainability, sample_anomaly_labels):
    # sample_features_for_explainability does not have 'evt_count_' columns
    corr_df, error = analyze_event_correlations(sample_features_for_explainability, sample_anomaly_labels)
    assert "No event count features found" in error
    assert corr_df is None

def test_analyze_event_correlations_mismatched_indices(features_with_events_for_corr):
    df_with_events, anomaly_labels_for_events = features_with_events_for_corr
    labels_mismatch = anomaly_labels_for_events.iloc[0:10] # Different length
    corr_df, error = analyze_event_correlations(df_with_events, labels_mismatch)
    assert error is None # Should align to common subset
    assert corr_df is not None
    assert len(corr_df.columns) <= 3 # Anomalous, Normal, Overall (or less if one group missing in subset)

def test_analyze_event_correlations_empty_inputs():
    df, err = analyze_event_correlations(pd.DataFrame(), pd.Series([],dtype='int'))
    assert "Input DataFrame or labels Series is empty" in err; assert df is None
    df, err = analyze_event_correlations(pd.DataFrame({'A':[1]}), pd.Series([],dtype='int'))
    assert "Input DataFrame or labels Series is empty" in err; assert df is None

# --- More detailed tests for explainability functions ---

def test_get_cluster_feature_summary_error_cases(sample_features_for_explainability, sample_cluster_labels):
    features, cluster_labels = sample_features_for_explainability, sample_cluster_labels

    # Not a series/dataframe
    _, error = get_cluster_feature_summary(features.values, cluster_labels)
    assert "Inputs must be pandas DataFrame and Series" in error

    # Empty dataframe
    _, error = get_cluster_feature_summary(pd.DataFrame(), cluster_labels)
    assert "Input DataFrame or labels Series is empty" in error

    # Mismatched index
    _, error = get_cluster_feature_summary(features, cluster_labels.reset_index(drop=True))
    assert "must have the same index" in error

    # No numeric features
    non_numeric_df = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w']}, index=['d1', 'd2'])
    labels = pd.Series([0, 1], index=non_numeric_df.index)
    _, error = get_cluster_feature_summary(non_numeric_df, labels)
    assert "No numeric features found" in error

from src.analysis_modules.explainability import analyze_significant_event_types

def test_analyze_significant_event_types(features_with_events_for_corr):
    df_with_events, anomaly_labels_for_events = features_with_events_for_corr

    # Test anomalous vs all others
    sig_df, error = analyze_significant_event_types(
        df_with_events, anomaly_labels_for_events, "evt_count_", at_risk_label=-1
    )
    assert error is None
    assert sig_df is not None
    assert not sig_df.empty
    assert 'lift' in sig_df.columns
    assert 'p_value' in sig_df.columns

    # Test anomalous vs specific baseline
    sig_df, error = analyze_significant_event_types(
        df_with_events, anomaly_labels_for_events, "evt_count_", at_risk_label=-1, baseline_label=1
    )
    assert error is None
    assert sig_df is not None

    # Test error cases
    _, error = analyze_significant_event_types(pd.DataFrame(), pd.Series([], dtype=int), "evt_count_", -1)
    assert "Input DataFrame or labels Series is empty" in error

    _, error = analyze_significant_event_types(df_with_events, anomaly_labels_for_events, "evt_count_", at_risk_label=100)
    assert "No devices found for the at-risk label" in error
