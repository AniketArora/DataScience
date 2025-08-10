import pytest
import pandas as pd
import numpy as np
from src.analysis_modules.explainability import (
    get_cluster_feature_summary,
    get_feature_importance_for_clusters_anova,
    compare_anomalous_vs_normal_features,
    generate_cluster_summary_text,
    generate_anomaly_summary_text,
    explain_anomalies_with_surrogate_model,
    analyze_event_correlations,
    analyze_significant_event_types,
)

@pytest.fixture
def sample_data():
    """Provides sample data for testing."""
    rng = np.random.RandomState(42)
    n_dev = 100
    n_feat = 3
    features = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    features.iloc[0:30, 0] += 2
    features.iloc[30:60, 0] -= 2
    features.iloc[0:20, 1] += 1.5
    mock_cluster_labels = pd.Series([0]*(n_dev//3) + [1]*(n_dev//3) + [2]*(n_dev - 2*(n_dev//3)), index=features.index)
    mock_anomaly_labels = pd.Series([1]*n_dev, index=features.index)
    anomalous_indices = features.sample(n=10, random_state=1).index
    mock_anomaly_labels.loc[anomalous_indices] = -1
    return features, mock_cluster_labels, mock_anomaly_labels, anomalous_indices

def test_get_cluster_feature_summary(sample_data):
    features, mock_cluster_labels, _, _ = sample_data
    summary_df, error = get_cluster_feature_summary(features, mock_cluster_labels)
    assert error is None
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.shape == (3, 3)

def test_get_feature_importance_for_clusters_anova(sample_data):
    features, mock_cluster_labels, _, _ = sample_data
    importance_df, error = get_feature_importance_for_clusters_anova(features, mock_cluster_labels)
    assert error is None
    assert isinstance(importance_df, pd.DataFrame)
    assert "F-Value" in importance_df.columns
    assert "P-Value" in importance_df.columns

def test_compare_anomalous_vs_normal_features(sample_data):
    features, _, mock_anomaly_labels, _ = sample_data
    comparison_df, error = compare_anomalous_vs_normal_features(features, mock_anomaly_labels)
    assert error is None
    assert isinstance(comparison_df, pd.DataFrame)
    assert "Anomalous_Mean" in comparison_df.columns
    assert "Normal_Mean" in comparison_df.columns

def test_generate_cluster_summary_text(sample_data):
    features, mock_cluster_labels, _, _ = sample_data
    summary_df, _ = get_cluster_feature_summary(features, mock_cluster_labels)
    overall_means_for_summary = features.mean()
    text = generate_cluster_summary_text(0, (mock_cluster_labels == 0).sum(), len(features), summary_df.loc[0], overall_means_for_summary)
    assert isinstance(text, str)
    assert "Cluster 0" in text
    assert "contains 33 devices" in text

def test_generate_anomaly_summary_text(sample_data):
    features, _, mock_anomaly_labels, anomalous_indices = sample_data
    comparison_df, _ = compare_anomalous_vs_normal_features(features, mock_anomaly_labels)
    text = generate_anomaly_summary_text(device_id=anomalous_indices[0], anomaly_score=-1.5, top_features_comparison=comparison_df)
    assert isinstance(text, str)
    assert f"Device {anomalous_indices[0]}" in text
    assert "is flagged as anomalous" in text

def test_explain_anomalies_with_surrogate_model(sample_data):
    features, _, mock_anomaly_labels, _ = sample_data
    features_cleaned_for_surrogate = features.dropna()
    labels_for_surrogate = mock_anomaly_labels.loc[features_cleaned_for_surrogate.index]
    tree_model, tree_importances, tree_report, error_tree = explain_anomalies_with_surrogate_model(
        features_cleaned_for_surrogate, labels_for_surrogate, max_depth=3, test_size=0.0
    )
    assert error_tree is None
    assert tree_model is not None
    assert isinstance(tree_importances, pd.Series)
    assert tree_report is None

def test_analyze_event_correlations(sample_data):
    features, _, mock_anomaly_labels, _ = sample_data
    mock_features_with_events = features.copy()
    mock_features_with_events['evt_count_Error_X'] = np.random.randint(0, 5, size=len(features))
    event_corr_df, error_ec = analyze_event_correlations(mock_features_with_events, mock_anomaly_labels)
    assert error_ec is None
    assert isinstance(event_corr_df, pd.DataFrame)
    assert "Anomalous_Mean_Events" in event_corr_df.index
    assert "Normal_Mean_Events" in event_corr_df.index
    assert "Overall_Mean_Events" in event_corr_df.index

def test_analyze_event_correlations_with_cluster_labels(sample_data):
    features, mock_cluster_labels, _, _ = sample_data
    mock_features_with_events = features.copy()
    mock_features_with_events['evt_count_Error_X'] = np.random.randint(0, 5, size=len(features))
    event_corr_df, error_ec = analyze_event_correlations(mock_features_with_events, mock_cluster_labels)
    assert error_ec is None
    assert isinstance(event_corr_df, pd.DataFrame)
    assert "Cluster_0_Mean_Events" in event_corr_df.index
    assert "Cluster_1_Mean_Events" in event_corr_df.index
    assert "Overall_Mean_Events" in event_corr_df.index

def test_analyze_event_correlations_no_events(sample_data):
    features, _, mock_anomaly_labels, _ = sample_data
    _, error = analyze_event_correlations(features, mock_anomaly_labels)
    assert "No event count features found" in error

def test_analyze_significant_event_types(sample_data):
    features, _, mock_anomaly_labels, _ = sample_data
    mock_features_with_events = features.copy()
    mock_features_with_events['evt_count_Error_X'] = 0
    anomalous_indices = mock_anomaly_labels[mock_anomaly_labels == -1].index
    mock_features_with_events.loc[anomalous_indices, 'evt_count_Error_X'] = 1

    sig_events_df, error_sig = analyze_significant_event_types(
        mock_features_with_events,
        mock_anomaly_labels,
        event_feature_prefix="evt_count_",
        at_risk_label=-1
    )
    assert error_sig is None
    assert isinstance(sig_events_df, pd.DataFrame)
    assert "lift" in sig_events_df.columns
    assert "p_value" in sig_events_df.columns
    assert sig_events_df.loc[sig_events_df['event_type'] == 'Error_X', 'lift'].iloc[0] > 1

def test_analyze_significant_event_types_with_clusters(sample_data):
    features, mock_cluster_labels, _, _ = sample_data
    mock_features_with_events = features.copy()
    mock_features_with_events['evt_count_Warning_Y'] = 0
    cluster_0_indices = mock_cluster_labels[mock_cluster_labels == 0].index
    mock_features_with_events.loc[cluster_0_indices, 'evt_count_Warning_Y'] = 1

    sig_events_df, error_sig = analyze_significant_event_types(
        mock_features_with_events,
        mock_cluster_labels,
        event_feature_prefix="evt_count_",
        at_risk_label=0,
        baseline_label=1
    )
    assert error_sig is None
    assert isinstance(sig_events_df, pd.DataFrame)
    assert sig_events_df.loc[sig_events_df['event_type'] == 'Warning_Y', 'lift'].iloc[0] > 1

def test_analyze_significant_event_types_errors(sample_data):
    features, _, mock_anomaly_labels, _ = sample_data
    mock_features_with_events = features.copy()
    mock_features_with_events['evt_count_Error_X'] = 0

    _, error_sig_no_risk = analyze_significant_event_types(
        mock_features_with_events,
        mock_anomaly_labels,
        event_feature_prefix="evt_count_",
        at_risk_label=-2
    )
    assert "No devices found for the at-risk label" in error_sig_no_risk

    _, error_no_events = analyze_significant_event_types(
        features,
        mock_anomaly_labels,
        event_feature_prefix="evt_count_",
        at_risk_label=-1
    )
    assert "No event count features found" in error_no_events

    _, error_no_baseline = analyze_significant_event_types(
        mock_features_with_events,
        mock_anomaly_labels,
        event_feature_prefix="evt_count_",
        at_risk_label=-1,
        baseline_label=5
    )
    assert "No devices found for the baseline label" in error_no_baseline
