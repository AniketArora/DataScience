import pandas as pd
import numpy as np
from scipy.stats import f_oneway # For ANOVA F-value for numeric features across clusters
import streamlit as st # For st.warning in compare_anomalous_vs_normal_features

def get_cluster_feature_summary(feature_df: pd.DataFrame, cluster_labels: pd.Series):
    """
    Calculates mean feature values for each cluster.

    Args:
        feature_df (pd.DataFrame): DataFrame of features (devices x features).
        cluster_labels (pd.Series): Series of cluster labels for each device.
                                    Should have the same index as feature_df.

    Returns:
        pd.DataFrame or None: DataFrame with clusters as index, features as columns,
                              and mean values as entries. Returns None on error.
        str or None: Error message.
    """
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(cluster_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or cluster_labels.empty:
        return None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(cluster_labels.index):
        return None, "Feature DataFrame and cluster labels must have the same index."

    try:
        # Ensure cluster_labels does not contain NaNs which would break groupby
        if cluster_labels.isnull().any():
            valid_indices = cluster_labels.dropna().index
            feature_df_cleaned = feature_df.loc[valid_indices]
            cluster_labels_cleaned = cluster_labels.loc[valid_indices]
            if feature_df_cleaned.empty:
                 return None, "No valid data after removing NaN cluster labels."
        else:
            feature_df_cleaned = feature_df
            cluster_labels_cleaned = cluster_labels

        numeric_feature_df = feature_df_cleaned.select_dtypes(include=np.number)
        if numeric_feature_df.empty:
            return None, "No numeric features found in the DataFrame to summarize."

        summary_df = numeric_feature_df.groupby(cluster_labels_cleaned).mean()
        return summary_df, None
    except Exception as e:
        return None, f"Error calculating cluster feature summary: {e}"

def get_feature_importance_for_clusters_anova(feature_df: pd.DataFrame, cluster_labels: pd.Series):
    """
    Ranks features by their importance in differentiating clusters using ANOVA F-value.
    Higher F-value suggests the feature differs more significantly across clusters.
    This is for numeric features only.

    Args:
        feature_df (pd.DataFrame): DataFrame of features.
        cluster_labels (pd.Series): Series of cluster labels.

    Returns:
        pd.DataFrame or None: DataFrame with features and their F-values and p-values, sorted by F-value.
        str or None: Error message.
    """
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(cluster_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or cluster_labels.empty:
        return None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(cluster_labels.index):
        return None, "Feature DataFrame and cluster labels must have the same index."

    valid_indices = cluster_labels.dropna().index
    feature_df_cleaned = feature_df.loc[valid_indices]
    cluster_labels_cleaned = cluster_labels.loc[valid_indices]

    if cluster_labels_cleaned.nunique() < 2:
        return None, "Need at least two clusters to perform ANOVA."
    if feature_df_cleaned.empty:
        return None, "Feature DataFrame is empty after handling NaN labels."

    numeric_feature_df = feature_df_cleaned.select_dtypes(include=np.number)
    if numeric_feature_df.empty:
        return None, "No numeric features found for ANOVA."

    f_values = []
    p_values = []
    features = []

    for feature_name in numeric_feature_df.columns:
        try:
            groups = [numeric_feature_df[feature_name][cluster_labels_cleaned == c] for c in cluster_labels_cleaned.unique()]
            groups = [g for g in groups if len(g.dropna()) > 1 and g.dropna().var() > 1e-6]

            if len(groups) < 2:
                f_stat, p_val = np.nan, np.nan
            else:
                f_stat, p_val = f_oneway(*groups)

            f_values.append(f_stat)
            p_values.append(p_val)
            features.append(feature_name)
        except Exception:
            f_values.append(np.nan)
            p_values.append(np.nan)
            features.append(feature_name)

    if not features:
        return None, "No features processed for ANOVA."

    importance_df = pd.DataFrame({'Feature': features, 'F-Value': f_values, 'P-Value': p_values})
    importance_df = importance_df.sort_values(by='F-Value', ascending=False).reset_index(drop=True)
    return importance_df, None


def compare_anomalous_vs_normal_features(feature_df: pd.DataFrame, anomaly_labels: pd.Series, anomalous_label_val=-1):
    """
    Compares mean feature values of anomalous devices vs. normal devices.

    Args:
        feature_df (pd.DataFrame): DataFrame of features.
        anomaly_labels (pd.Series): Series of anomaly labels (-1 for anomalous, 1 for normal, or similar).
        anomalous_label_val: The value in anomaly_labels that identifies an anomaly.

    Returns:
        pd.DataFrame or None: DataFrame comparing mean features for anomalous vs. normal.
        str or None: Error message.
    """
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(anomaly_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or anomaly_labels.empty:
        return None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(anomaly_labels.index):
        return None, "Feature DataFrame and anomaly labels must have the same index."

    try:
        numeric_feature_df = feature_df.select_dtypes(include=np.number)
        if numeric_feature_df.empty:
            return None, "No numeric features found."

        normal_mask = (anomaly_labels != anomalous_label_val)
        anomalous_mask = (anomaly_labels == anomalous_label_val)

        if not normal_mask.any() and not anomalous_mask.any():
             return None, "No normal or anomalous devices found to compare (all labels might be NaN or unexpected)."
        if not normal_mask.any():
            st.warning("No 'normal' devices found for comparison. Displaying only anomalous means.")
            anomalous_features_mean = numeric_feature_df[anomalous_mask].mean().rename("Anomalous_Mean")
            comparison_df = pd.DataFrame(anomalous_features_mean)
            comparison_df['Normal_Mean'] = np.nan
            comparison_df['Difference (Anomalous - Normal)'] = np.nan
            comparison_df['Relative_Difference (%)'] = np.nan
            return comparison_df, None
        if not anomalous_mask.any():
            st.warning("No 'anomalous' devices found for comparison. Displaying only normal means.")
            normal_features_mean = numeric_feature_df[normal_mask].mean().rename("Normal_Mean")
            comparison_df = pd.DataFrame(normal_features_mean)
            comparison_df['Anomalous_Mean'] = np.nan
            comparison_df['Difference (Anomalous - Normal)'] = np.nan
            comparison_df['Relative_Difference (%)'] = np.nan
            return comparison_df, None

        normal_features_mean = numeric_feature_df[normal_mask].mean().rename("Normal_Mean")
        anomalous_features_mean = numeric_feature_df[anomalous_mask].mean().rename("Anomalous_Mean")

        comparison_df = pd.concat([normal_features_mean, anomalous_features_mean], axis=1)
        comparison_df['Difference (Anomalous - Normal)'] = comparison_df['Anomalous_Mean'] - comparison_df['Normal_Mean']
        comparison_df['Relative_Difference (%)'] = (comparison_df['Difference (Anomalous - Normal)'] / comparison_df['Normal_Mean'].abs()) * 100
        comparison_df['Relative_Difference (%)'] = comparison_df['Relative_Difference (%)'].replace([np.inf, -np.inf], np.nan)

        return comparison_df.sort_values(by='Difference (Anomalous - Normal)', key=abs, ascending=False), None
    except Exception as e:
        return None, f"Error comparing anomalous vs. normal features: {e}"


def generate_cluster_summary_text(cluster_id, cluster_size, total_devices, top_distinguishing_features: pd.DataFrame, num_features_to_mention=3):
    """Generates a simple natural language summary for a cluster."""
    text = f"Cluster {cluster_id} contains {cluster_size} devices ({cluster_size/total_devices:.1%} of total). "
    if top_distinguishing_features is not None and not top_distinguishing_features.empty:
        text += "It is primarily characterized by: "
        features_to_list = top_distinguishing_features['Feature'].head(num_features_to_mention).tolist()
        if features_to_list:
            text += ", ".join(features_to_list) + "."
        else:
            text += "no single set of strongly distinguishing features based on the current analysis."
    else:
        text += "Further analysis needed to determine its unique characteristics."
    return text

def generate_anomaly_summary_text(device_id, anomaly_score, top_features_comparison: pd.DataFrame, num_features_to_mention=3):
    """Generates a simple natural language summary for an anomalous device (conceptual)."""
    text = f"Device {device_id} is flagged as anomalous with a score of {anomaly_score:.2f}. "
    if top_features_comparison is not None and not top_features_comparison.empty:
        text += "Compared to normal devices, its key differing features include: "
        features_to_list = []
        for idx, row in top_features_comparison.head(num_features_to_mention).iterrows():
            # Check if 'Feature' column exists, otherwise use index name
            feature_name = row.name if 'Feature' not in top_features_comparison.columns else row['Feature']
            direction = "higher" if row.get('Difference (Anomalous - Normal)', 0) > 0 else "lower"
            features_to_list.append(f"{feature_name} (notably {direction})")
        if features_to_list:
            text += "; ".join(features_to_list) + "."
        else:
            text += "its feature differences are not strongly pronounced in the top list."
    else:
        text += "Detailed feature comparison not available."
    return text


if __name__ == '__main__':
    # --- Example Data ---
    rng = np.random.RandomState(42)
    n_dev = 100
    n_feat = 3
    features = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    features.iloc[0:30, 0] += 2  # Group A for feat_0
    features.iloc[30:60, 0] -= 2 # Group B for feat_0
    features.iloc[0:20, 1] += 1.5 # Group C for feat_1 (overlaps with A)

    mock_cluster_labels = pd.Series([0]*(n_dev//3) + [1]*(n_dev//3) + [2]*(n_dev - 2*(n_dev//3)), index=features.index)

    mock_anomaly_labels = pd.Series([1]*n_dev, index=features.index)
    anomalous_indices = features.sample(n=10, random_state=1).index
    mock_anomaly_labels.loc[anomalous_indices] = -1

    print("--- Cluster Feature Summary (Means) ---")
    summary_df, error = get_cluster_feature_summary(features, mock_cluster_labels)
    if error: print(f"Error: {error}")
    else: print(summary_df)

    print("\n--- Feature Importance for Clusters (ANOVA) ---")
    importance_df, error = get_feature_importance_for_clusters_anova(features, mock_cluster_labels)
    if error: print(f"Error: {error}")
    else: print(importance_df)

    print("\n--- Anomalous vs. Normal Feature Comparison ---")
    comparison_df, error = compare_anomalous_vs_normal_features(features, mock_anomaly_labels)
    if error: print(f"Error: {error}")
    else: print(comparison_df)

    print("\n--- Natural Language Summaries (Examples) ---")
    if importance_df is not None:
        print(generate_cluster_summary_text(cluster_id=0, cluster_size=33, total_devices=100, top_distinguishing_features=importance_df))

    if comparison_df is not None and not anomalous_indices.empty:
        # Pass the comparison_df which has feature names as index
        # The generate_anomaly_summary_text function needs to handle this.
        # Re-creating a simple top_features for this test based on comparison_df structure
        temp_comparison_for_summary = comparison_df.reset_index().rename(columns={'index': 'Feature'})
        print(generate_anomaly_summary_text(device_id=anomalous_indices[0], anomaly_score=-1.5, top_features_comparison=temp_comparison_for_summary))

```
