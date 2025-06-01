import pandas as pd
import numpy as np
from scipy.stats import f_oneway # For ANOVA F-value for numeric features across clusters
import streamlit as st # For st.warning in compare_anomalous_vs_normal_features
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split # For splitting if we want to show tree performance
from sklearn.metrics import classification_report # For evaluating surrogate tree


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
    """
    # ... (function content as before) ...
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
    """
    # ... (function content as before) ...
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
    # ... (function content as before) ...
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
    # ... (function content as before) ...
    text = f"Device {device_id} is flagged as anomalous with a score of {anomaly_score:.2f}. "
    if top_features_comparison is not None and not top_features_comparison.empty:
        text += "Compared to normal devices, its key differing features include: "
        features_to_list = []
        for idx, row in top_features_comparison.head(num_features_to_mention).iterrows():
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

# --- New Surrogate Model Explanation Function ---
def explain_anomalies_with_surrogate_model(
    feature_df: pd.DataFrame,
    anomaly_labels: pd.Series,
    anomalous_label_val=-1,
    normal_label_val=1, # Not directly used in fitting but good for context
    max_depth=5,
    random_state=42,
    test_size=0.2
):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(anomaly_labels, pd.Series):
        return None, None, None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or anomaly_labels.empty:
        return None, None, None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(anomaly_labels.index):
        return None, None, None, "Feature DataFrame and anomaly labels must have the same index."
    if anomaly_labels.nunique() < 2:
        return None, None, None, "Anomaly labels must have at least two distinct classes (anomalous and normal)."

    X = feature_df
    y = anomaly_labels.copy() # Ensure we are working with a copy

    try:
        # Map labels to 0 and 1 if they are different (e.g. 1 for normal, -1 for anomaly)
        # DecisionTreeClassifier prefers 0 and 1 or distinct integers.
        # Let's assume anomalous_label_val is the "positive" class for the tree if it's -1 (minority)
        # For class_weight='balanced', the actual label values don't matter as much as their counts.
        # However, for consistency and plot_tree class_names, it's good to be clear.
        # For this function, we'll just use the labels as they are, assuming they are distinct enough.

        surrogate_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced')
        report_dict = None # Initialize report_dict

        if test_size > 0 and test_size < 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            surrogate_tree.fit(X_train, y_train)
            y_pred_test = surrogate_tree.predict(X_test)
            # Ensure target_names are correctly mapped if labels are not 0,1
            # For now, assuming labels are like -1 (anomalous) and 1 (normal)
            # Scikit-learn handles these, but report might be more readable with string names
            unique_labels = sorted(y.unique()) # e.g. [-1, 1]
            target_names_report = [f"Class {label}" for label in unique_labels]
            if anomalous_label_val in unique_labels and normal_label_val in unique_labels:
                 target_names_report = [
                    "Anomalous" if l == anomalous_label_val else "Normal" for l in unique_labels
                 ]


            report_dict = classification_report(y_test, y_pred_test, target_names=target_names_report, output_dict=True, zero_division=0)
        else:
            surrogate_tree.fit(X, y)

        importances = pd.Series(surrogate_tree.feature_importances_, index=X.columns, name="surrogate_tree_feature_importance")
        importances = importances.sort_values(ascending=False)
        return surrogate_tree, importances, report_dict, None
    except Exception as e:
        return None, None, None, f"Error training surrogate decision tree: {e}"


if __name__ == '__main__':
    # ... (existing __main__ content from previous step) ...
    rng = np.random.RandomState(42)
    n_dev = 100
    n_feat = 3
    features = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    features.iloc[0:30, 0] += 2; features.iloc[30:60, 0] -= 2; features.iloc[0:20, 1] += 1.5
    mock_cluster_labels = pd.Series([0]*(n_dev//3) + [1]*(n_dev//3) + [2]*(n_dev - 2*(n_dev//3)), index=features.index)
    mock_anomaly_labels = pd.Series([1]*n_dev, index=features.index)
    anomalous_indices = features.sample(n=10, random_state=1).index
    mock_anomaly_labels.loc[anomalous_indices] = -1

    print("--- Cluster Feature Summary (Means) ---")
    summary_df, error = get_cluster_feature_summary(features, mock_cluster_labels)
    if error:
        print(f"Error: {error}")
    else:
        print(summary_df)
    print("\n--- Feature Importance for Clusters (ANOVA) ---")
    importance_df, error = get_feature_importance_for_clusters_anova(features, mock_cluster_labels)
    if error:
        print(f"Error: {error}")
    else:
        print(importance_df)
    print("\n--- Anomalous vs. Normal Feature Comparison ---")
    comparison_df, error = compare_anomalous_vs_normal_features(features, mock_anomaly_labels)
    if error:
        print(f"Error: {error}")
    else:
        print(comparison_df)
    print("\n--- Natural Language Summaries (Examples) ---")
    if importance_df is not None: print(generate_cluster_summary_text(cluster_id=0, cluster_size=33, total_devices=100, top_distinguishing_features=importance_df))
    if comparison_df is not None and not anomalous_indices.empty:
        temp_comparison_for_summary = comparison_df.reset_index().rename(columns={'index': 'Feature'})
        print(generate_anomaly_summary_text(device_id=anomalous_indices[0], anomaly_score=-1.5, top_features_comparison=temp_comparison_for_summary))

    # --- Example for Surrogate Model Anomaly Explanation ---
    print("\n--- Surrogate Model Anomaly Explanation ---")
    # Using 'features' and 'mock_anomaly_labels' from above
    if 'features' in locals() and 'mock_anomaly_labels' in locals() and not features.empty:
        # Ensure no NaNs in the feature set for surrogate model
        features_cleaned_for_surrogate = features.dropna()
        labels_for_surrogate = mock_anomaly_labels.loc[features_cleaned_for_surrogate.index]

        if not features_cleaned_for_surrogate.empty and labels_for_surrogate.nunique() >=2:
            tree_model, tree_importances, tree_report, error_tree = explain_anomalies_with_surrogate_model(
                features_cleaned_for_surrogate, labels_for_surrogate, max_depth=3, test_size=0 # test_size=0 for no split
            )
            if error_tree:
                print(f"Error training surrogate tree: {error_tree}")
            else:
                print("Surrogate Tree Feature Importances:\n", tree_importances.head())
                if tree_report:
                    print("Surrogate Tree Classification Report (on test set if test_size > 0):\n", pd.DataFrame(tree_report).transpose())
                print("Surrogate tree model trained (plotting not shown here).")
        else:
            print("Skipping surrogate model example: not enough data or classes after cleaning.")
    else:
        print("Skipping surrogate model example: features or labels not available.")
