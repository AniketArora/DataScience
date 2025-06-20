import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree # plot_tree is not used in this module directly
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)


@st.cache_data
def get_cluster_feature_summary(feature_df: pd.DataFrame, cluster_labels: pd.Series):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(cluster_labels, pd.Series):
        msg = "Inputs must be pandas DataFrame and Series."
        logger.warning(msg)
        return None, msg
    if feature_df.empty or cluster_labels.empty:
        msg = "Input DataFrame or labels Series is empty."
        logger.warning(msg)
        return None, msg
    if not feature_df.index.equals(cluster_labels.index):
        msg = "Feature DataFrame and cluster labels must have the same index."
        logger.warning(msg)
        return None, msg
    try:
        if cluster_labels.isnull().any():
            valid_indices = cluster_labels.dropna().index
            feature_df_cleaned = feature_df.loc[valid_indices]
            cluster_labels_cleaned = cluster_labels.loc[valid_indices]
            if feature_df_cleaned.empty:
                msg = "No valid data after removing NaN cluster labels."
                logger.warning(msg)
                return None, msg
        else:
            feature_df_cleaned = feature_df; cluster_labels_cleaned = cluster_labels
        numeric_feature_df = feature_df_cleaned.select_dtypes(include=np.number)
        if numeric_feature_df.empty:
            msg = "No numeric features found."
            logger.warning(msg)
            return None, msg
        summary_df = numeric_feature_df.groupby(cluster_labels_cleaned).mean()
        return summary_df, None
    except Exception as e:
        logger.error("Error calculating cluster feature summary: %s", e, exc_info=True)
        return None, f"Error calculating cluster feature summary: {e}"

@st.cache_data
def get_feature_importance_for_clusters_anova(feature_df: pd.DataFrame, cluster_labels: pd.Series):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(cluster_labels, pd.Series):
        msg = "Inputs must be pandas DataFrame and Series."
        logger.warning(msg)
        return None, msg
    if feature_df.empty or cluster_labels.empty:
        msg = "Input DataFrame or labels Series is empty."
        logger.warning(msg)
        return None, msg
    if not feature_df.index.equals(cluster_labels.index):
        msg = "Indices must match."
        logger.warning(msg)
        return None, msg
    valid_indices = cluster_labels.dropna().index
    feature_df_cleaned = feature_df.loc[valid_indices]; cluster_labels_cleaned = cluster_labels.loc[valid_indices]
    if cluster_labels_cleaned.nunique() < 2:
        msg = "Need at least two clusters for ANOVA."
        logger.warning(msg)
        return None, msg
    if feature_df_cleaned.empty:
        msg = "Feature DataFrame empty after NaN label handling."
        logger.warning(msg)
        return None, msg
    numeric_feature_df = feature_df_cleaned.select_dtypes(include=np.number)
    if numeric_feature_df.empty:
        msg = "No numeric features for ANOVA."
        logger.warning(msg)
        return None, msg
    f_values, p_values, features = [], [], []
    for feature_name in numeric_feature_df.columns:
        try:
            groups = [numeric_feature_df[feature_name][cluster_labels_cleaned == c] for c in cluster_labels_cleaned.unique()]
            groups = [g for g in groups if len(g.dropna()) > 1 and g.dropna().var() > 1e-6] # Ensure variance > 0 and enough samples
            if len(groups) < 2: f_stat, p_val = np.nan, np.nan
            else: f_stat, p_val = f_oneway(*groups)
            f_values.append(f_stat); p_values.append(p_val); features.append(feature_name)
        except Exception as e_anova: # More specific error logging
            logger.warning("ANOVA calculation failed for feature '%s': %s", feature_name, e_anova)
            f_values.append(np.nan); p_values.append(np.nan); features.append(feature_name)
    if not features:
        msg = "No features processed by ANOVA."
        logger.warning(msg)
        return None, msg
    importance_df = pd.DataFrame({'Feature': features, 'F-Value': f_values, 'P-Value': p_values})
    return importance_df.sort_values(by='F-Value', ascending=False).reset_index(drop=True), None

@st.cache_data
def compare_anomalous_vs_normal_features(feature_df: pd.DataFrame, anomaly_labels: pd.Series, anomalous_label_val=-1):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(anomaly_labels, pd.Series):
        msg = "Inputs must be pandas DataFrame and Series."
        logger.warning(msg)
        return None, msg
    if feature_df.empty or anomaly_labels.empty:
        msg = "Input DataFrame or labels Series is empty."
        logger.warning(msg)
        return None, msg
    if not feature_df.index.equals(anomaly_labels.index):
        msg = "Indices must match."
        logger.warning(msg)
        return None, msg
    try:
        numeric_feature_df = feature_df.select_dtypes(include=np.number)
        if numeric_feature_df.empty:
            msg = "No numeric features."
            logger.warning(msg)
            return None, msg
        normal_mask = (anomaly_labels != anomalous_label_val); anomalous_mask = (anomaly_labels == anomalous_label_val)
        if not normal_mask.any() and not anomalous_mask.any():
            msg = "No normal or anomalous devices."
            logger.warning(msg)
            return None, msg
        if not normal_mask.any():
            msg = "No 'normal' devices found for comparison."
            logger.warning(msg)
            anomalous_features_mean = numeric_feature_df[anomalous_mask].mean().rename("Anomalous_Mean")
            comparison_df = pd.DataFrame(anomalous_features_mean); comparison_df['Normal_Mean'] = np.nan
            comparison_df['Difference (Anomalous - Normal)'] = np.nan; comparison_df['Relative_Difference (%)'] = np.nan
            return comparison_df, msg
        if not anomalous_mask.any():
            msg = "No 'anomalous' devices found for comparison."
            logger.warning(msg)
            normal_features_mean = numeric_feature_df[normal_mask].mean().rename("Normal_Mean")
            comparison_df = pd.DataFrame(normal_features_mean); comparison_df['Anomalous_Mean'] = np.nan
            comparison_df['Difference (Anomalous - Normal)'] = np.nan; comparison_df['Relative_Difference (%)'] = np.nan
            return comparison_df, msg
        normal_features_mean = numeric_feature_df[normal_mask].mean().rename("Normal_Mean")
        anomalous_features_mean = numeric_feature_df[anomalous_mask].mean().rename("Anomalous_Mean")
        comparison_df = pd.concat([normal_features_mean, anomalous_features_mean], axis=1)
        comparison_df['Difference (Anomalous - Normal)'] = comparison_df['Anomalous_Mean'] - comparison_df['Normal_Mean']
        comparison_df['Relative_Difference (%)'] = (comparison_df['Difference (Anomalous - Normal)'] / comparison_df['Normal_Mean'].abs().replace(0, np.nan)) * 100
        comparison_df['Relative_Difference (%)'] = comparison_df['Relative_Difference (%)'].replace([np.inf, -np.inf], np.nan)
        return comparison_df.sort_values(by='Difference (Anomalous - Normal)', key=abs, ascending=False), None
    except Exception as e:
        logger.error("Error comparing anomalous vs normal features: %s", e, exc_info=True)
        return None, f"Error comparing features: {e}"

@st.cache_data
def explain_anomalies_with_surrogate_model(
    feature_df: pd.DataFrame, anomaly_labels: pd.Series, anomalous_label_val=-1, normal_label_val=1,
    max_depth=5, random_state=42, test_size=0.2 ):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(anomaly_labels, pd.Series):
        msg = "Inputs must be pandas DataFrame and Series."
        logger.warning(msg)
        return None, None, None, msg
    if feature_df.empty or anomaly_labels.empty:
        msg = "Input DataFrame or labels Series is empty."
        logger.warning(msg)
        return None, None, None, msg
    if not feature_df.index.equals(anomaly_labels.index):
        msg = "Indices must match."
        logger.warning(msg)
        return None, None, None, msg
    if anomaly_labels.nunique() < 2:
        msg = "Need at least two distinct classes for surrogate model training."
        logger.warning(msg)
        return None, None, None, msg
    X = feature_df; y = anomaly_labels.copy()
    try:
        surrogate_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced')
        report_dict = None
        if test_size > 0 and test_size < 1 and len(X) * test_size >= 2 and y.nunique() >=2 :
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            except ValueError as ve: # Stratify can fail if one class has too few samples
                logger.warning("Stratified split failed for surrogate model (error: %s), falling back to non-stratified split.", ve)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            surrogate_tree.fit(X_train, y_train); y_pred_test = surrogate_tree.predict(X_test)
            unique_labels_y = sorted(y.unique()); unique_labels_pred = sorted(pd.Series(y_pred_test).unique())
            present_labels = sorted(list(set(unique_labels_y).union(set(unique_labels_pred))))
            class_map = {normal_label_val: "Normal", anomalous_label_val: "Anomalous"}
            target_names_report = [class_map.get(l, f"Class {l}") for l in present_labels]
            report_dict = classification_report(y_test, y_pred_test, labels=present_labels, target_names=target_names_report, output_dict=True, zero_division=0)
        else: surrogate_tree.fit(X, y)
        importances = pd.Series(surrogate_tree.feature_importances_, index=X.columns, name="surrogate_tree_feature_importance").sort_values(ascending=False)
        return surrogate_tree, importances, report_dict, None
    except Exception as e:
        logger.error("Error training surrogate tree: %s", e, exc_info=True)
        return None, None, None, f"Error training surrogate tree: {e}"

@st.cache_data
def generate_cluster_summary_text(
    cluster_id, cluster_size, total_devices,
    cluster_mean_features_for_this_cluster: pd.Series,
    overall_mean_features: pd.Series,
    num_features_to_mention=3
):
    text = f"Cluster {cluster_id} contains {cluster_size} devices ({cluster_size/total_devices:.1%} of total). "
    if cluster_mean_features_for_this_cluster is None or cluster_mean_features_for_this_cluster.empty or \
       overall_mean_features is None or overall_mean_features.empty:
        text += "Feature mean data not available for detailed characterization."
        return text
    common_features = cluster_mean_features_for_this_cluster.index.intersection(overall_mean_features.index)
    if not common_features.empty:
        cluster_means_common = cluster_mean_features_for_this_cluster[common_features]
        overall_means_common = overall_mean_features[common_features]
        epsilon = 1e-9
        relative_difference_pct = ((cluster_means_common - overall_means_common) / (overall_means_common.abs() + epsilon)) * 100
        diff_df = pd.DataFrame({
            'feature': common_features, 'cluster_mean': cluster_means_common,
            'overall_mean': overall_means_common, 'abs_rel_diff_pct': relative_difference_pct.abs(),
            'rel_diff_pct': relative_difference_pct
        })
        distinguishing_features_sorted = diff_df.sort_values(by='abs_rel_diff_pct', ascending=False)
        top_features_details = []
        for _, row in distinguishing_features_sorted.head(num_features_to_mention).iterrows():
            if row['abs_rel_diff_pct'] > 20:
                direction = "higher" if row['rel_diff_pct'] > 0 else "lower"
                top_features_details.append(f"{row['feature']} (value: {row['cluster_mean']:.2f}, notably {direction} than overall mean {row['overall_mean']:.2f})")
            elif row['abs_rel_diff_pct'] > 5:
                direction = "higher" if row['rel_diff_pct'] > 0 else "lower"
                top_features_details.append(f"{row['feature']} (value: {row['cluster_mean']:.2f}, slightly {direction} than overall mean {row['overall_mean']:.2f})")
        if top_features_details: text += "It is characterized by: " + "; ".join(top_features_details) + "."
        else: text += "Its characteristics are broadly similar to average."
    else: text += "Could not compare cluster means."
    return text

@st.cache_data
def generate_anomaly_summary_text(
    device_id, anomaly_score, top_features_comparison: pd.DataFrame = None,
    surrogate_tree_importances: pd.Series = None, num_features_to_mention=3 ):
    text = f"Device {device_id} is flagged as anomalous with a score of {anomaly_score:.2f}. "
    if surrogate_tree_importances is not None and not surrogate_tree_importances.empty:
        text += "Surrogate tree suggests key factors: "
        top_tree_features = surrogate_tree_importances.head(num_features_to_mention)
        feature_list = [f"{idx} (importance: {val:.2f})" for idx, val in top_tree_features.items() if val > 0.01]
        if feature_list: text += "; ".join(feature_list) + "."
        else: text += "surrogate tree did not highlight strong individual feature contributions."
    elif top_features_comparison is not None and not top_features_comparison.empty:
        text += "Compared to normal devices, its differing features include: "
        features_to_list = []
        for feature_name, row in top_features_comparison.head(num_features_to_mention).iterrows():
            direction = "notably higher" if row.get('Difference (Anomalous - Normal)', 0) > 0 else \
                        "notably lower" if row.get('Difference (Anomalous - Normal)', 0) < 0 else "similar"
            features_to_list.append(f"{feature_name} (value: {row['Anomalous_Mean']:.2f} vs normal mean: {row['Normal_Mean']:.2f})")
        if features_to_list: text += "; ".join(features_to_list) + "."
        else: text += "feature differences not strongly pronounced."
    else: text += "Detailed feature contribution analysis not available."
    return text

# --- New Event Correlation Function ---
@st.cache_data
def analyze_event_correlations(
    all_device_features_df_with_event_features: pd.DataFrame,
    result_labels: pd.Series,
    event_feature_prefix="evt_count_"
):
    if not isinstance(all_device_features_df_with_event_features, pd.DataFrame) or \
       not isinstance(result_labels, pd.Series):
        msg = "Inputs must be pandas DataFrame and Series."
        logger.warning(msg)
        return None, msg
    if all_device_features_df_with_event_features.empty or result_labels.empty:
        msg = "Input DataFrame or labels Series is empty."
        logger.warning(msg)
        return None, msg

    common_index = all_device_features_df_with_event_features.index.intersection(result_labels.index)
    if common_index.empty:
        msg = "Feature DataFrame and result labels must have common indices."
        logger.warning(msg)
        return None, msg
    df_aligned = all_device_features_df_with_event_features.loc[common_index]
    labels_aligned = result_labels.loc[common_index]

    event_count_cols = [col for col in df_aligned.columns if col.startswith(event_feature_prefix)]
    if not event_count_cols:
        msg = f"No event count features found with prefix '{event_feature_prefix}'."
        logger.warning(msg)
        return None, msg

    try:
        event_analysis_df = df_aligned[event_count_cols].groupby(labels_aligned).mean()
        overall_mean_events = df_aligned[event_count_cols].mean()
        overall_mean_events.name = "Overall_Mean"
        event_analysis_df = pd.concat([event_analysis_df, overall_mean_events.to_frame().T])

        new_index_names = {}
        for label_val in event_analysis_df.index:
            if label_val == "Overall_Mean": new_index_names[label_val] = "Overall_Mean_Events"
            elif label_val == -1: new_index_names[label_val] = "Anomalous_Mean_Events"
            elif label_val == 1 and set(labels_aligned.unique()).issubset({-1,1,0}):
                 new_index_names[label_val] = "Normal_Mean_Events"
            else: new_index_names[label_val] = f"Cluster_{label_val}_Mean_Events"
        event_analysis_df = event_analysis_df.rename(index=new_index_names)

        return event_analysis_df.T, None
    except Exception as e:
        logger.error("Error during event correlation analysis: %s", e, exc_info=True)
        return None, f"Error during event correlation analysis: {e}"


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    n_dev = 100; n_feat = 3
    features = pd.DataFrame(rng.rand(n_dev, n_feat), columns=[f'feat_{j}' for j in range(n_feat)], index=[f'dev_{i}' for i in range(n_dev)])
    features.iloc[0:30, 0] += 2; features.iloc[30:60, 0] -= 2; features.iloc[0:20, 1] += 1.5
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
    if summary_df is not None and importance_df is not None:
        overall_means_for_summary = features.mean()
        print(generate_cluster_summary_text(0, (mock_cluster_labels == 0).sum(), len(features), summary_df.loc[0], overall_means_for_summary))
    if comparison_df is not None and not anomalous_indices.empty:
        print(generate_anomaly_summary_text(device_id=anomalous_indices[0], anomaly_score=-1.5, top_features_comparison=comparison_df))

    print("\n--- Surrogate Model Anomaly Explanation ---")
    if not features.empty:
        features_cleaned_for_surrogate = features.dropna()
        if not features_cleaned_for_surrogate.empty:
            labels_for_surrogate = mock_anomaly_labels.loc[features_cleaned_for_surrogate.index]
            if labels_for_surrogate.nunique() >=2:
                tree_model, tree_importances, tree_report, error_tree = explain_anomalies_with_surrogate_model(
                    features_cleaned_for_surrogate, labels_for_surrogate, max_depth=3, test_size=0.0
                )
                if error_tree: print(f"Error training surrogate tree: {error_tree}")
                else:
                    print("Surrogate Tree Feature Importances:\n", tree_importances.head())
                    if tree_report: print("Surrogate Tree Report:\n", pd.DataFrame(tree_report).transpose())
                    print("Surrogate tree model trained.")
            else: print("Skipping surrogate: Not enough distinct classes after cleaning.")
        else: print("Skipping surrogate: Features empty after NaN drop.")
    else: print("Skipping surrogate: 'features' not available.")

    print("\n--- Event Correlation Analysis ---")
    if 'features' in locals() and 'mock_anomaly_labels' in locals():
        mock_features_with_events = features.copy()
        mock_features_with_events['evt_count_Error_X'] = rng.randint(0, 5, size=len(features))
        mock_features_with_events['evt_count_Warning_Y'] = rng.randint(0, 10, size=len(features))
        anomalous_idx_for_event_test = mock_anomaly_labels[mock_anomaly_labels == -1].index
        mock_features_with_events.loc[anomalous_idx_for_event_test, 'evt_count_Error_X'] += 3
        event_corr_df, error_ec = analyze_event_correlations(mock_features_with_events, mock_anomaly_labels)
        if error_ec: print(f"Error: {error_ec}")
        else: print("Event Correlation with Anomaly Labels (Mean Event Counts):\n", event_corr_df)
        if 'mock_cluster_labels' in locals():
           event_corr_clusters_df, error_ecc = analyze_event_correlations(mock_features_with_events, mock_cluster_labels)
           if error_ecc: print(f"Error: {error_ecc}")
           else: print("\nEvent Correlation with Cluster Labels (Mean Event Counts):\n", event_corr_clusters_df)
