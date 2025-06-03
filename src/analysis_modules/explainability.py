import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def get_cluster_feature_summary(feature_df: pd.DataFrame, cluster_labels: pd.Series):
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
            if feature_df_cleaned.empty: return None, "No valid data after removing NaN cluster labels."
        else:
            feature_df_cleaned = feature_df; cluster_labels_cleaned = cluster_labels
        numeric_feature_df = feature_df_cleaned.select_dtypes(include=np.number)
        if numeric_feature_df.empty: return None, "No numeric features found."
        summary_df = numeric_feature_df.groupby(cluster_labels_cleaned).mean()
        return summary_df, None
    except Exception as e: return None, f"Error calculating cluster feature summary: {e}"

def get_feature_importance_for_clusters_anova(feature_df: pd.DataFrame, cluster_labels: pd.Series):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(cluster_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or cluster_labels.empty: return None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(cluster_labels.index): return None, "Indices must match."
    valid_indices = cluster_labels.dropna().index
    feature_df_cleaned = feature_df.loc[valid_indices]; cluster_labels_cleaned = cluster_labels.loc[valid_indices]
    if cluster_labels_cleaned.nunique() < 2: return None, "Need at least two clusters."
    if feature_df_cleaned.empty: return None, "Feature DataFrame empty after NaN label handling."
    numeric_feature_df = feature_df_cleaned.select_dtypes(include=np.number)
    if numeric_feature_df.empty: return None, "No numeric features for ANOVA."
    f_values, p_values, features = [], [], []
    for feature_name in numeric_feature_df.columns:
        try:
            groups = [numeric_feature_df[feature_name][cluster_labels_cleaned == c] for c in cluster_labels_cleaned.unique()]
            groups = [g for g in groups if len(g.dropna()) > 1 and g.dropna().var() > 1e-6]
            if len(groups) < 2: f_stat, p_val = np.nan, np.nan
            else: f_stat, p_val = f_oneway(*groups)
            f_values.append(f_stat); p_values.append(p_val); features.append(feature_name)
        except Exception: f_values.append(np.nan); p_values.append(np.nan); features.append(feature_name)
    if not features: return None, "No features processed."
    importance_df = pd.DataFrame({'Feature': features, 'F-Value': f_values, 'P-Value': p_values})
    return importance_df.sort_values(by='F-Value', ascending=False).reset_index(drop=True), None

def compare_anomalous_vs_normal_features(feature_df: pd.DataFrame, anomaly_labels: pd.Series, anomalous_label_val=-1):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(anomaly_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or anomaly_labels.empty: return None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(anomaly_labels.index): return None, "Indices must match."
    try:
        numeric_feature_df = feature_df.select_dtypes(include=np.number)
        if numeric_feature_df.empty: return None, "No numeric features."
        normal_mask = (anomaly_labels != anomalous_label_val); anomalous_mask = (anomaly_labels == anomalous_label_val)
        if not normal_mask.any() and not anomalous_mask.any(): return None, "No normal or anomalous devices."
        if not normal_mask.any():
            # This st.warning will not work here as this module should not directly call Streamlit UI elements.
            # The warning/message should be handled by the caller in main.py.
            # For now, returning a specific message or modified DataFrame.
            anomalous_features_mean = numeric_feature_df[anomalous_mask].mean().rename("Anomalous_Mean")
            comparison_df = pd.DataFrame(anomalous_features_mean); comparison_df['Normal_Mean'] = np.nan
            comparison_df['Difference (Anomalous - Normal)'] = np.nan; comparison_df['Relative_Difference (%)'] = np.nan
            return comparison_df, "No 'normal' devices found for comparison."
        if not anomalous_mask.any():
            normal_features_mean = numeric_feature_df[normal_mask].mean().rename("Normal_Mean")
            comparison_df = pd.DataFrame(normal_features_mean); comparison_df['Anomalous_Mean'] = np.nan
            comparison_df['Difference (Anomalous - Normal)'] = np.nan; comparison_df['Relative_Difference (%)'] = np.nan
            return comparison_df, "No 'anomalous' devices found for comparison."
        normal_features_mean = numeric_feature_df[normal_mask].mean().rename("Normal_Mean")
        anomalous_features_mean = numeric_feature_df[anomalous_mask].mean().rename("Anomalous_Mean")
        comparison_df = pd.concat([normal_features_mean, anomalous_features_mean], axis=1)
        comparison_df['Difference (Anomalous - Normal)'] = comparison_df['Anomalous_Mean'] - comparison_df['Normal_Mean']
        comparison_df['Relative_Difference (%)'] = (comparison_df['Difference (Anomalous - Normal)'] / comparison_df['Normal_Mean'].abs().replace(0, np.nan)) * 100
        comparison_df['Relative_Difference (%)'] = comparison_df['Relative_Difference (%)'].replace([np.inf, -np.inf], np.nan)
        return comparison_df.sort_values(by='Difference (Anomalous - Normal)', key=abs, ascending=False), None
    except Exception as e: return None, f"Error comparing features: {e}"

def explain_anomalies_with_surrogate_model(
    feature_df: pd.DataFrame, anomaly_labels: pd.Series, anomalous_label_val=-1, normal_label_val=1,
    max_depth=5, random_state=42, test_size=0.2 ):
    if not isinstance(feature_df, pd.DataFrame) or not isinstance(anomaly_labels, pd.Series):
        return None, None, None, "Inputs must be pandas DataFrame and Series."
    if feature_df.empty or anomaly_labels.empty: return None, None, None, "Input DataFrame or labels Series is empty."
    if not feature_df.index.equals(anomaly_labels.index): return None, None, None, "Indices must match."
    if anomaly_labels.nunique() < 2: return None, None, None, "Need at least two distinct classes."
    X = feature_df; y = anomaly_labels.copy()
    try:
        surrogate_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced')
        report_dict = None
        if test_size > 0 and test_size < 1 and len(X) * test_size >= 2 and y.nunique() >=2 : # Ensure test set is meaningful
             # Stratify might fail if a class has only 1 member, common in anomaly detection.
             # Try-except for stratify, or ensure enough samples per class.
             # For simplicity, if stratify fails, do non-stratified split.
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            surrogate_tree.fit(X_train, y_train); y_pred_test = surrogate_tree.predict(X_test)
            unique_labels_y = sorted(y.unique()); unique_labels_pred = sorted(pd.Series(y_pred_test).unique())
            # Ensure labels used for report match what's in y_test and y_pred_test
            present_labels = sorted(list(set(unique_labels_y).union(set(unique_labels_pred))))

            target_names_report = [f"Class {label}" for label in present_labels]
            # A more robust way to define target_names based on actual values and user intent:
            class_map = {normal_label_val: "Normal", anomalous_label_val: "Anomalous"}
            target_names_report = [class_map.get(l, f"Class {l}") for l in present_labels]

            report_dict = classification_report(y_test, y_pred_test, labels=present_labels, target_names=target_names_report, output_dict=True, zero_division=0)
        else: surrogate_tree.fit(X, y)
        importances = pd.Series(surrogate_tree.feature_importances_, index=X.columns, name="surrogate_tree_feature_importance").sort_values(ascending=False)
        return surrogate_tree, importances, report_dict, None
    except Exception as e: return None, None, None, f"Error training surrogate tree: {e}"

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
def analyze_event_correlations(
    all_device_features_df_with_event_features: pd.DataFrame,
    result_labels: pd.Series,
    event_feature_prefix="evt_count_"
):
    if not isinstance(all_device_features_df_with_event_features, pd.DataFrame) or \
       not isinstance(result_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if all_device_features_df_with_event_features.empty or result_labels.empty:
        return None, "Input DataFrame or labels Series is empty."

    common_index = all_device_features_df_with_event_features.index.intersection(result_labels.index)
    if common_index.empty:
        return None, "Feature DataFrame and result labels must have common indices."
    df_aligned = all_device_features_df_with_event_features.loc[common_index]
    labels_aligned = result_labels.loc[common_index]

    event_count_cols = [col for col in df_aligned.columns if col.startswith(event_feature_prefix)]
    if not event_count_cols:
        return None, f"No event count features found with prefix '{event_feature_prefix}'."

    try:
        event_analysis_df = df_aligned[event_count_cols].groupby(labels_aligned).mean()
        overall_mean_events = df_aligned[event_count_cols].mean()
        overall_mean_events.name = "Overall_Mean"
        # Use pd.concat to join, ensuring consistent columns even if some groups are missing
        event_analysis_df = pd.concat([event_analysis_df, overall_mean_events.to_frame().T])

        # Make index names more readable
        new_index_names = {}
        for label_val in event_analysis_df.index:
            if label_val == "Overall_Mean": new_index_names[label_val] = "Overall_Mean_Events"
            elif label_val == -1: new_index_names[label_val] = "Anomalous_Mean_Events"
            elif label_val == 1 and set(labels_aligned.unique()).issubset({-1,1,0}): # Check if it's likely anomaly labels (0 could be uncertain)
                 new_index_names[label_val] = "Normal_Mean_Events"
            else: new_index_names[label_val] = f"Cluster_{label_val}_Mean_Events"
        event_analysis_df = event_analysis_df.rename(index=new_index_names)

        return event_analysis_df.T, None
    except Exception as e:
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


# --- New Significant Event Types Function ---
from scipy.stats import chi2_contingency, fisher_exact

def analyze_significant_event_types(
    all_device_features_df_with_event_features: pd.DataFrame,
    result_labels: pd.Series,
    event_feature_prefix: str,
    at_risk_label: any,
    baseline_label: any = None
):
    """
    Analyzes event types to identify those significantly associated with an at-risk group compared to a baseline group.

    Args:
        all_device_features_df_with_event_features (pd.DataFrame): DataFrame containing device features,
                                                                  including event count columns.
        result_labels (pd.Series): Series indicating the group label for each device (e.g., cluster ID, anomaly label).
        event_feature_prefix (str): Prefix to identify event count columns (e.g., "evt_count_").
        at_risk_label (any): The label value identifying the 'at-risk' group.
        baseline_label (any, optional): The label value identifying the 'baseline' group.
                                        If None, baseline is all devices not in the 'at-risk' group.

    Returns:
        pd.DataFrame: DataFrame with analysis results (lift, p-value, counts) for each event type,
                      sorted by lift (descending).
        str: Error message if an issue occurred, None otherwise.
    """
    if not isinstance(all_device_features_df_with_event_features, pd.DataFrame) or \
       not isinstance(result_labels, pd.Series):
        return None, "Inputs must be pandas DataFrame and Series."
    if all_device_features_df_with_event_features.empty or result_labels.empty:
        return None, "Input DataFrame or labels Series is empty."

    common_index = all_device_features_df_with_event_features.index.intersection(result_labels.index)
    if common_index.empty:
        return None, "Feature DataFrame and result labels must have common indices."

    df_aligned = all_device_features_df_with_event_features.loc[common_index]
    labels_aligned = result_labels.loc[common_index]

    event_count_cols = [col for col in df_aligned.columns if col.startswith(event_feature_prefix)]
    if not event_count_cols:
        return None, f"No event count features found with prefix '{event_feature_prefix}'."

    at_risk_mask = (labels_aligned == at_risk_label)
    if not at_risk_mask.any():
        return None, f"No devices found for the at-risk label '{at_risk_label}'."

    if baseline_label is not None:
        baseline_mask = (labels_aligned == baseline_label)
        if not baseline_mask.any():
            return None, f"No devices found for the baseline label '{baseline_label}'."
    else:
        baseline_mask = ~at_risk_mask # All devices not in at-risk group
        if not baseline_mask.any():
            return None, "No devices found for the baseline group (all non at-risk devices)."


    results = []

    for event_col in event_count_cols:
        event_type_name = event_col.replace(event_feature_prefix, "")

        # Devices in at-risk group where event occurred (count > 0)
        at_risk_event_occurred = df_aligned.loc[at_risk_mask, event_col] > 0
        a = at_risk_event_occurred.sum()
        # Total devices in at-risk group
        n_at_risk = at_risk_mask.sum()
        b = n_at_risk - a

        # Devices in baseline group where event occurred (count > 0)
        baseline_event_occurred = df_aligned.loc[baseline_mask, event_col] > 0
        c = baseline_event_occurred.sum()
        # Total devices in baseline group
        n_baseline = baseline_mask.sum()
        d = n_baseline - c

        # Contingency table
        #        Event Occurred | Event Did Not Occur
        # At-Risk |      a       |        b
        # Baseline|      c       |        d
        contingency_table = np.array([[a, b], [c, d]])

        # Calculate Lift
        # Lift = (a / (a+b)) / (c / (c+d))
        # (Incidence of event in at-risk group) / (Incidence of event in baseline group)
        incidence_at_risk = a / n_at_risk if n_at_risk > 0 else 0
        incidence_baseline = c / n_baseline if n_baseline > 0 else 0

        if incidence_baseline == 0: # Avoid division by zero
            lift = np.inf if incidence_at_risk > 0 else np.nan # Or some other indicator like -1 or a large number
        else:
            lift = incidence_at_risk / incidence_baseline

        p_value = np.nan
        chi2_note = ""
        try:
            # Check for low expected frequencies, which might make chi2 unreliable
            expected_freq_too_low = False
            if np.any(contingency_table < 5): # A common heuristic
                 chi2_note = "Fisher's exact test might be more appropriate due to low counts in contingency table."

            # Ensure there's enough data to run chi2 (e.g., non-zero rows/columns)
            if contingency_table.sum() > 0 and np.all(contingency_table.sum(axis=0) > 0) and np.all(contingency_table.sum(axis=1) > 0):
                chi2, p, dof, expected = chi2_contingency(contingency_table, correction=False) # correction=False is often recommended
                p_value = p
                if np.any(expected < 5) and not chi2_note: # Check expected frequencies from chi2_contingency itself
                    chi2_note = "Fisher's exact test might be more appropriate due to low expected frequencies (<5)."
            else:
                chi2_note = "Skipped chi2: not enough data (e.g. all zeros in a row/column)."

        except ValueError as ve: # e.g. if sum of some row/col is zero
            p_value = np.nan # Could not calculate
            chi2_note = f"Chi2 calculation error: {ve}"


        results.append({
            "event_type": event_type_name,
            "lift": lift,
            "p_value": p_value,
            "at_risk_group_event_count": a, # num at-risk devices where event occurred
            "baseline_group_event_count": c, # num baseline devices where event occurred
            "at_risk_group_total_devices": n_at_risk,
            "baseline_group_total_devices": n_baseline,
            "chi2_contingency_note": chi2_note
        })

    if not results:
        return pd.DataFrame(), "No event types processed or no results generated."

    results_df = pd.DataFrame(results)
    # Sort by lift (descending) then p-value (ascending) as a secondary sort
    results_df = results_df.sort_values(by=['lift', 'p_value'], ascending=[False, True]).reset_index(drop=True)

    return results_df, None


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
        mock_features_with_events['evt_count_Info_Z'] = rng.randint(0, 2, size=len(features)) # Low count event
        anomalous_idx_for_event_test = mock_anomaly_labels[mock_anomaly_labels == -1].index
        normal_idx_for_event_test = mock_anomaly_labels[mock_anomaly_labels == 1].index

        # Make Error_X more prevalent in anomalous
        mock_features_with_events.loc[anomalous_idx_for_event_test, 'evt_count_Error_X'] = rng.randint(1, 5, size=len(anomalous_idx_for_event_test))
        mock_features_with_events.loc[normal_idx_for_event_test, 'evt_count_Error_X'] = rng.randint(0, 2, size=len(normal_idx_for_event_test))
        # Make Info_Z more prevalent in normal (baseline)
        mock_features_with_events.loc[normal_idx_for_event_test, 'evt_count_Info_Z'] = 1


        event_corr_df, error_ec = analyze_event_correlations(mock_features_with_events, mock_anomaly_labels)
        if error_ec: print(f"Error: {error_ec}")
        else: print("Event Correlation with Anomaly Labels (Mean Event Counts):\n", event_corr_df)

        if 'mock_cluster_labels' in locals():
           event_corr_clusters_df, error_ecc = analyze_event_correlations(mock_features_with_events, mock_cluster_labels)
           if error_ecc: print(f"Error: {error_ecc}")
           else: print("\nEvent Correlation with Cluster Labels (Mean Event Counts):\n", event_corr_clusters_df)

    print("\n--- Significant Event Type Analysis (Lift & Chi2) ---")
    if 'mock_features_with_events' in locals() and 'mock_anomaly_labels' in locals():
        # Test case 1: Anomalous (-1) vs Baseline (all others, which is 1 in this case)
        sig_events_df, error_sig = analyze_significant_event_types(
            mock_features_with_events,
            mock_anomaly_labels,
            event_feature_prefix="evt_count_",
            at_risk_label=-1
        )
        if error_sig:
            print(f"Error in significant event analysis (anomalous vs all others): {error_sig}")
        else:
            print("Significant Events (Anomalous [-1] vs. Rest [1]):\n", sig_events_df)

        # Test case 2: Anomalous (-1) vs specific Baseline (1)
        # This should yield similar results to above if only -1 and 1 are present.
        sig_events_df_specific_baseline, error_sig_sb = analyze_significant_event_types(
            mock_features_with_events,
            mock_anomaly_labels, # Using anomaly labels that have -1 and 1
            event_feature_prefix="evt_count_",
            at_risk_label=-1,
            baseline_label=1
        )
        if error_sig_sb:
            print(f"Error in significant event analysis (anomalous vs specific baseline): {error_sig_sb}")
        else:
            print("\nSignificant Events (Anomalous [-1] vs. Baseline [1]):\n", sig_events_df_specific_baseline)

        # Test case 3: Using cluster labels (e.g. cluster 0 vs cluster 1)
        if 'mock_cluster_labels' in locals() and mock_cluster_labels.nunique() > 1:
            # Make one event type more prevalent in cluster 0
            cluster_0_indices = mock_cluster_labels[mock_cluster_labels == 0].index
            mock_features_with_events.loc[cluster_0_indices, 'evt_count_Warning_Y'] += 5

            sig_events_clusters_df, error_sig_c = analyze_significant_event_types(
                mock_features_with_events,
                mock_cluster_labels,
                event_feature_prefix="evt_count_",
                at_risk_label=0, # Cluster 0 as at-risk
                baseline_label=1  # Cluster 1 as baseline
            )
            if error_sig_c:
                print(f"Error in significant event analysis (cluster 0 vs cluster 1): {error_sig_c}")
            else:
                print("\nSignificant Events (Cluster 0 vs. Cluster 1):\n", sig_events_clusters_df)

            # Test case 4: Cluster 0 vs all other clusters
            sig_events_clusters_df_vs_rest, error_sig_c_rest = analyze_significant_event_types(
                mock_features_with_events,
                mock_cluster_labels,
                event_feature_prefix="evt_count_",
                at_risk_label=0, # Cluster 0 as at-risk
                baseline_label=None # All other clusters as baseline
            )
            if error_sig_c_rest:
                print(f"Error in significant event analysis (cluster 0 vs rest): {error_sig_c_rest}")
            else:
                print("\nSignificant Events (Cluster 0 vs. Rest of Clusters):\n", sig_events_clusters_df_vs_rest)
        else:
            print("\nSkipping cluster-based significant event tests: not enough distinct clusters or labels not available.")
    else:
        print("\nSkipping significant event type analysis: mock event data not available.")
