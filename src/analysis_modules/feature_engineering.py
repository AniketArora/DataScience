import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf

def extract_basic_stats(series: pd.Series, prefix=""):
    features = {}
    series_cleaned_for_stats = series.dropna()
    if series_cleaned_for_stats.empty:
        for stat in ['mean', 'std', 'median', 'min', 'max', 'skewness', 'kurtosis_val', 'sum', 'variance', 'iqr']:
            features[f'{prefix}{stat}'] = np.nan
        if prefix: features[f'{prefix}sum'] = np.nan # Ensure sum is also NaN if all were NaN
        else: features['sum'] = np.nan
        return features

    features[f'{prefix}mean'] = series_cleaned_for_stats.mean()
    features[f'{prefix}std'] = series_cleaned_for_stats.std()
    features[f'{prefix}median'] = series_cleaned_for_stats.median()
    features[f'{prefix}min'] = series_cleaned_for_stats.min()
    features[f'{prefix}max'] = series_cleaned_for_stats.max()
    features[f'{prefix}skewness'] = skew(series_cleaned_for_stats)
    features[f'{prefix}kurtosis_val'] = kurtosis(series_cleaned_for_stats)
    features[f'{prefix}sum'] = series_cleaned_for_stats.sum()
    features[f'{prefix}variance'] = series_cleaned_for_stats.var()
    features[f'{prefix}iqr'] = series_cleaned_for_stats.quantile(0.75) - series_cleaned_for_stats.quantile(0.25)
    return features

def extract_trend_features(series: pd.Series, prefix=""):
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty or len(series_cleaned) < 2:
        features[f'{prefix}slope'] = np.nan
        return features
    x = np.arange(len(series_cleaned))
    try:
        coeffs = np.polyfit(x, series_cleaned.values, 1)
        features[f'{prefix}slope'] = coeffs[0]
    except (np.linalg.LinAlgError, TypeError):
        features[f'{prefix}slope'] = np.nan
    return features

def extract_volatility_features(series: pd.Series, prefix=""):
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty or len(series_cleaned) < 2:
        features[f'{prefix}mean_abs_diff'] = np.nan
        features[f'{prefix}std_diff'] = np.nan
        return features
    diff_series = series_cleaned.diff().dropna()
    if diff_series.empty:
        features[f'{prefix}mean_abs_diff'] = 0.0
        features[f'{prefix}std_diff'] = 0.0
        return features
    features[f'{prefix}mean_abs_diff'] = diff_series.abs().mean()
    features[f'{prefix}std_diff'] = diff_series.std()
    return features

def extract_autocorrelation_features(series: pd.Series, lags=[1, 5, 10], prefix=""):
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
        return features
    max_possible_nlags = len(series_cleaned) - 1
    if max_possible_nlags < 1:
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
        return features
    nlags_to_compute = 0
    if lags: nlags_to_compute = min(max(lags), max_possible_nlags)
    acf_values = []
    if nlags_to_compute >= 1 :
        try: acf_values = acf(series_cleaned, nlags=nlags_to_compute, fft=False)
        except Exception:
            for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
            return features
    for lag in lags:
        if lag > 0 and lag < len(acf_values):
            features[f'{prefix}acf_lag_{lag}'] = acf_values[lag]
        else: features[f'{prefix}acf_lag_{lag}'] = np.nan
    return features

def extract_rolling_stats_features(series: pd.Series, windows, prefix=""):
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        for window in windows:
            for stat_type_val in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
                 features[f'{prefix}rolling_{stat_type_val}_w{window}'] = np.nan
        return features
    for window in windows:
        if window > len(series_cleaned):
            for stat_type_val in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
                 features[f'{prefix}rolling_{stat_type_val}_w{window}'] = np.nan
            continue
        current_mean_of_means, current_mean_of_stds = np.nan, np.nan
        current_std_of_means, current_std_of_stds = np.nan, np.nan
        if window == 1:
            if not series_cleaned.empty:
                current_mean_of_means = series_cleaned.mean()
                current_mean_of_stds = 0.0
                current_std_of_means = series_cleaned.std() if len(series_cleaned) >= 2 else 0.0
                current_std_of_stds = 0.0
        elif not series_cleaned.empty:
            rolling_mean_intermediate = series_cleaned.rolling(window=window, min_periods=1).mean().dropna()
            rolling_std_intermediate = series_cleaned.rolling(window=window, min_periods=2).std().dropna()
            if not rolling_mean_intermediate.empty:
                current_mean_of_means = rolling_mean_intermediate.mean()
                if len(rolling_mean_intermediate) >= 2: current_std_of_means = rolling_mean_intermediate.std()
            if not rolling_std_intermediate.empty:
                current_mean_of_stds = rolling_std_intermediate.mean()
                if len(rolling_std_intermediate) >= 2: current_std_of_stds = rolling_std_intermediate.std()
            if series_cleaned.nunique() == 1:
                current_mean_of_stds = 0.0; current_std_of_stds = 0.0
        features[f'{prefix}rolling_mean_of_means_w{window}'] = current_mean_of_means
        features[f'{prefix}rolling_mean_of_stds_w{window}'] = current_mean_of_stds
        features[f'{prefix}rolling_std_of_means_w{window}'] = current_std_of_means
        features[f'{prefix}rolling_std_of_stds_w{window}'] = current_std_of_stds
    return features

def extract_event_features_for_series(
    target_series: pd.Series,
    device_event_df: pd.DataFrame,
    prefix="evt_",
    top_n_event_types=5
):
    features = {}
    # Initialize with a definite set of features expected for consistency, even if values are 0 or NaN
    features[f'{prefix}total_events_in_period'] = 0
    # For specific event type counts/time_since/freq, we'll add them if they exist or based on a global list later.

    if not isinstance(target_series, pd.Series) or not isinstance(target_series.index, pd.DatetimeIndex) or target_series.empty:
        # If target series is invalid, we can't determine time range, so return basic structure.
        # If a global list of event types were available, we'd populate NaNs/0s for those here.
        return features
    if not isinstance(device_event_df, pd.DataFrame) or device_event_df.empty or \
       not all(col in device_event_df.columns for col in ['timestamp', 'event_type']):
        return features # No event data or malformed, return basic structure

    try:
        events_df_processed = device_event_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(events_df_processed['timestamp']):
            events_df_processed['timestamp'] = pd.to_datetime(events_df_processed['timestamp'], errors='coerce')
        events_df_processed = events_df_processed.dropna(subset=['timestamp'])
        if events_df_processed.empty: return features
    except Exception:
        return features

    series_start_time = target_series.index.min()
    series_end_time = target_series.index.max()

    relevant_events = events_df_processed[
        (events_df_processed['timestamp'] >= series_start_time) &
        (events_df_processed['timestamp'] <= series_end_time)
    ].copy()

    features[f'{prefix}total_events_in_period'] = len(relevant_events)
    if relevant_events.empty: return features

    event_type_counts = relevant_events['event_type'].value_counts()
    event_types_to_featurize = event_type_counts.head(top_n_event_types).index.tolist()

    for event_type in event_types_to_featurize:
        clean_event_type_name = str(event_type).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30] # Truncate long names
        features[f'{prefix}count_{clean_event_type_name}'] = event_type_counts.get(event_type, 0)

    relevant_events_sorted = relevant_events.sort_values(by='timestamp', ascending=False)
    for event_type in event_types_to_featurize:
        clean_event_type_name = str(event_type).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30]
        last_occurrence = relevant_events_sorted[relevant_events_sorted['event_type'] == event_type]['timestamp'].max()
        if pd.notna(last_occurrence):
            time_since = (series_end_time - last_occurrence).total_seconds() / 3600.0
            features[f'{prefix}hours_since_last_{clean_event_type_name}'] = time_since
        else:
            features[f'{prefix}hours_since_last_{clean_event_type_name}'] = np.nan

    series_duration_hours = (series_end_time - series_start_time).total_seconds() / 3600.0
    if series_duration_hours > 1e-6:
        for event_type in event_types_to_featurize:
            clean_event_type_name = str(event_type).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30]
            count = event_type_counts.get(event_type, 0)
            features[f'{prefix}freq_per_hour_{clean_event_type_name}'] = count / series_duration_hours
    else:
         for event_type in event_types_to_featurize:
            clean_event_type_name = str(event_type).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30]
            features[f'{prefix}freq_per_hour_{clean_event_type_name}'] = np.nan # Freq is undefined for zero duration

    return features

@st.cache_data
def generate_all_features_for_series(series: pd.Series, name="ts_",
                                     device_event_df: pd.DataFrame = None,
                                     all_possible_event_types: list = None,
                                     top_n_event_types_to_focus=5):
    all_features = {}
    all_features.update(extract_basic_stats(series, prefix=f"{name}basic_"))
    all_features.update(extract_trend_features(series, prefix=f"{name}trend_"))
    all_features.update(extract_volatility_features(series, prefix=f"{name}vol_"))
    all_features.update(extract_autocorrelation_features(series, lags=[1, 5, 10], prefix=f"{name}acf_"))

    series_cleaned_for_len = series.dropna()
    len_s = len(series_cleaned_for_len)
    dynamic_rolling_windows = []
    if len_s >= 40: dynamic_rolling_windows.extend([5,10,20])
    elif len_s >= 20: dynamic_rolling_windows.extend([5,10])
    elif len_s >= 10: dynamic_rolling_windows.append(5)
    if len_s > 0 :
        short_window_candidate = max(1, len_s // 2)
        is_standard_dynamic = short_window_candidate in [5,10,20]
        if not is_standard_dynamic and short_window_candidate not in dynamic_rolling_windows:
            if len_s >= short_window_candidate : dynamic_rolling_windows.append(short_window_candidate)
        if 1 not in dynamic_rolling_windows and len_s >=1 : dynamic_rolling_windows.append(1)
    dynamic_rolling_windows = sorted(list(set(w for w in dynamic_rolling_windows if 0 < w <= len_s)))
    if not series_cleaned_for_len.empty and dynamic_rolling_windows:
        computed_rolling_features = extract_rolling_stats_features(series, windows=dynamic_rolling_windows, prefix=f"{name}roll_")
        all_features.update(computed_rolling_features)
    fixed_expected_windows_for_output_keys = [1, 4, 5, 10, 20]
    for window_val in fixed_expected_windows_for_output_keys:
        for stat_type_val in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
            key_name = f'{name}roll_{stat_type_val}_w{window_val}'
            if key_name not in all_features: all_features[key_name] = np.nan

    # Initialize placeholders for event features based on all_possible_event_types
    all_features[f'{name}evt_total_events_in_period'] = 0 # Default to 0 if no events/no event_df
    if all_possible_event_types:
        for clean_event_type_name in all_possible_event_types:
            all_features[f'{name}evt_count_{clean_event_type_name}'] = 0
            all_features[f'{name}evt_hours_since_last_{clean_event_type_name}'] = np.nan
            all_features[f'{name}evt_freq_per_hour_{clean_event_type_name}'] = 0.0

    if device_event_df is not None and not device_event_df.empty and \
       isinstance(series.index, pd.DatetimeIndex) and not series.empty:
        # Extract features for top_n local events. These will overwrite placeholders if names match.
        event_features = extract_event_features_for_series(
            series, device_event_df,
            prefix=f"{name}evt_",
            top_n_event_types=top_n_event_types_to_focus
        )
        all_features.update(event_features)
        # Ensure that any global event types not captured in this series' top_n still have their default placeholder values
        # This is handled by the pre-initialization if all_possible_event_types was provided.
        # If all_possible_event_types was NOT provided, then only features for this series' events are present.

    return pd.Series(all_features)

if __name__ == '__main__':
    idx = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = [i + (i % 7) * 2 + np.random.randn() * 5 + (i/10)**1.5 for i in range(50)]
    sample_ts = pd.Series(data, index=idx, name="MySensor")
    sample_ts.iloc[[3, 10, 20, 35]] = np.nan

    print("--- Features for Sample Time Series (No Events) ---")
    ts_features = generate_all_features_for_series(sample_ts, name="sensor_X_")
    print(ts_features)

    print("\n--- Features with Event Data (Consistent Columns) ---")
    sample_event_data = {
        'timestamp': pd.to_datetime(['2023-01-05', '2023-01-10', '2023-01-05', '2023-02-10', '2023-01-15']),
        'event_type': ['Error_A', 'Warning_B', 'Error_A', 'Error_A', 'Info_C_Long_Name_Event'] # Info_C is new
    }
    sample_device_events_df = pd.DataFrame(sample_event_data)

    global_cleaned_event_types = ["Error_A", "Warning_B", "Info_C_Long_Name_Event", "Unseen_Global_Event"]

    ts_with_event_features_consistent = generate_all_features_for_series(
        sample_ts,
        name="sensor_X_",
        device_event_df=sample_device_events_df,
        all_possible_event_types=global_cleaned_event_types,
        top_n_event_types_to_focus=3
    )
    print(ts_with_event_features_consistent[ts_with_event_features_consistent.index.str.startswith("sensor_X_evt_")])

    print("\n--- Event Features for series with no relevant events ---")
    short_ts_for_event_test = pd.Series(np.arange(5), index=pd.date_range(start='2020-01-01', periods=5, freq='D'))
    event_features_none = generate_all_features_for_series(short_ts_for_event_test, name="short_evt_test_", device_event_df=sample_device_events_df, all_possible_event_types=global_cleaned_event_types)
    print(event_features_none[event_features_none.index.str.startswith("short_evt_test_evt_")])

    print("\n--- Event Features with various event types (local top N focus) ---")
    complex_event_data = {
        'timestamp': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08']),
        'event_type': ['Type A', 'Type B', 'Type A', 'Type C', 'Type A', 'Type B', 'Type D']
    }
    complex_device_events_df = pd.DataFrame(complex_event_data)
    target_series_for_complex_events = pd.Series(np.arange(10), index=pd.date_range(start='2023-01-01', periods=10, freq='D'), name="ComplexTarget")
    event_features_complex = generate_all_features_for_series(
        target_series_for_complex_events,
        name="complex_",
        device_event_df=complex_device_events_df,
        all_possible_event_types=["Type A", "Type B", "Type C", "Type D", "Type E"], # Global list for consistency
        top_n_event_types_to_focus=3 # extract_event_features will focus on top 3 local, generate_all ensures all global are present
    )
    print(event_features_complex[event_features_complex.index.str.startswith("complex_evt_")])


# --- Function to be run in ProcessPoolExecutor ---
def run_feature_engineering_for_all_devices(
    data_df_original_serializable: pd.DataFrame,
    ts_specs_serializable: dict,
    event_df_serializable: pd.DataFrame,
    global_top_event_types_cleaned_serializable: list
):
    """
    Computes features for all devices/entities.
    This function is designed to be run in a ProcessPoolExecutor.
    It calls generate_all_features_for_series for each entity.
    """
    if data_df_original_serializable.empty or \
       ts_specs_serializable.get("timestamp_col", "None") == "None" or \
       ts_specs_serializable.get("selected_value_col_for_analysis", "None") == "None":
        return pd.DataFrame()

    df_for_all_features = data_df_original_serializable.copy()
    id_cols_for_all = ts_specs_serializable["id_cols"]
    ts_col_for_all = ts_specs_serializable["timestamp_col"]
    val_col_to_process_for_all = ts_specs_serializable["selected_value_col_for_analysis"]

    temp_id_col_name_all_features = "_temp_unique_id_all_features_"
    if id_cols_for_all:
        df_for_all_features[temp_id_col_name_all_features] = df_for_all_features[id_cols_for_all].astype(str).agg('-'.join, axis=1)
        unique_entities = sorted(df_for_all_features[temp_id_col_name_all_features].unique())
    else:
        df_for_all_features[temp_id_col_name_all_features] = "DefaultTimeSeries"
        unique_entities = ["DefaultTimeSeries"]

    all_features_list = []

    event_df_main_for_all = event_df_serializable
    event_id_col_main_all = ts_specs_serializable.get('event_device_id_col', 'device_id')
    event_type_col_main_all = ts_specs_serializable.get('event_event_type_col', 'event_type')
    event_ts_col_main_all = ts_specs_serializable.get('event_timestamp_col', 'timestamp')
    # Determine top_n_event_types_to_focus for generate_all_features_for_series
    # This could be a fixed number or passed in ts_specs_serializable if made configurable
    top_n_event_types_focus = ts_specs_serializable.get('top_n_event_types_for_series_features', 5)


    for i, entity_id_val in enumerate(unique_entities):
        entity_df_all = df_for_all_features[df_for_all_features[temp_id_col_name_all_features] == entity_id_val]
        try:
            entity_df_all[ts_col_for_all] = pd.to_datetime(entity_df_all[ts_col_for_all], errors='coerce')
            entity_df_all = entity_df_all.dropna(subset=[ts_col_for_all, val_col_to_process_for_all])

            if entity_df_all.empty:
                continue

            entity_df_all = entity_df_all.sort_values(by=ts_col_for_all)
            processed_entity_series_all = entity_df_all.groupby(ts_col_for_all)[val_col_to_process_for_all].mean().rename(val_col_to_process_for_all)

            if processed_entity_series_all.empty or len(processed_entity_series_all) < 2:
                continue

            device_specific_events_all = pd.DataFrame()
            if not event_df_main_for_all.empty and id_cols_for_all and entity_id_val != "DefaultTimeSeries":
                if event_id_col_main_all in event_df_main_for_all.columns:
                    # Ensure consistent data types for comparison if IDs are numeric/string mixes
                    device_specific_events_all = event_df_main_for_all[event_df_main_for_all[event_id_col_main_all].astype(str) == str(entity_id_val)]
            elif not event_df_main_for_all.empty and not id_cols_for_all and entity_id_val == "DefaultTimeSeries": # Events for the whole dataset
                 device_specific_events_all = event_df_main_for_all

            features_series, error_msg_feat = generate_all_features_for_series(
                processed_entity_series_all,
                name=f"{val_col_to_process_for_all}_",
                device_event_df=device_specific_events_all if not device_specific_events_all.empty else None,
                all_possible_event_types=global_top_event_types_cleaned_serializable, # Pass the global list
                top_n_event_types_to_focus=top_n_event_types_focus # Use the determined focus number
            )

            if error_msg_feat: # generate_all_features_for_series now returns tuple (Series|None, error_msg|None)
                # Log error_msg_feat if logging is set up
                print(f"Feature generation error for {entity_id_val}: {error_msg_feat}")
                continue
            if features_series is not None and not features_series.empty:
                features_series_df = features_series.to_frame().T
                features_series_df.index = [entity_id_val]
                all_features_list.append(features_series_df)

        except Exception as e_feat_loop:
            print(f"Error processing entity {entity_id_val} in background: {e_feat_loop}")
            continue

    if all_features_list:
        final_features_df = pd.concat(all_features_list)
        final_features_df.dropna(axis=1, how='all', inplace=True)
        return final_features_df
    else:
        return pd.DataFrame()
