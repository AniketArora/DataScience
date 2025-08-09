import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
import logging
import streamlit as st

logger = logging.getLogger(__name__)

def extract_basic_stats(series: pd.Series, prefix=""):
    features = {}
    series_cleaned_for_stats = series.dropna()
    if series_cleaned_for_stats.empty:
        for stat in ['mean', 'std', 'median', 'min', 'max', 'skewness', 'kurtosis_val', 'sum', 'variance', 'iqr']:
            features[f'{prefix}{stat}'] = np.nan
        if prefix: features[f'{prefix}sum'] = np.nan
        else: features['sum'] = np.nan
        return features

    features[f'{prefix}mean'] = series_cleaned_for_stats.mean()
    features[f'{prefix}std'] = series_cleaned_for_stats.std() # Pandas handles single value series returning NaN
    features[f'{prefix}median'] = series_cleaned_for_stats.median()
    features[f'{prefix}min'] = series_cleaned_for_stats.min()
    features[f'{prefix}max'] = series_cleaned_for_stats.max()
    # Scipy's skew and kurtosis return NaN for < 3 values if bias=True (default)
    features[f'{prefix}skewness'] = skew(series_cleaned_for_stats)
    features[f'{prefix}kurtosis_val'] = kurtosis(series_cleaned_for_stats)
    features[f'{prefix}sum'] = series_cleaned_for_stats.sum()
    features[f'{prefix}variance'] = series_cleaned_for_stats.var() # Pandas handles single value series returning NaN
    q75 = series_cleaned_for_stats.quantile(0.75)
    q25 = series_cleaned_for_stats.quantile(0.25)
    features[f'{prefix}iqr'] = q75 - q25 if pd.notna(q75) and pd.notna(q25) else np.nan
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
    if series_cleaned.empty:
        features[f'{prefix}mean_abs_diff'] = np.nan
        features[f'{prefix}std_diff'] = np.nan
        return features
    if len(series_cleaned) < 2: # Explicitly handle series with < 2 non-NaN points
        features[f'{prefix}mean_abs_diff'] = 0.0
        features[f'{prefix}std_diff'] = 0.0
        return features
    diff_series = series_cleaned.diff().dropna()
    if diff_series.empty: # Happens if series_cleaned has only 1 point after diff+dropna (i.e. original had 1 or 2 points)
        features[f'{prefix}mean_abs_diff'] = 0.0 if len(series_cleaned) == 1 else np.nan # Mean abs diff is 0 for 1 pt, undefined for 2 pts resulting in 1 diff
        features[f'{prefix}std_diff'] = 0.0 if len(series_cleaned) == 1 else np.nan # Std of diff is 0 for 1 pt, undefined for 2 pts
        return features
    features[f'{prefix}mean_abs_diff'] = diff_series.abs().mean()
    features[f'{prefix}std_diff'] = diff_series.std() # Pandas std() returns NaN for a single value series (ddof=1)
    return features

def extract_autocorrelation_features(series: pd.Series, lags: list, prefix=""): # lags is now a list
    features = {}
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
        return features

    # Ensure all requested lags are positive and less than series length
    valid_lags = [lag for lag in lags if lag > 0 and lag < len(series_cleaned)]
    if not valid_lags:
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan # Ensure all original lag keys are present
        return features

    max_lag_to_compute = max(valid_lags)

    try:
        acf_values = acf(series_cleaned, nlags=max_lag_to_compute, fft=False)
        for lag in lags: # Iterate through original requested lags
            if lag in valid_lags and lag < len(acf_values):
                features[f'{prefix}acf_lag_{lag}'] = acf_values[lag]
            else:
                features[f'{prefix}acf_lag_{lag}'] = np.nan
    except Exception as e:
        logger.warning(f"ACF calculation failed for series {series.name if series.name else 'Unnamed'}: {e}")
        for lag in lags: features[f'{prefix}acf_lag_{lag}'] = np.nan
    return features

def extract_rolling_stats_features(series: pd.Series, windows: list, prefix=""):
    features = {}
    series_cleaned = series.dropna()

    # Initialize all possible features to NaN. This is a safeguard.
    for window in windows:
        for stat_name in ['mean_of_means', 'mean_of_stds', 'std_of_means', 'std_of_stds']:
            features[f'{prefix}{stat_name}_w{window}'] = np.nan

    if series_cleaned.empty:
        return features

    # If the series is constant, all std-based features are 0, and the mean is the constant value.
    # This is a critical edge case.
    if series_cleaned.nunique() == 1:
        constant_value = series_cleaned.iloc[0]
        for window in windows:
            features[f'{prefix}mean_of_means_w{window}'] = constant_value
            features[f'{prefix}mean_of_stds_w{window}'] = 0.0
            features[f'{prefix}std_of_means_w{window}'] = 0.0
            features[f'{prefix}std_of_stds_w{window}'] = 0.0
        return features

    for window in windows:
        if not isinstance(window, int) or window <= 0:
            logger.warning(f"Invalid window size {window} provided. Skipping.")
            continue

        # Use min_periods=1 to include calculations for partial windows at the start of the series.
        # This prevents losing data for smaller series and ensures rolling objects are not empty.
        rolling_mean = series_cleaned.rolling(window=window, min_periods=1).mean()

        # Calculate stats over the rolling mean series
        features[f'{prefix}mean_of_means_w{window}'] = rolling_mean.mean()
        std_of_means = rolling_mean.std()
        features[f'{prefix}std_of_means_w{window}'] = std_of_means if pd.notna(std_of_means) else 0.0

        # For rolling std, min_periods=2 is required for a non-NaN result.
        # We fillna(0) because the std of a single point or a constant window is 0.
        if window == 1:
            # The std of a rolling window of size 1 is always 0.
            rolling_std = pd.Series(0.0, index=series_cleaned.index)
        else:
            rolling_std = series_cleaned.rolling(window=window, min_periods=2).std().fillna(0)

        # Calculate stats over the rolling std series
        features[f'{prefix}mean_of_stds_w{window}'] = rolling_std.mean()
        std_of_stds = rolling_std.std()
        features[f'{prefix}std_of_stds_w{window}'] = std_of_stds if pd.notna(std_of_stds) else 0.0

    return features

def extract_event_features_for_series(
    target_series: pd.Series,
    device_event_df: pd.DataFrame,
    prefix="evt_",
    event_timestamp_col_name = 'timestamp',
    event_type_col_name = 'event_type',
    all_possible_event_types_cleaned_names: list = None,
    top_n_event_types=5 # Fallback if all_possible_event_types_cleaned_names is None
):
    features = {}
    features[f'{prefix}total_events_in_period'] = 0

    if not isinstance(target_series, pd.Series) or not isinstance(target_series.index, pd.DatetimeIndex) or target_series.empty:
        return features, "Target series is invalid or empty for event feature extraction."

    if device_event_df is None or device_event_df.empty:
        return features, None

    required_cols = [event_timestamp_col_name, event_type_col_name]
    if not all(col in device_event_df.columns for col in required_cols):
        return features, f"Missing essential event columns in device_event_df. Expected: {', '.join(required_cols)}."

    try:
        events_df_processed = device_event_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(events_df_processed[event_timestamp_col_name]):
            events_df_processed[event_timestamp_col_name] = pd.to_datetime(events_df_processed[event_timestamp_col_name], errors='coerce')

        events_df_processed = events_df_processed.dropna(subset=[event_timestamp_col_name, event_type_col_name])
        if events_df_processed.empty:
            return features, None
    except Exception as e:
        return features, f"Error processing event timestamps or types: {e}"

    series_start_time = target_series.index.min()
    series_end_time = target_series.index.max()

    relevant_events = events_df_processed[
        (events_df_processed[event_timestamp_col_name] >= series_start_time) &
        (events_df_processed[event_timestamp_col_name] <= series_end_time)
    ].copy()

    features[f'{prefix}total_events_in_period'] = len(relevant_events)
    if relevant_events.empty:
        return features, None

    event_type_counts = relevant_events[event_type_col_name].value_counts()

    event_types_to_process_locally = []
    locally_found_raw_event_types = event_type_counts.index.tolist()

    if all_possible_event_types_cleaned_names:
        for raw_local_event_type in locally_found_raw_event_types:
            cleaned_local_name = str(raw_local_event_type).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30]
            if cleaned_local_name in all_possible_event_types_cleaned_names:
                event_types_to_process_locally.append(raw_local_event_type)
    else:
        # Fallback: if no global list provided, use top_n_event_types from local events
        event_types_to_process_locally = event_type_counts.iloc[:min(top_n_event_types, len(event_type_counts))].index.tolist()

    if not event_types_to_process_locally:
        return features, None

    relevant_events_sorted = relevant_events.sort_values(by=event_timestamp_col_name, ascending=False)
    series_duration_hours = (series_end_time - series_start_time).total_seconds() / 3600.0

    for raw_event_type in event_types_to_process_locally:
        clean_event_type_name = str(raw_event_type).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30]

        current_count = event_type_counts.get(raw_event_type, 0)
        features[f'{prefix}count_{clean_event_type_name}'] = current_count

        type_specific_events = relevant_events_sorted[relevant_events_sorted[event_type_col_name] == raw_event_type]
        if not type_specific_events.empty:
            last_occurrence = type_specific_events[event_timestamp_col_name].max()
            time_since = (series_end_time - last_occurrence).total_seconds() / 3600.0
            features[f'{prefix}hours_since_last_{clean_event_type_name}'] = time_since
        else:
            features[f'{prefix}hours_since_last_{clean_event_type_name}'] = np.nan

        if series_duration_hours > 1e-6:
            features[f'{prefix}freq_per_hour_{clean_event_type_name}'] = current_count / series_duration_hours
        else:
            features[f'{prefix}freq_per_hour_{clean_event_type_name}'] = np.nan

    return features, None

@st.cache_data
def generate_all_features_for_series(series: pd.Series, name="ts_",
                                     device_event_df: pd.DataFrame = None,
                                     all_possible_event_types: list = None,
                                     event_type_col: str = "event_type",
                                     event_ts_col: str = "timestamp",
                                     top_n_event_types_to_focus=5,
                                     acf_lags_list: list = None,
                                     rolling_windows_list: list = None
                                     ):
    try:
        series_name_for_log = series.name if series is not None and series.name is not None else 'Unnamed Series'
        all_features = {}

        acf_lags_to_use = acf_lags_list if acf_lags_list is not None and len(acf_lags_list)>0 else [1, 5, 10]
        default_rolling_windows = [1, 5, 10, 20]
        rolling_windows_to_use = rolling_windows_list if rolling_windows_list is not None and len(rolling_windows_list) > 0 else default_rolling_windows

        all_features.update(extract_basic_stats(series, prefix=f"{name}basic_"))
        all_features.update(extract_trend_features(series, prefix=f"{name}trend_"))
        all_features.update(extract_volatility_features(series, prefix=f"{name}vol_"))
        all_features.update(extract_autocorrelation_features(series, lags=acf_lags_to_use, prefix=f"{name}acf_"))

        if rolling_windows_to_use: # Ensure list is not empty
             all_features.update(extract_rolling_stats_features(series, windows=rolling_windows_to_use, prefix=f"{name}roll_"))
        else: # If rolling_windows_to_use ended up empty, ensure no error and potentially log
            logger.debug(f"No rolling windows to process for series {series_name_for_log}.")


        # Initialize placeholders for ALL event features based on all_possible_event_types (cleaned names)
        temp_event_features_initialized = {}
        temp_event_features_initialized[f'{name}evt_total_events_in_period'] = 0
        if all_possible_event_types:
            for clean_event_type_name in all_possible_event_types:
                temp_event_features_initialized[f'{name}evt_count_{clean_event_type_name}'] = 0
                temp_event_features_initialized[f'{name}evt_hours_since_last_{clean_event_type_name}'] = np.nan
                temp_event_features_initialized[f'{name}evt_freq_per_hour_{clean_event_type_name}'] = 0.0
        all_features.update(temp_event_features_initialized)

        event_features_calculated_dict, event_err_msg = None, None
        if device_event_df is not None and not device_event_df.empty and \
           isinstance(series.index, pd.DatetimeIndex) and not series.empty:

            event_features_calculated_dict, event_err_msg = extract_event_features_for_series(
                target_series=series,
                device_event_df=device_event_df,
                prefix=f"{name}evt_",
                event_timestamp_col_name=event_ts_col,
                event_type_col_name=event_type_col,
                all_possible_event_types_cleaned_names=all_possible_event_types,
                top_n_event_types=top_n_event_types_to_focus
            )
            if event_err_msg:
                logger.warning("Error during event feature extraction for series %s: %s", series_name_for_log, event_err_msg)
            if event_features_calculated_dict:
                all_features.update(event_features_calculated_dict)

        return pd.Series(all_features), None
    except Exception as e:
        series_name_for_log = series.name if series is not None and series.name is not None else 'Unnamed Series'
        error_message_to_return = f"Error in generate_all_features_for_series for series {series_name_for_log}: {e}"
        logger.debug(error_message_to_return, exc_info=True)
        return None, error_message_to_return


if __name__ == '__main__':
    idx = pd.date_range(start='2023-01-01', periods=50, freq='D')
    data = [i + (i % 7) * 2 + np.random.randn() * 5 + (i/10)**1.5 for i in range(50)]
    sample_ts = pd.Series(data, index=idx, name="MySensor")
    sample_ts.iloc[[3, 10, 20, 35]] = np.nan

    print("--- Features for Sample Time Series (No Events) ---")
    ts_features, err = generate_all_features_for_series(sample_ts, name="sensor_X_")
    if err: print(f"Error: {err}")
    else: print(ts_features)

    print("\n--- Features with Event Data (Consistent Columns) ---")
    sample_event_data = {
        'timestamp': pd.to_datetime(['2023-01-05', '2023-01-10', '2023-01-05', '2023-02-10', '2023-01-15']),
        'event_type': ['Error_A', 'Warning_B', 'Error_A', 'Error_A', 'Info_C_Long_Name_Event']
    }
    sample_device_events_df = pd.DataFrame(sample_event_data)

    global_cleaned_event_types = ["Error_A", "Warning_B", "Info_C_Long_Name_Event", "Unseen_Global_Event"]

    ts_with_event_features_consistent, err = generate_all_features_for_series(
        sample_ts,
        name="sensor_X_",
        device_event_df=sample_device_events_df,
        all_possible_event_types=global_cleaned_event_types,
        acf_lags_list=[2,4], rolling_windows_list=[2,4,8], # Test with new params
        top_n_event_types_to_focus=3
    )
    if err: print(f"Error: {err}")
    elif ts_with_event_features_consistent is not None: print(ts_with_event_features_consistent[ts_with_event_features_consistent.index.str.startswith("sensor_X_evt_")])

    print("\n--- Event Features for series with no relevant events ---")
    short_ts_for_event_test = pd.Series(np.arange(5), index=pd.date_range(start='2020-01-01', periods=5, freq='D'))
    event_features_none, err = generate_all_features_for_series(short_ts_for_event_test, name="short_evt_test_", device_event_df=sample_device_events_df, all_possible_event_types=global_cleaned_event_types)
    if err: print(f"Error: {err}")
    elif event_features_none is not None: print(event_features_none[event_features_none.index.str.startswith("short_evt_test_evt_")])

    print("\n--- Event Features with various event types (local top N focus) ---")
    complex_event_data = {
        'timestamp': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08']),
        'event_type': ['Type A', 'Type B', 'Type A', 'Type C', 'Type A', 'Type B', 'Type D']
    }
    complex_device_events_df = pd.DataFrame(complex_event_data)
    target_series_for_complex_events = pd.Series(np.arange(10), index=pd.date_range(start='2023-01-01', periods=10, freq='D'), name="ComplexTarget")
    event_features_complex, err = generate_all_features_for_series(
        target_series_for_complex_events,
        name="complex_",
        device_event_df=complex_device_events_df,
        all_possible_event_types=["Type A", "Type B", "Type C", "Type D", "Type E"],
        event_type_col='event_type',
        event_ts_col='timestamp',
        top_n_event_types_to_focus=3
    )
    if err: print(f"Error: {err}")
    elif event_features_complex is not None: print(event_features_complex[event_features_complex.index.str.startswith("complex_evt_")])


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
    logger.info("Starting feature engineering for all devices.")
    if data_df_original_serializable.empty or \
       ts_specs_serializable.get("timestamp_col", "None") == "None" or \
       ts_specs_serializable.get("selected_value_col_for_analysis", "None") == "None":
        logger.warning("Aborting feature engineering for all devices due to empty data or missing critical spec columns.")
        return pd.DataFrame(), []

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
    top_n_event_types_focus = ts_specs_serializable.get('top_n_event_types_for_series_features', 5)

    # Extract ACF lags and Rolling Windows from ts_specs_serializable, with defaults
    acf_lags_config = ts_specs_serializable.get('acf_lags', [1, 5, 10])
    rolling_windows_config = ts_specs_serializable.get('rolling_windows', [1, 5, 10, 20])

    error_summary_list = []

    for i, entity_id_val in enumerate(unique_entities):
        entity_df_all = df_for_all_features[df_for_all_features[temp_id_col_name_all_features] == entity_id_val]
        try:
            entity_df_all[ts_col_for_all] = pd.to_datetime(entity_df_all[ts_col_for_all], errors='coerce')
            entity_df_all = entity_df_all.dropna(subset=[ts_col_for_all, val_col_to_process_for_all])

            if entity_df_all.empty:
                error_summary_list.append((entity_id_val, "No valid data after NA drop for value or timestamp column."))
                continue

            entity_df_all = entity_df_all.sort_values(by=ts_col_for_all)
            processed_entity_series_all = entity_df_all.groupby(ts_col_for_all)[val_col_to_process_for_all].mean().rename(val_col_to_process_for_all)

            if processed_entity_series_all.empty or len(processed_entity_series_all) < 2:
                error_summary_list.append((entity_id_val, f"Series too short after processing (length {len(processed_entity_series_all)})."))
                continue

            device_specific_events_all = pd.DataFrame()
            if not event_df_main_for_all.empty and id_cols_for_all and entity_id_val != "DefaultTimeSeries":
                if event_id_col_main_all in event_df_main_for_all.columns:
                    device_specific_events_all = event_df_main_for_all[event_df_main_for_all[event_id_col_main_all].astype(str) == str(entity_id_val)]
            elif not event_df_main_for_all.empty and not id_cols_for_all and entity_id_val == "DefaultTimeSeries":
                 device_specific_events_all = event_df_main_for_all

            features_series, error_msg_feat = generate_all_features_for_series(
                processed_entity_series_all,
                name=f"{val_col_to_process_for_all}_",
                device_event_df=device_specific_events_all if not device_specific_events_all.empty else None,
                all_possible_event_types=global_top_event_types_cleaned_serializable,
                event_type_col=event_type_col_main_all,
                event_ts_col=event_ts_col_main_all,
                top_n_event_types_to_focus=top_n_event_types_focus,
                acf_lags_list=acf_lags_config,
                rolling_windows_list=rolling_windows_config
            )

            if error_msg_feat:
                error_summary_list.append((entity_id_val, error_msg_feat))
                logger.warning("Feature generation error for entity '%s': %s", entity_id_val, error_msg_feat)
                continue
            if features_series is not None and not features_series.empty:
                features_series_df = features_series.to_frame().T
                features_series_df.index = [entity_id_val]
                all_features_list.append(features_series_df)
            elif not error_msg_feat:
                 error_summary_list.append((entity_id_val, "No features generated, skipping."))
                 logger.info("No features generated for entity '%s' (series might be too short or unsuitable, but no hard error), skipping.", entity_id_val)

        except Exception as e_feat_loop:
            error_msg = f"Unhandled exception: {e_feat_loop}"
            error_summary_list.append((entity_id_val, error_msg))
            logger.error("Unhandled exception processing entity '%s' in background: %s", entity_id_val, e_feat_loop, exc_info=True)
            continue

    logger.info("Completed feature engineering for %d devices. %d errors/warnings.", len(all_features_list), len(error_summary_list))
    if all_features_list:
        final_features_df = pd.concat(all_features_list)
        final_features_df.dropna(axis=1, how='all', inplace=True)
        return final_features_df, error_summary_list
    else:
        return pd.DataFrame(), error_summary_list
