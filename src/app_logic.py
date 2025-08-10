import streamlit as st
import pandas as pd
import json
import concurrent.futures
from .database import get_schemas_postgres, get_tables_postgres, fetch_data_postgres
from .analysis_modules.feature_engineering import run_feature_engineering_for_all_devices
from .config_utils import gather_settings_for_save, apply_loaded_settings_to_session_state

def handle_postgres_connection(db_conn):
    if db_conn:
        schemas, err = get_schemas_postgres(db_conn)
        if err:
            st.sidebar.error(f"Error fetching schemas: {err}")
            return

        selected_schema = st.sidebar.selectbox("Select Schema", ["None"] + schemas)
        if selected_schema != "None":
            tables, err = get_tables_postgres(db_conn, selected_schema)
            if err:
                st.sidebar.error(f"Error fetching tables: {err}")
                return

            selected_table = st.sidebar.selectbox("Select Table", ["None"] + tables)
            if selected_table != "None":
                if st.sidebar.button("Fetch Data"):
                    df, err = fetch_data_postgres(db_conn, selected_schema, selected_table)
                    if err:
                        st.sidebar.error(f"Error fetching data: {err}")
                    else:
                        st.session_state.data_df_original = df
                        st.session_state.data_df = df.copy()
                        st.sidebar.success("Data fetched successfully!")

def handle_event_data_upload():
    event_file = st.sidebar.file_uploader(
        "Upload Event Data File (CSV or Excel)",
        type=['csv', 'xlsx'],
        key="event_file_uploader_general"
    )
    if event_file:
        if st.session_state.get('last_event_file_id') != event_file.id:
            try:
                if event_file.name.endswith('.csv'):
                    df_event_temp = pd.read_csv(event_file)
                else:
                    df_event_temp = pd.read_excel(event_file)

                id_col_event = st.session_state.time_series_specs['event_device_id_col']
                ts_col_event = st.session_state.time_series_specs['event_timestamp_col']
                type_col_event = st.session_state.time_series_specs['event_event_type_col']
                required_event_cols = [id_col_event, ts_col_event, type_col_event]

                if not all(col in df_event_temp.columns for col in required_event_cols):
                    st.sidebar.error(f"Event data must contain specified columns: {', '.join(required_event_cols)}")
                else:
                    df_event_temp[ts_col_event] = pd.to_datetime(df_event_temp[ts_col_event], errors='coerce')
                    df_event_temp = df_event_temp.dropna(subset=[ts_col_event])
                    st.session_state.event_df = df_event_temp
                    st.session_state.last_event_file_id = event_file.id
                    st.sidebar.success(f"Loaded event data: {st.session_state.event_df.shape}")
                    if 'global_top_event_types_cleaned' in st.session_state:
                         del st.session_state['global_top_event_types_cleaned']
                    if 'event_df_last_loaded_id' in st.session_state:
                         del st.session_state['event_df_last_loaded_id']
            except Exception as e:
                st.sidebar.error(f"Error loading event data: {e}")

def handle_time_series_settings():
    if not st.session_state.data_df_original.empty:
        df_columns = ["None"] + st.session_state.data_df_original.columns.tolist()

        current_id_cols = [col for col in st.session_state.time_series_specs.get("id_cols", []) if col in df_columns]
        st.session_state.time_series_specs["id_cols"] = st.sidebar.multiselect(
            "Select Device/Entity ID Column(s) (Optional)", options=df_columns,
            default=current_id_cols, key="ts_id_cols_general"
        )

        current_ts_col = st.session_state.time_series_specs.get("timestamp_col", "None")
        if current_ts_col not in df_columns: current_ts_col = "None"
        st.session_state.time_series_specs["timestamp_col"] = st.sidebar.selectbox(
            "Select Timestamp Column", options=df_columns,
            index=df_columns.index(current_ts_col), key="ts_timestamp_col_general"
        )

        current_value_cols = [col for col in st.session_state.time_series_specs.get("value_cols", []) if col in df_columns]
        st.session_state.time_series_specs["value_cols"] = st.sidebar.multiselect(
            "Select Value/Metric Column(s) to Analyze", options=df_columns,
            default=current_value_cols, key="ts_value_cols_general"
        )
    else:
        st.sidebar.info("Load main data to configure time series settings.")

def handle_single_time_series_selection():
    ts_specs = st.session_state.time_series_specs
    id_cols = ts_specs.get("id_cols", [])
    ts_col = ts_specs.get("timestamp_col", "None")
    value_cols_to_analyze = ts_specs.get("value_cols", [])

    if ts_col != "None" and value_cols_to_analyze:
        unique_ids_display = ["DefaultTimeSeries"]
        temp_id_col_name_single_series = "_temp_unique_id_single_series_"

        if id_cols:
            if not st.session_state.data_df_original.empty:
                df_for_ids = st.session_state.data_df_original.copy()
                df_for_ids[temp_id_col_name_single_series] = df_for_ids[id_cols].astype(str).agg('-'.join, axis=1)
                unique_ids_list = sorted(df_for_ids[temp_id_col_name_single_series].unique().tolist())
                unique_ids_display = ["None"] + unique_ids_list
            else:
                unique_ids_display = ["None"]

        current_selected_id = ts_specs.get("selected_id", "None")
        if current_selected_id not in unique_ids_display: current_selected_id = "None" if "None" in unique_ids_display else unique_ids_display[0]

        ts_specs["selected_id"] = st.sidebar.selectbox(
            "Select specific Device/Entity ID to analyze:",
            options=unique_ids_display,
            index=unique_ids_display.index(current_selected_id),
            key="single_series_selected_id_general"
        )

        value_col_options = ["None"] + value_cols_to_analyze
        current_selected_value_col = ts_specs.get("selected_value_col_for_analysis", "None")
        if current_selected_value_col not in value_col_options: current_selected_value_col = "None"

        ts_specs["selected_value_col_for_analysis"] = st.sidebar.selectbox(
            "Select specific Value/Metric for single series analysis:",
            options=value_col_options,
            index=value_col_options.index(current_selected_value_col),
            key="single_series_selected_value_col_general"
        )

        if st.sidebar.button("Prepare Selected Time Series for Analysis", key="prepare_single_series_button_general"):
            if ts_specs["selected_id"] != "None" and ts_col != "None" and ts_specs["selected_value_col_for_analysis"] != "None":
                df_orig_copy_single = st.session_state.data_df_original.copy()

                selected_entity_id_val = ts_specs["selected_id"]
                timestamp_col_val = ts_col
                value_col_val = ts_specs["selected_value_col_for_analysis"]

                entity_series_df_single = df_orig_copy_single
                if id_cols and selected_entity_id_val != "DefaultTimeSeries":
                    df_orig_copy_single[temp_id_col_name_single_series] = df_orig_copy_single[id_cols].astype(str).agg('-'.join, axis=1)
                    entity_series_df_single = df_orig_copy_single[df_orig_copy_single[temp_id_col_name_single_series] == selected_entity_id_val]

                if entity_series_df_single.empty:
                    st.sidebar.error(f"No data found for ID: {selected_entity_id_val} with selected columns.")
                    ts_specs["processed_series"] = None
                else:
                    try:
                        entity_series_df_single[timestamp_col_val] = pd.to_datetime(entity_series_df_single[timestamp_col_val], errors='coerce')
                        st.session_state.original_selected_series_for_profiling = entity_series_df_single.set_index(timestamp_col_val)[value_col_val].copy()

                        entity_series_df_single = entity_series_df_single.dropna(subset=[timestamp_col_val, value_col_val])

                        if entity_series_df_single.empty:
                            st.sidebar.error(f"No valid data after NA drop for {value_col_val} or {timestamp_col_val}.")
                            ts_specs["processed_series"] = None
                        else:
                            entity_series_df_single = entity_series_df_single.sort_values(by=timestamp_col_val)
                            processed_series_temp = entity_series_df_single.groupby(timestamp_col_val)[value_col_val].mean().rename(value_col_val)
                            ts_specs["processed_series"] = processed_series_temp
                            st.sidebar.success(f"Prepared '{value_col_val}' for '{selected_entity_id_val}'. Length: {len(processed_series_temp)}")
                    except Exception as e:
                        st.sidebar.error(f"Error preparing series: {e}")
                        ts_specs["processed_series"] = None
            else:
                st.sidebar.warning("Please select a valid Device/Entity ID, Timestamp, and Value/Metric.")
                ts_specs["processed_series"] = None
    else:
        st.sidebar.info("Select Timestamp and at least one Value/Metric in 'Time Series Settings' to enable single series preparation.")

def handle_population_analysis_settings():
    ts_specs = st.session_state.time_series_specs
    compute_all_button_disabled_main = True
    if ts_specs.get("timestamp_col", "None") != "None" and \
       ts_specs.get("selected_value_col_for_analysis", "None") != "None" and \
       not st.session_state.data_df_original.empty:
        compute_all_button_disabled_main = False
    else:
        st.sidebar.warning("Load data and select Timestamp & a specific Value/Metric in 'Time Series Settings' to enable population feature computation.")

    if st.sidebar.button("Compute Features for ALL Devices", key="compute_all_features_button_general", disabled=compute_all_button_disabled_main):
        if st.session_state.get('feature_computation_running', False):
            st.sidebar.warning("Feature computation is already in progress.")
        else:
            st.session_state.running_all_features_computation = True
            st.session_state.feature_computation_running = True
            st.session_state.feature_computation_future = None

        try:
            parsed_acf_lags = [int(x.strip()) for x in st.session_state.widget_fe_acf_lags.split(',') if x.strip()]
            if not parsed_acf_lags: parsed_acf_lags = [1, 5, 10]
            st.session_state.fe_acf_lags_general = parsed_acf_lags
        except ValueError:
            st.sidebar.error("Invalid ACF Lags format. Using last valid or default.")

        try:
            parsed_rolling_windows = [int(x.strip()) for x in st.session_state.widget_fe_rolling_windows.split(',') if x.strip()]
            if not parsed_rolling_windows: parsed_rolling_windows = [1, 5, 10, 20]
            if any(w <= 0 for w in parsed_rolling_windows):
                st.sidebar.error("Rolling windows must be positive integers. Using last valid or default.")
            else:
                st.session_state.fe_rolling_windows_general = parsed_rolling_windows
        except ValueError:
            st.sidebar.error("Invalid Rolling Windows format. Using last valid or default.")

        st.session_state.all_device_features_df = pd.DataFrame()
        st.session_state.population_anomaly_results = {}
        st.session_state.clustering_results = {}
        st.session_state.kmeans_stats_df = None
        st.session_state.res_df_anomalies_sorted = pd.DataFrame()
        st.session_state.surrogate_tree_explainer = None

        data_df_original_serializable = st.session_state.data_df_original.copy()
        ts_specs_serializable = st.session_state.time_series_specs.copy()
        ts_specs_serializable['acf_lags'] = st.session_state.fe_acf_lags_general
        ts_specs_serializable['rolling_windows'] = st.session_state.fe_rolling_windows_general

        event_df_serializable = st.session_state.event_df.copy()
        global_top_event_types_cleaned_serializable = list(st.session_state.get("global_top_event_types_cleaned", []))

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        st.session_state.feature_computation_future = executor.submit(
            run_feature_engineering_for_all_devices,
            data_df_original_serializable,
            ts_specs_serializable,
            event_df_serializable,
            global_top_event_types_cleaned_serializable
        )
        executor.shutdown(wait=False)
        st.sidebar.info("Feature computation started in the background.")
        st.session_state.running_all_features_computation = False

        st.experimental_rerun()

def handle_feature_engineering_settings():
    st.session_state.widget_fe_acf_lags = st.sidebar.text_input(
        "ACF Lags (comma-separated integers)",
        value=",".join(map(str, st.session_state.fe_acf_lags_general)),
        key="acf_lags_input_widget"
    )
    st.session_state.widget_fe_rolling_windows = st.sidebar.text_input(
        "Rolling Windows (comma-separated integers)",
        value=",".join(map(str, st.session_state.fe_rolling_windows_general)),
        key="rolling_windows_input_widget"
    )

def handle_save_load_app_state():
    if st.sidebar.button("Save App State", key="save_app_state_button_main_general"):
        app_settings = gather_settings_for_save()
        json_settings = json.dumps(app_settings, indent=4)
        st.sidebar.download_button(
            label="Download Settings File (.json)",
            data=json_settings,
            file_name="data_analyzer_settings.json",
            mime="application/json",
            key="download_app_settings_json_main_general"
        )

    uploaded_settings_file_main = st.sidebar.file_uploader(
        "Load App State (.json)",
        type=['json'],
        key="load_app_settings_uploader_main_general"
    )
    if uploaded_settings_file_main is not None:
        try:
            loaded_settings_dict = json.load(uploaded_settings_file_main)
            success, message = apply_loaded_settings_to_session_state(loaded_settings_dict)
            if success:
                st.sidebar.success(f"Settings loaded! {message} Applying and rerunning...")
                st.session_state.load_app_settings_uploader_main_general = None
                st.experimental_rerun()
            else:
                st.sidebar.error(message)
        except Exception as e:
            st.sidebar.error(f"Error parsing settings file: {e}")
