import streamlit as st
import pandas as pd
import numpy as np # For NaNs if needed in session state init
import json # For future save/load, good to have early
import concurrent.futures
import logging

# --- Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("Universal Data Analyzer application started.")


# --- Utility / Helper Functions (e.g., config_utils if not separate) ---

# For now, keep reset function here. config_utils can be for save/load later.
def reset_all_dependent_states(clear_data_too=False):
    """Resets session state variables that depend on loaded data or settings."""
    # Time Series Specs & Processed Data
    st.session_state.time_series_specs = {
        "id_cols": [], "timestamp_col": "None", "value_cols": [],
        "selected_id": "None", "selected_value_col_for_analysis": "None",
        "processed_series": None,
        # User specified event col names
        "event_device_id_col": "device_id",
        "event_timestamp_col": "timestamp",
        "event_event_type_col": "event_type"
    }
    st.session_state.single_series_features_display = None

    # Population Analysis
    st.session_state.all_device_features_df = pd.DataFrame()
    st.session_state.population_anomaly_results = {}
    st.session_state.clustering_results = {}
    st.session_state.kmeans_stats_df = None # For Elbow/Silhouette plots
    st.session_state.res_df_anomalies_sorted = pd.DataFrame() # For explainability
    st.session_state.surrogate_tree_explainer = None
    st.session_state.running_all_features_computation = False # Flag

    # Validation
    # Explicitly do not clear failed_ids_text_area_general here, user might want to keep it.

    if clear_data_too:
        st.session_state.data_df_original = pd.DataFrame()
        st.session_state.data_df = pd.DataFrame() # General display df
        st.session_state.event_df = pd.DataFrame()
        st.session_state.db_conn = None # Reset DB connection object
        # Reset other data specific flags if any (like event_df_last_loaded_id)
        if 'event_df_last_loaded_id' in st.session_state:
            del st.session_state['event_df_last_loaded_id']
        if 'global_top_event_types_cleaned' in st.session_state:
            del st.session_state['global_top_event_types_cleaned']


# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Universal Data Analyzer 📊", layout="wide")

# --- Initialize Session State (Core variables) ---
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "General Analysis" # Default mode
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None
if 'data_df_original' not in st.session_state:
    st.session_state.data_df_original = pd.DataFrame()
if 'data_df' not in st.session_state: # For general display, filtering etc.
    st.session_state.data_df = pd.DataFrame()
if 'event_df' not in st.session_state:
    st.session_state.event_df = pd.DataFrame()

# Initialize other states if they are not reset by reset_all_dependent_states
# or if they need to exist before reset_all_dependent_states is ever called.
if 'time_series_specs' not in st.session_state: # Ensure this dict exists
    reset_all_dependent_states() # Call once to initialize all sub-keys

if 'feature_computation_future' not in st.session_state:
    st.session_state.feature_computation_future = None
if 'feature_computation_running' not in st.session_state:
    st.session_state.feature_computation_running = False
if 'feature_generation_errors' not in st.session_state: # Initialize new state for errors
    st.session_state.feature_generation_errors = []

# --- Imports for modules (will be used later but good to have grouped) ---
import matplotlib.pyplot as plt
from src.database import connect_postgres, get_schemas_postgres, get_tables_postgres, fetch_data_postgres
from src.analysis_modules.profiling import (
    get_series_summary_stats,
    get_missing_values_summary,
    perform_stationarity_test
)
from src.analysis_modules.decomposition import decompose_time_series
from src.analysis_modules.anomalies import (
    detect_anomalies_zscore,
    detect_anomalies_iqr,
    detect_anomalies_isolation_forest,
    detect_anomalies_one_class_svm
)
from src.analysis_modules.feature_engineering import generate_all_features_for_series, run_feature_engineering_for_all_devices
from src.analysis_modules.clustering import (
    perform_kmeans_clustering,
    get_kmeans_elbow_silhouette_data,
    perform_dbscan_clustering
)
from src.analysis_modules.explainability import (
    get_cluster_feature_summary,
    get_feature_importance_for_clusters_anova,
    compare_anomalous_vs_normal_features,
    generate_cluster_summary_text,
    generate_anomaly_summary_text,
    explain_anomalies_with_surrogate_model,
    analyze_event_correlations
)
from sklearn.tree import plot_tree # For visualizing surrogate tree
from src.config_utils import (
    gather_settings_for_save,
    apply_loaded_settings_to_session_state,
    # Assuming these key lists are defined in config_utils as per previous setup
    # If not, they would need to be defined here or the functions adapted
)

# --- Main App Title ---
st.title("Universal Data Analyzer 📊")

# === APP MODE SELECTOR (SIDEBAR) ===
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Choose Mode:",
    ("General Analysis", "Guided Workflows"),
    key="app_mode_selector",
    on_change=reset_all_dependent_states, args=(True,) # Reset all on mode change for safety
)
st.sidebar.markdown("---")

# === SHARED SIDEBAR ELEMENTS (e.g., Data Source) ===
# These might be common to both modes, or you might move them into "General Analysis" mode
# For now, let's assume data loading is primarily a "General Analysis" setup task.

if 'db_type_general' not in st.session_state: st.session_state.db_type_general = "PostgreSQL"
# (Add other db param initializations for pg_host_general etc. if needed, or rely on text_input defaults)

# Initialize other widget states to avoid issues on first run after mode switch or load
# These are examples, more might be needed as UI is rebuilt
# For Population Anomaly Detection
if 'if_contam_general' not in st.session_state: st.session_state.if_contam_general = 0.1
if 'ocsvm_nu_general' not in st.session_state: st.session_state.ocsvm_nu_general = 0.05
if 'ocsvm_kernel_general' not in st.session_state: st.session_state.ocsvm_kernel_general = "rbf"
if 'ocsvm_gamma_general' not in st.session_state: st.session_state.ocsvm_gamma_general = "scale"
# For Clustering
if 'scale_data_clustering_kmeans_general' not in st.session_state: st.session_state.scale_data_clustering_kmeans_general = True
if 'scale_data_clustering_dbscan_general' not in st.session_state: st.session_state.scale_data_clustering_dbscan_general = True
if 'k_min_stats_general' not in st.session_state: st.session_state.k_min_stats_general = 2
if 'k_max_stats_general' not in st.session_state: st.session_state.k_max_stats_general = 10
if 'kmeans_k_final_general' not in st.session_state: st.session_state.kmeans_k_final_general = 3
if 'dbscan_eps_general' not in st.session_state: st.session_state.dbscan_eps_general = 0.5
if 'dbscan_min_samples_general' not in st.session_state: st.session_state.dbscan_min_samples_general = 5
# For Surrogate Tree
if 'surrogate_tree_depth_general' not in st.session_state: st.session_state.surrogate_tree_depth_general = 4

# Feature Engineering Defaults
if 'fe_acf_lags_general' not in st.session_state:
    st.session_state.fe_acf_lags_general = [1, 5, 10]
if 'fe_rolling_windows_general' not in st.session_state:
    st.session_state.fe_rolling_windows_general = [1, 5, 10, 20]


# === GENERAL ANALYSIS MODE ===
if app_mode == "General Analysis":
    st.header("General Analysis Mode")

    # --- Database Connection (Sidebar) ---
    # (Import connect_postgres, fetch_postgres_data, connect_elasticsearch, fetch_elasticsearch_data from src.database)
    # For now, placeholder for brevity. Subtask should re-implement full DB connection UI from previous state.
    # This was: st.sidebar.header("Database Connection"), db_type selectbox, text_inputs for host, port, etc.
    # buttons for connect and fetch. On fetch, data_df_original is populated and states are reset.
    # Crucially, on successful fetch:
    #   st.session_state.data_df_original = fetched_df
    #   reset_all_dependent_states() # Keep existing event_df, db_conn
    #   st.session_state.data_df = st.session_state.data_df_original.copy()

    # --- Database Connection UI (Sidebar) ---
    st.sidebar.header("PostgreSQL Connection")
    st.session_state.db_host = st.sidebar.text_input("Host", value=st.session_state.db_host, key="pg_host_input")
    st.session_state.db_port = st.sidebar.text_input("Port", value=st.session_state.db_port, key="pg_port_input")
    st.session_state.db_name = st.sidebar.text_input("Database Name", value=st.session_state.db_name, key="pg_dbname_input")
    st.session_state.db_user = st.sidebar.text_input("User", value=st.session_state.db_user, key="pg_user_input")
    st.session_state.db_password = st.sidebar.text_input("Password", type="password", value=st.session_state.db_password, key="pg_password_input")

    if st.sidebar.button("Connect to PostgreSQL", key="pg_connect_button"):
        logger.info("Attempting to connect to PostgreSQL...")
        db_config = {
            "host": st.session_state.db_host,
            "port": st.session_state.db_port,
            "dbname": st.session_state.db_name,
            "user": st.session_state.db_user,
            "password": st.session_state.db_password,
        }
        conn, err_msg = connect_postgres(db_config)
        if err_msg:
            st.sidebar.error(f"Connection Failed: {err_msg}")
            logger.error("PostgreSQL connection failed: %s", err_msg) # exc_info=True is better in connect_postgres itself
            st.session_state.db_conn = None
            st.session_state.available_schemas = [] # Clear schemas on failed connection
            st.session_state.available_tables = []  # Clear tables on failed connection
            st.session_state.selected_schema = None
            st.session_state.selected_table = None
        else:
            st.session_state.db_conn = conn
            st.sidebar.success("Successfully connected to PostgreSQL!")
            logger.info("PostgreSQL connection successful.")
            # Fetch schemas immediately after successful connection
            schemas, err_schemas = get_schemas_postgres(st.session_state.db_conn)
            if err_schemas:
                st.sidebar.error(f"Error fetching schemas: {err_schemas}")
                # Error already logged in get_schemas_postgres
                st.session_state.available_schemas = []
            else:
                st.session_state.available_schemas = ["None"] + (schemas if schemas else [])
            st.session_state.selected_schema = "None" # Reset schema selection
            st.session_state.available_tables = [] # Reset table list
            st.session_state.selected_table = None # Reset table selection

    # Schema and Table Selection
    if st.session_state.db_conn:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Schema and Table Selection")

        # Ensure "None" is the first option if available_schemas is populated
        current_schema_options = st.session_state.get('available_schemas', ["None"])
        if not current_schema_options: # Should not happen if connection was successful and fetched
             current_schema_options = ["None"]
        elif "None" not in current_schema_options and current_schema_options != ["None"]: # handles empty list from failed fetch
             current_schema_options = ["None"] + current_schema_options


        selected_schema_val = st.sidebar.selectbox(
            "Select Schema",
            options=current_schema_options,
            index=current_schema_options.index(st.session_state.selected_schema) if st.session_state.selected_schema in current_schema_options else 0,
            key="pg_schema_select",
        )

        if selected_schema_val != st.session_state.selected_schema:
            st.session_state.selected_schema = selected_schema_val
            if selected_schema_val != "None":
                tables, err_tables = get_tables_postgres(st.session_state.db_conn, selected_schema_val)
                if err_tables:
                    st.sidebar.error(f"Error fetching tables: {err_tables}")
                    st.session_state.available_tables = ["None"]
                else:
                    st.session_state.available_tables = ["None"] + (tables if tables else [])
            else:
                st.session_state.available_tables = ["None"]
            st.session_state.selected_table = "None" # Reset table selection when schema changes

        if st.session_state.selected_schema and st.session_state.selected_schema != "None":
            current_table_options = st.session_state.get('available_tables', ["None"])
            if not current_table_options: # Safety for empty list
                current_table_options = ["None"]
            elif "None" not in current_table_options and current_table_options != ["None"]:
                 current_table_options = ["None"] + current_table_options

            selected_table_val = st.sidebar.selectbox(
                "Select Table",
                options=current_table_options,
                index=current_table_options.index(st.session_state.selected_table) if st.session_state.selected_table in current_table_options else 0,
                key="pg_table_select",
            )
            st.session_state.selected_table = selected_table_val
        else:
            # Ensure selected_table is reset if no schema is selected
            st.session_state.selected_table = "None"

        # Data Fetching Button
        if st.session_state.selected_schema and st.session_state.selected_schema != "None" and \
           st.session_state.selected_table and st.session_state.selected_table != "None":
            if st.sidebar.button("Fetch Data from Table", key="pg_fetch_data_button"):
                limit_val_pg = 20000 # Default limit, can be made configurable
                # limit_val_pg = st.sidebar.number_input("Max rows to fetch", min_value=100, value=20000, step=100, key="pg_fetch_limit")
                logger.info("Fetching data from %s.%s with limit %s", st.session_state.selected_schema, st.session_state.selected_table, limit_val_pg)
                df, err_fetch = fetch_data_postgres(
                    st.session_state.db_conn,
                    st.session_state.selected_schema,
                    st.session_state.selected_table,
                    limit=limit_val_pg
                )
                if err_fetch:
                    st.sidebar.error(f"Error fetching data: {err_fetch}")
                    logger.error("Failed to fetch data for %s.%s: %s", st.session_state.selected_schema, st.session_state.selected_table, err_fetch)
                    # Optionally clear data if fetch fails
                else:
                    st.session_state.data_df_original = df
                    reset_all_dependent_states(clear_data_too=False) # Reset downstream, keep new data
                    st.session_state.data_df = st.session_state.data_df_original.copy()
                    st.sidebar.success(f"Fetched data from '{st.session_state.selected_schema}.{st.session_state.selected_table}'. Shape: {df.shape}")
                    logger.info("Successfully fetched data. Shape: %s", df.shape)
        else:
            st.sidebar.info("Select a schema and table to enable data fetching.")


    # --- Event Data Upload (Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: Load Event Data")

    # UI for user to specify event column names
    # These default values will be used if not changed by user or loaded from settings
    st.session_state.time_series_specs['event_device_id_col'] = st.sidebar.text_input(
        "Event Device ID Column Name",
        value=st.session_state.time_series_specs.get('event_device_id_col', "device_id"),
        key="event_id_col_input_general"
    )
    st.session_state.time_series_specs['event_timestamp_col'] = st.sidebar.text_input(
        "Event Timestamp Column Name",
        value=st.session_state.time_series_specs.get('event_timestamp_col', "timestamp"),
        key="event_ts_col_input_general"
    )
    st.session_state.time_series_specs['event_event_type_col'] = st.sidebar.text_input(
        "Event Type Column Name",
        value=st.session_state.time_series_specs.get('event_event_type_col', "event_type"),
        key="event_type_col_input_general"
    )

    event_file = st.sidebar.file_uploader(
        "Upload Event Data File (CSV or Excel)",
        type=['csv', 'xlsx'],
        key="event_file_uploader_general"
    )
    if event_file: # This block processes the file immediately upon upload if not already processed
        # Check if this specific file instance has been processed to avoid reprocessing on every rerun
        if st.session_state.get('last_event_file_id') != event_file.id:
            try:
                if event_file.name.endswith('.csv'):
                    df_event_temp = pd.read_csv(event_file)
                else:
                    df_event_temp = pd.read_excel(event_file)

                # Use user-specified column names
                id_col_event = st.session_state.time_series_specs['event_device_id_col']
                ts_col_event = st.session_state.time_series_specs['event_timestamp_col']
                type_col_event = st.session_state.time_series_specs['event_event_type_col']
                required_event_cols = [id_col_event, ts_col_event, type_col_event]

                if not all(col in df_event_temp.columns for col in required_event_cols):
                    st.sidebar.error(f"Event data must contain specified columns: {', '.join(required_event_cols)}")
                    # st.session_state.event_df = pd.DataFrame() # Don't clear if a valid one was there before
                else:
                    df_event_temp[ts_col_event] = pd.to_datetime(df_event_temp[ts_col_event], errors='coerce')
                    df_event_temp = df_event_temp.dropna(subset=[ts_col_event])
                    # Rename columns to standard ones for internal use if needed, or use specified names throughout
                    # For now, assume functions will use these specified names by taking them as args.
                    # Or, rename them here:
                    # df_event_temp = df_event_temp.rename(columns={
                    #     id_col_event: "device_id", ts_col_event: "timestamp", type_col_event: "event_type"
                    # })
                    st.session_state.event_df = df_event_temp
                    st.session_state.last_event_file_id = event_file.id # Mark as processed
                    st.sidebar.success(f"Loaded event data: {st.session_state.event_df.shape}")
                    # Reset dependent states like global_top_event_types
                    if 'global_top_event_types_cleaned' in st.session_state:
                         del st.session_state['global_top_event_types_cleaned']
                    if 'event_df_last_loaded_id' in st.session_state: # For global event type caching
                         del st.session_state['event_df_last_loaded_id']


            except Exception as e:
                st.sidebar.error(f"Error loading event data: {e}")
                # st.session_state.event_df = pd.DataFrame() # Don't clear if valid one was there

    # --- Time Series Settings (Sidebar UI only for this part) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Time Series Settings")
    if not st.session_state.data_df_original.empty:
        df_columns = ["None"] + st.session_state.data_df_original.columns.tolist()

        # Ensure selections are valid if columns change
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

    st.sidebar.markdown("---")
    st.sidebar.header("Single Time Series Selection")

    # Get selections from session state (set by Time Series Settings UI)
    ts_specs = st.session_state.time_series_specs
    id_cols = ts_specs.get("id_cols", [])
    ts_col = ts_specs.get("timestamp_col", "None")
    value_cols_to_analyze = ts_specs.get("value_cols", [])

    if ts_col != "None" and value_cols_to_analyze:
        # Entity ID selection
        unique_ids_display = ["DefaultTimeSeries"] # Default if no ID cols or if analyzing whole dataset as one
        temp_id_col_name_single_series = "_temp_unique_id_single_series_" # Use a distinct temp col name

        if id_cols:
            if not st.session_state.data_df_original.empty:
                df_for_ids = st.session_state.data_df_original.copy()
                df_for_ids[temp_id_col_name_single_series] = df_for_ids[id_cols].astype(str).agg('-'.join, axis=1)
                unique_ids_list = sorted(df_for_ids[temp_id_col_name_single_series].unique().tolist())
                unique_ids_display = ["None"] + unique_ids_list
            else:
                unique_ids_display = ["None"] # No data to get IDs from

        current_selected_id = ts_specs.get("selected_id", "None")
        if current_selected_id not in unique_ids_display: current_selected_id = "None" if "None" in unique_ids_display else unique_ids_display[0]

        ts_specs["selected_id"] = st.sidebar.selectbox(
            "Select specific Device/Entity ID to analyze:",
            options=unique_ids_display,
            index=unique_ids_display.index(current_selected_id),
            key="single_series_selected_id_general"
        )

        # Value column selection for analysis
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
                        # Store this for profiling original segment before value_col NA drop
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

    st.sidebar.markdown("---")
    st.sidebar.header("Population Analysis Settings")

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
            st.session_state.running_all_features_computation = True  # UI flag to show "processing" section
            st.session_state.feature_computation_running = True     # Actual flag for background task status
            st.session_state.feature_computation_future = None      # Reset future object

        # Parse and validate ACF Lags from UI
        try:
            parsed_acf_lags = [int(x.strip()) for x in st.session_state.widget_fe_acf_lags.split(',') if x.strip()]
            if not parsed_acf_lags: parsed_acf_lags = [1, 5, 10] # Default if empty
            st.session_state.fe_acf_lags_general = parsed_acf_lags
        except ValueError:
            st.sidebar.error("Invalid ACF Lags format. Using last valid or default.")
            # Proceed with st.session_state.fe_acf_lags_general which holds last valid or default

        # Parse and validate Rolling Windows from UI
        try:
            parsed_rolling_windows = [int(x.strip()) for x in st.session_state.widget_fe_rolling_windows.split(',') if x.strip()]
            if not parsed_rolling_windows: parsed_rolling_windows = [1, 5, 10, 20] # Default if empty
            # Further validation for positive integers, etc., could be added here
            if any(w <= 0 for w in parsed_rolling_windows):
                st.sidebar.error("Rolling windows must be positive integers. Using last valid or default.")
            else:
                st.session_state.fe_rolling_windows_general = parsed_rolling_windows
        except ValueError:
            st.sidebar.error("Invalid Rolling Windows format. Using last valid or default.")

            # Reset downstream results that depend on all_device_features_df
            st.session_state.all_device_features_df = pd.DataFrame()
            st.session_state.population_anomaly_results = {}
            st.session_state.clustering_results = {}
            st.session_state.kmeans_stats_df = None
            st.session_state.res_df_anomalies_sorted = pd.DataFrame()
            st.session_state.surrogate_tree_explainer = None

            # Prepare data for the executor
            data_df_original_serializable = st.session_state.data_df_original.copy()
            ts_specs_serializable = st.session_state.time_series_specs.copy()
        # Add parsed ACF lags and Rolling Windows to ts_specs_serializable
        ts_specs_serializable['acf_lags'] = st.session_state.fe_acf_lags_general
        ts_specs_serializable['rolling_windows'] = st.session_state.fe_rolling_windows_general

            event_df_serializable = st.session_state.event_df.copy()
            global_top_event_types_cleaned_serializable = list(st.session_state.get("global_top_event_types_cleaned", []))

            try:
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
                st.session_state.feature_computation_future = executor.submit(
                    run_feature_engineering_for_all_devices,
                    data_df_original_serializable,
                ts_specs_serializable, # This now contains acf_lags and rolling_windows
                    event_df_serializable,
                    global_top_event_types_cleaned_serializable
                )
            executor.shutdown(wait=False)
                st.sidebar.info("Feature computation started in the background.")
            logger.info("Background feature computation task submitted with ACF Lags: %s, Rolling Windows: %s",
                        st.session_state.fe_acf_lags_general, st.session_state.fe_rolling_windows_general)
            except Exception as e:
                st.sidebar.error(f"Failed to start feature computation: {e}")
                logger.error("Failed to start feature computation task: %s", e, exc_info=True)
                st.session_state.feature_computation_running = False
            st.session_state.running_all_features_computation = False

        st.experimental_rerun()

    # --- Feature Engineering Config UI (moved slightly above Save/Load for better flow) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Feature Engineering Settings")
    st.session_state.widget_fe_acf_lags = st.sidebar.text_input(
        "ACF Lags (comma-separated integers)",
        value=",".join(map(str, st.session_state.fe_acf_lags_general)),
        key="acf_lags_input_widget" # Ensure this key is unique and not fe_acf_lags_general
    )
    st.session_state.widget_fe_rolling_windows = st.sidebar.text_input(
        "Rolling Windows (comma-separated integers)",
        value=",".join(map(str, st.session_state.fe_rolling_windows_general)),
        key="rolling_windows_input_widget" # Ensure this key is unique
    )

    # --- Save/Load App State ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Save/Load App State")

    if st.sidebar.button("Save App State", key="save_app_state_button_main_general"):
        logger.info("Saving app state.")
        app_settings = gather_settings_for_save()
        json_settings = json.dumps(app_settings, indent=4)
        # The download button is created on the fly when save is clicked.
        # Streamlit doesn't allow creating a download button that then triggers a separate callback for data generation easily.
        # So, we generate data then immediately offer it.
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
            logger.info("Loading app state from %s.", uploaded_settings_file_main.name)
            loaded_settings_dict = json.load(uploaded_settings_file_main)
            success, message = apply_loaded_settings_to_session_state(loaded_settings_dict)
            if success:
                st.sidebar.success(f"Settings loaded! {message} Applying and rerunning...")
                logger.info("App state loaded successfully.")
                # Clear the uploader so it's ready for a new file if needed, and to prevent reprocessing on simple reruns
                st.session_state.load_app_settings_uploader_main_general = None
                st.experimental_rerun()
            else:
                st.sidebar.error(message)
                logger.warning("Failed to apply loaded settings: %s", message)
        except Exception as e:
            st.sidebar.error(f"Error parsing or applying settings file: {e}")
            logger.error("Error parsing or applying settings file: %s", e, exc_info=True)
        # It's good practice to clear the file uploader after processing to avoid re-processing on every script run
        # However, direct assignment to the key in st.session_state for file_uploader is the way to "reset" it.
        # This is done above on success. If an error occurs, user might want to try again or upload different file.


    # --- Main Area Data Display ---
    st.subheader("Loaded Main Data Overview")
    if not st.session_state.data_df_original.empty:
        st.dataframe(st.session_state.data_df_original.head())
        st.info(f"Original data shape: {st.session_state.data_df_original.shape}")
        if not st.session_state.event_df.empty:
            st.subheader("Loaded Event Data Overview")
            st.dataframe(st.session_state.event_df.head())
            st.info(f"Event data shape: {st.session_state.event_df.shape}")
    else:
        st.info("Load data using the sidebar (or connect to a database - to be re-implemented).")

    # --- Main Area: Display Processed Single Series & Analysis Tabs ---
    processed_series_display = st.session_state.time_series_specs.get("processed_series")
    if processed_series_display is not None and not processed_series_display.empty:
        st.subheader(f"Selected Time Series: {processed_series_display.name} for Entity '{st.session_state.time_series_specs.get('selected_id', 'N/A')}'")

        # --- Plotting with Matplotlib and Event Overlays ---
        fig_single_ts, ax_single_ts = plt.subplots(figsize=(12, 5))
        ax_single_ts.plot(processed_series_display.index, processed_series_display.values, label=processed_series_display.name or "Value")
        ax_single_ts.set_title(f"Time Series: {processed_series_display.name or 'Selected Metric'}")
        ax_single_ts.set_xlabel("Timestamp")
        ax_single_ts.set_ylabel("Value")

        if not st.session_state.event_df.empty and ts_specs.get("selected_id") != "None":
            event_df_loaded = st.session_state.event_df
            current_device_id_plot = ts_specs["selected_id"]
            event_id_col = ts_specs.get('event_device_id_col', 'device_id')

            device_events_plot = pd.DataFrame()
            if current_device_id_plot != "DefaultTimeSeries" and event_id_col in event_df_loaded.columns:
                # Ensure consistent data types for comparison if IDs are numeric/string mixes
                device_events_plot = event_df_loaded[event_df_loaded[event_id_col].astype(str) == str(current_device_id_plot)]
            elif current_device_id_plot == "DefaultTimeSeries" and not ts_specs.get("id_cols"):
                device_events_plot = event_df_loaded

            if not device_events_plot.empty:
                plotted_event_types_legend = []
                event_ts_col_plot = ts_specs.get('event_timestamp_col', 'timestamp')
                event_type_col_plot = ts_specs.get('event_event_type_col', 'event_type')

                for _, event_row in device_events_plot.iterrows():
                    event_time = event_row[event_ts_col_plot]
                    event_type = event_row[event_type_col_plot]
                    color_map_plot = {'error': 'red', 'warning': 'orange', 'info': 'blue', 'maintenance': 'green'}
                    event_color_plot = color_map_plot.get(str(event_type).lower(), 'gray')

                    label_for_legend = f"Event: {event_type}" if event_type not in plotted_event_types_legend else None
                    ax_single_ts.axvline(x=event_time, color=event_color_plot, linestyle='--', lw=1, label=label_for_legend)
                    if label_for_legend: plotted_event_types_legend.append(event_type)
                if plotted_event_types_legend: ax_single_ts.legend(fontsize='small')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_single_ts)

        # --- Analysis Tabs for Single Series ---
        st.markdown("---")
        st.subheader("Single Series Analysis Tools")

        series_name_for_tabs = processed_series_display.name or "Selected Series"

        tab_profiling, tab_decomposition, tab_anomaly_single, tab_eng_features_single = st.tabs([
            "📊 Profiling", "📉 Decomposition", "❗ Anomaly Detection (Single)", "⚙️ Engineered Features (Single)"
        ])

        with tab_profiling:
            st.subheader(f"Profiling: {series_name_for_tabs}")
            st.write("Summary Statistics (on processed series, after NA removal and aggregation):")
            st.dataframe(get_series_summary_stats(processed_series_display))

            original_series_for_profiling = st.session_state.get("original_selected_series_for_profiling")
            if original_series_for_profiling is not None and not original_series_for_profiling.empty:
                 st.write("Missing Values Analysis (on original selected segment before value NAs removed):")
                 missing_summary = get_missing_values_summary(original_series_for_profiling)
                 st.dataframe(missing_summary)
            else:
                 st.write("Missing Values Analysis: Original segment not available or empty.")

            st.write("Stationarity Test (ADF - on processed series):")
            adf_res = perform_stationarity_test(processed_series_display)
            if isinstance(adf_res, dict) and "error" in adf_res: st.error(adf_res["error"])
            else: st.json(adf_res)


        with tab_decomposition:
            st.subheader(f"Decomposition: {series_name_for_tabs}")
            decomp_model_s = st.selectbox("Model", ["additive", "multiplicative"], key="decomp_model_s_general")
            # Attempt to infer frequency for period default or use a sensible default
            inferred_freq = pd.infer_freq(processed_series_display.index)
            default_period = 7
            if inferred_freq == 'D': default_period = 7
            elif inferred_freq == 'H': default_period = 24
            elif inferred_freq == 'M' or inferred_freq == 'ME': default_period = 12
            elif len(processed_series_display) > 20: default_period = max(2, len(processed_series_display) // 10) # Heuristic
            else: default_period = 2


            period_s = st.number_input("Period", min_value=2, value=default_period, key="decomp_period_s_general",
                                       help="Seasonality period (e.g., 7 for daily data with weekly seasonality, 12 for monthly with yearly).")
            if st.button("Decompose", key="run_decomp_s_general"):
                res_s, err_s = decompose_time_series(processed_series_display, model=decomp_model_s, period=period_s)
                if err_s: st.error(err_s)
                if res_s:
                    st.success("Decomposition successful.")
                    fig_decomp, axes_decomp = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                    axes_decomp[0].plot(res_s.observed); axes_decomp[0].set_ylabel('Observed')
                    axes_decomp[1].plot(res_s.trend); axes_decomp[1].set_ylabel('Trend')
                    axes_decomp[2].plot(res_s.seasonal); axes_decomp[2].set_ylabel('Seasonal')
                    axes_decomp[3].plot(res_s.resid); axes_decomp[3].set_ylabel('Residual')
                    plt.tight_layout()
                    st.pyplot(fig_decomp)


        with tab_anomaly_single:
            st.subheader(f"Statistical Anomaly Detection: {series_name_for_tabs}")
            method_s_anom = st.selectbox("Method", ["Z-score", "IQR"], key="anom_s_method_general")

            anomalies_s = None
            scores_s = None
            error_s = None

            if method_s_anom == "Z-score":
                thresh_s_z = st.slider("Z-score Threshold", 1.0, 5.0, 3.0, 0.1, key="anom_s_z_thresh_general")
                window_s_z = st.number_input("Rolling Window (0 for global)", min_value=0, value=0, key="anom_s_z_window_general", help="0 for global Z-score, >0 for rolling Z-score.")
                if st.button("Detect Z-score Anomalies", key="run_anom_s_z_general"):
                    anomalies_s, scores_s, error_s = detect_anomalies_zscore(processed_series_display, threshold=thresh_s_z, window=window_s_z if window_s_z > 0 else None)
            elif method_s_anom == "IQR":
                k_s_iqr = st.slider("IQR K-factor (multiplier for IQR range)", 0.5, 3.0, 1.5, 0.1, key="anom_s_iqr_k_general")
                if st.button("Detect IQR Anomalies", key="run_anom_s_iqr_general"):
                    anomalies_s, scores_s, error_s = detect_anomalies_iqr(processed_series_display, k=k_s_iqr)

            if error_s: st.error(error_s)
            if anomalies_s is not None and not error_s:
                st.success(f"Found {anomalies_s.sum()} anomalies using {method_s_anom}.")
                fig_as, ax_as = plt.subplots(figsize=(12,5))
                ax_as.plot(processed_series_display.index, processed_series_display.values, label="Original Data")
                if anomalies_s.sum() > 0:
                    ax_as.scatter(processed_series_display.index[anomalies_s], processed_series_display.values[anomalies_s], color='red', label='Anomalies', zorder=5)
                ax_as.set_title(f"{method_s_anom} Anomalies on {series_name_for_tabs}")
                ax_as.legend()
                st.pyplot(fig_as)
                if scores_s is not None:
                    st.write("Anomaly Scores/Details:")
                    st.dataframe(scores_s[anomalies_s])


        with tab_eng_features_single:
            st.subheader(f"Engineered Features: {series_name_for_tabs}")
            if st.button("Compute Features for This Series", key="compute_eng_feat_s_general"):
                single_dev_events_eng = pd.DataFrame()
                event_df_for_feat = st.session_state.get('event_df', pd.DataFrame())

                if not event_df_for_feat.empty and ts_specs.get("selected_id") != "None" and ts_specs.get("selected_id") != "DefaultTimeSeries":
                    event_id_col_fe = ts_specs.get('event_device_id_col', 'device_id')
                    # Ensure selected_id and event_id_col_fe are compatible for filtering (e.g. string comparison)
                    single_dev_events_eng = event_df_for_feat[event_df_for_feat[event_id_col_fe].astype(str) == str(ts_specs["selected_id"])]

                st.session_state.single_series_features_display, err_feat_eng = generate_all_features_for_series(
                    processed_series_display,
                    name=f"{series_name_for_tabs}_", # Prefix for feature names
                    device_event_df=single_dev_events_eng if not single_dev_events_eng.empty else None,
                    all_possible_event_types=st.session_state.get("global_top_event_types_cleaned", []), # Use cached global event types
                    event_type_col=ts_specs.get('event_event_type_col', 'event_type'),
                    event_ts_col=ts_specs.get('event_timestamp_col', 'timestamp')
                )
                if err_feat_eng: st.error(f"Error generating features: {err_feat_eng}")
                elif st.session_state.single_series_features_display is not None: st.success("Features computed.")
                else: st.warning("Feature computation returned no result or an empty result.")

            if st.session_state.get("single_series_features_display") is not None and not st.session_state.single_series_features_display.empty:
                st.dataframe(st.session_state.single_series_features_display.rename("Value"))
            elif st.session_state.get("single_series_features_display") is not None and st.session_state.single_series_features_display.empty:
                st.info("Feature computation resulted in an empty set of features (this might be normal for very short series or specific configurations).")

    # Determine global top event types if event_df is loaded
    if 'event_df' in st.session_state and not st.session_state.event_df.empty:
        # Check if event_df instance has changed or if global types not computed yet
        event_df_current_id = id(st.session_state.event_df)
        if st.session_state.get('event_df_last_loaded_id_for_global_types') != event_df_current_id or \
           'global_top_event_types_cleaned' not in st.session_state:
            try:
                event_type_col_name = ts_specs.get('event_event_type_col', 'event_type')
                if event_type_col_name in st.session_state.event_df.columns:
                    all_event_counts_main = st.session_state.event_df[event_type_col_name].value_counts()
                    num_global_event_types_main = st.number_input("Number of Top Event Types to Consider Globally", min_value=1, max_value=50, value=10, key="num_global_event_types_general", help="Define how many most frequent event types will be used as features if event data is provided.")
                    st.session_state.global_top_event_types_cleaned = [
                        str(etype).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
                        for etype in all_event_counts_main.head(num_global_event_types_main).index.tolist()
                    ]
                    st.session_state.event_df_last_loaded_id_for_global_types = event_df_current_id
                    st.caption(f"Global top event types for features: {st.session_state.global_top_event_types_cleaned}")
                else:
                    st.warning(f"Event type column '{event_type_col_name}' not found in event data. Cannot determine global event types.")
                    st.session_state.global_top_event_types_cleaned = []
            except Exception as e_global_events:
                st.warning(f"Could not determine global top event types: {e_global_events}")
                st.session_state.global_top_event_types_cleaned = []
        # If already computed for this event_df, just ensure it's a list
        elif 'global_top_event_types_cleaned' not in st.session_state:
             st.session_state.global_top_event_types_cleaned = []

    else:
        st.session_state.global_top_event_types_cleaned = []

    # --- Handle Background Feature Computation Status ---
    if st.session_state.get('feature_computation_running', False) and st.session_state.get('feature_computation_future') is not None:
        future = st.session_state.feature_computation_future
        if future.running():
            st.info("Feature computation is running in the background. Please wait... The UI remains interactive. Results will appear below once complete.")
        elif future.done():
            try:
                # Unpack the result tuple
                computed_features_df, collected_errors_list = future.result()
                st.session_state.all_device_features_df = computed_features_df
                st.session_state.feature_generation_errors = collected_errors_list # Store errors

                if computed_features_df is not None and not computed_features_df.empty:
                    st.success(f"Feature computation complete. Generated features for {len(computed_features_df)} devices/entities.")
                elif computed_features_df is not None and computed_features_df.empty and not collected_errors_list :
                     st.warning("Feature computation completed but no features were generated. All devices/entities might have been skipped due to data issues or filters.")
                elif computed_features_df is None and collected_errors_list:
                     st.error("Feature computation resulted in errors and no features.")
                     logger.error("Feature computation resulted in errors and no features DataFrame was returned.")
                else:
                     st.warning("Feature computation completed. Some features were generated, but some errors also occurred.")
                     logger.warning("Feature computation completed with some errors for individual entities.")


            except Exception as e:
                st.session_state.all_device_features_df = pd.DataFrame()
                st.session_state.feature_generation_errors = [("Overall Task", f"Feature computation task failed: {e}")]
                st.error(f"Feature computation task failed: {e}")
                logger.error("Feature computation background task failed: %s", e, exc_info=True)
            finally:
                logger.info("Feature computation background task processing finished (either completed or failed).")
                st.session_state.feature_computation_running = False
                st.session_state.feature_computation_future = None
                st.session_state.running_all_features_computation = False
                st.experimental_rerun()

    elif st.session_state.get("running_all_features_computation"):
        st.header("Device Behavior Feature Engineering (All Devices)")
        st.info("Initializing feature computation... If this message persists, check for errors in the console or sidebar.")

    # Display results and errors if available and not currently running
    if not st.session_state.get('feature_computation_running', False):
        if not st.session_state.all_device_features_df.empty:
            st.subheader("Overview of All Device Features")
            st.dataframe(st.session_state.all_device_features_df.head())
            st.info(f"Feature matrix shape: {st.session_state.all_device_features_df.shape}")

        if st.session_state.get('feature_generation_errors'):
            st.error("Some errors occurred during feature generation for individual devices/entities:")
            with st.expander("View Error Details"):
                for entity_id, error_msg in st.session_state.feature_generation_errors:
                    st.write(f"- Device/Entity '{entity_id}': {error_msg}")
                    logger.warning("Entity '%s' failed feature generation: %s", entity_id, error_msg)

        @st.cache_data # Cache the conversion to CSV
        def convert_df_to_csv(df):
            return df.to_csv().encode('utf-8')
        csv_all_features = convert_df_to_csv(st.session_state.all_device_features_df)
        st.download_button(
            label="Download All Features as CSV",
            data=csv_all_features,
            file_name="all_device_features.csv",
            mime="text/csv",
            key="download_all_features_csv_general"
        )

        # --- Population-Level Analysis Tabs ---
        st.markdown("---")
        st.header("Population-Level Analysis Tools")

        # Clean features DataFrame by dropping rows with any NaNs, as most ML models require this.
        # Keep original features in all_device_features_df for reference or other types of analysis if needed.
        features_df_cleaned = st.session_state.all_device_features_df.dropna()

        if features_df_cleaned.empty:
            st.warning("No data available for population analysis after removing rows with NaN features. Some devices might have had issues generating a full feature set.")
        else:
            st.info(f"Cleaned feature matrix for ML (NaN rows dropped): {features_df_cleaned.shape}")

            pop_tab_anomalies, pop_tab_clustering = st.tabs([
                "🕵️ Anomaly Detection (All Devices)", "🧩 Device Behavior Clustering"
            ])

            with pop_tab_anomalies:
                st.subheader("Anomaly Detection on Device Features")
                # Method selection & parameters
                anomaly_method_pop = st.selectbox(
                    "Anomaly Detection Method",
                    ["Isolation Forest", "One-Class SVM"],
                    key="anomaly_method_pop_general"
                )

                if anomaly_method_pop == "Isolation Forest":
                    st.session_state.if_contam_general = st.slider(
                        "Contamination (Isolation Forest)", 0.01, 0.5, st.session_state.if_contam_general, 0.01,
                        key="if_contam_slider_general" # Changed key
                    )
                    if st.button("Run Isolation Forest", key="run_if_pop_general"):
                        labels, scores, error = detect_anomalies_isolation_forest(features_df_cleaned, contamination=st.session_state.if_contam_general)
                        st.session_state.population_anomaly_results = {"labels": labels, "scores": scores, "error": error, "method": "Isolation Forest"}

                elif anomaly_method_pop == "One-Class SVM":
                    st.session_state.ocsvm_nu_general = st.slider(
                        "Nu (One-Class SVM - approx. outlier fraction)", 0.01, 0.5, st.session_state.ocsvm_nu_general, 0.01,
                        key="ocsvm_nu_slider_general" # Changed key
                    )
                    st.session_state.ocsvm_kernel_general = st.selectbox(
                        "Kernel", ["rbf", "linear", "poly", "sigmoid"],
                        index=["rbf", "linear", "poly", "sigmoid"].index(st.session_state.ocsvm_kernel_general),
                        key="ocsvm_kernel_selector_general" # Changed key
                    )
                    st.session_state.ocsvm_gamma_general = st.select_slider( # Changed to select_slider
                        "Gamma", options=['scale', 'auto', 0.001, 0.01, 0.1, 1],
                        value=st.session_state.ocsvm_gamma_general,
                        key="ocsvm_gamma_slider_general" # Changed key
                    )
                    if st.button("Run One-Class SVM", key="run_ocsvm_pop_general"):
                        labels, scores, error = detect_anomalies_one_class_svm(
                            features_df_cleaned, nu=st.session_state.ocsvm_nu_general, kernel=st.session_state.ocsvm_kernel_general, gamma=st.session_state.ocsvm_gamma_general
                        )
                        st.session_state.population_anomaly_results = {"labels": labels, "scores": scores, "error": error, "method": "One-Class SVM"}

                # Display anomaly results
                pop_anom_res = st.session_state.get("population_anomaly_results", {})
                if pop_anom_res:
                    if pop_anom_res.get("error"):
                        st.error(f"Error during {pop_anom_res.get('method', '')} anomaly detection: {pop_anom_res['error']}")
                    elif pop_anom_res.get("labels") is not None:
                        st.success(f"{pop_anom_res.get('method', '')} completed. Found {sum(pop_anom_res['labels'] == -1)} anomalies.")
                        res_df = features_df_cleaned.copy()
                        res_df['anomaly_label'] = pop_anom_res['labels']
                        if pop_anom_res.get("scores") is not None:
                            res_df['anomaly_score'] = pop_anom_res['scores']

                        st.session_state.res_df_anomalies_sorted = res_df.sort_values(by='anomaly_score', ascending=True if pop_anom_res.get('method') == "Isolation Forest" else False) # Lower IF scores are more anomalous
                        st.dataframe(st.session_state.res_df_anomalies_sorted[st.session_state.res_df_anomalies_sorted['anomaly_label'] == -1].head())

                        # --- Explain Anomalies Sub-section ---
                        st.markdown("---")
                        st.subheader("Explain Anomalies")
                        anomalous_devices_df = st.session_state.res_df_anomalies_sorted[st.session_state.res_df_anomalies_sorted['anomaly_label'] == -1]
                        normal_devices_df = st.session_state.res_df_anomalies_sorted[st.session_state.res_df_anomalies_sorted['anomaly_label'] == 1]

                        if not anomalous_devices_df.empty and not normal_devices_df.empty:
                            comparison_df, anova_results = compare_anomalous_vs_normal_features(anomalous_devices_df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore'),
                                                                                              normal_devices_df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore'))
                            st.write("Mean Feature Values (Anomalous vs. Normal):")
                            st.dataframe(comparison_df)
                            st.write("ANOVA F-statistic & p-value (for difference in means):")
                            st.dataframe(anova_results)

                            # Surrogate Model
                            st.markdown("##### Surrogate Decision Tree for Anomaly Explanation")
                            st.session_state.surrogate_tree_depth_general = st.slider("Max Tree Depth", 2, 10, st.session_state.surrogate_tree_depth_general, 1, key="surrogate_depth_pop_anom_general")
                            if st.button("Train Surrogate Tree", key="train_surrogate_pop_anom_general"):
                                st.session_state.surrogate_tree_explainer, error_surrogate = explain_anomalies_with_surrogate_model(
                                    features_df_cleaned, pop_anom_res['labels'], max_depth=st.session_state.surrogate_tree_depth_general
                                )
                                if error_surrogate: st.error(error_surrogate)
                                elif st.session_state.surrogate_tree_explainer: st.success("Surrogate tree trained.")

                            if st.session_state.get('surrogate_tree_explainer'):
                                tree_model = st.session_state.surrogate_tree_explainer['model']
                                tree_feature_names = st.session_state.surrogate_tree_explainer['feature_names']
                                tree_importances = pd.Series(tree_model.feature_importances_, index=tree_feature_names).sort_values(ascending=False)
                                st.write("Top differentiating features from Surrogate Tree:")
                                st.dataframe(tree_importances.head(10))

                                fig_tree, ax_tree = plt.subplots(figsize=(20, 10)) # Adjust size as needed
                                plot_tree(tree_model, feature_names=tree_feature_names, class_names=['Normal', 'Anomaly'], filled=True, rounded=True, ax=ax_tree, fontsize=10)
                                st.pyplot(fig_tree)

                                # Anomaly Summary Text (using surrogate importances if available)
                                top_anomalous_device_id = anomalous_devices_df.index[0] if not anomalous_devices_df.empty else "N/A"
                                anomaly_summary_txt = generate_anomaly_summary_text(
                                    anomalous_devices_df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore'),
                                    normal_devices_df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore'),
                                    top_anomalous_device_id,
                                    comparison_df,
                                    surrogate_tree_importances=tree_importances
                                )
                                st.write(f"Summary for most anomalous device ({top_anomalous_device_id}):")
                                st.markdown(anomaly_summary_txt)
                            else:
                                # Anomaly Summary Text (without surrogate importances)
                                top_anomalous_device_id = anomalous_devices_df.index[0] if not anomalous_devices_df.empty else "N/A"
                                anomaly_summary_txt = generate_anomaly_summary_text(
                                    anomalous_devices_df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore'),
                                    normal_devices_df.drop(columns=['anomaly_label', 'anomaly_score'], errors='ignore'),
                                    top_anomalous_device_id,
                                    comparison_df
                                )
                                st.write(f"Summary for most anomalous device ({top_anomalous_device_id}):")
                                st.markdown(anomaly_summary_txt)


                            # Event Correlation with Anomalies
                            if not st.session_state.event_df.empty and id_cols : # Check if event data and ID cols exist
                                st.markdown("##### Event Correlation with Anomalies")
                                event_corr_anom_df, err_corr_anom = analyze_event_correlations(
                                    st.session_state.event_df,
                                    pop_anom_res['labels'], # Anomaly labels
                                    features_df_cleaned.index, # Device IDs from feature matrix
                                    ts_specs.get('event_device_id_col', 'device_id'),
                                    ts_specs.get('event_event_type_col', 'event_type'),
                                    st.session_state.get("global_top_event_types_cleaned", []) # Use globally defined event types for consistency
                                )
                                if err_corr_anom: st.error(f"Event correlation error: {err_corr_anom}")
                                elif not event_corr_anom_df.empty:
                                    st.write("Mean Event Counts (Anomalous vs. Normal Devices):")
                                    st.dataframe(event_corr_anom_df)
                                else: st.info("No event correlation data generated.")
                        else:
                            st.info("Not enough normal or anomalous devices to compare features or explain.")


            with pop_tab_clustering:
                st.subheader("Device Behavior Clustering")
                # Method selection & parameters
                clustering_method_pop = st.selectbox(
                    "Clustering Method", ["K-Means", "DBSCAN"], key="clustering_method_pop_general"
                )

                # Common: Scale data option
                # Updated to use method-specific keys for scaling to persist selection correctly
                scale_data_for_clustering = False
                if clustering_method_pop == "K-Means":
                    st.session_state.scale_data_clustering_kmeans_general = st.checkbox("Scale Features (StandardScaler)", value=st.session_state.scale_data_clustering_kmeans_general, key="scale_kmeans_pop_general")
                    scale_data_for_clustering = st.session_state.scale_data_clustering_kmeans_general
                elif clustering_method_pop == "DBSCAN":
                    st.session_state.scale_data_clustering_dbscan_general = st.checkbox("Scale Features (StandardScaler)", value=st.session_state.scale_data_clustering_dbscan_general, key="scale_dbscan_pop_general")
                    scale_data_for_clustering = st.session_state.scale_data_clustering_dbscan_general

                if clustering_method_pop == "K-Means":
                    # K-Means specific: K selection (Elbow/Silhouette)
                    cols_k_stats = st.columns(2)
                    st.session_state.k_min_stats_general = cols_k_stats[0].number_input("Min K for Stats", 2, 10, st.session_state.k_min_stats_general, 1, key="k_min_pop_general")
                    st.session_state.k_max_stats_general = cols_k_stats[1].number_input("Max K for Stats", st.session_state.k_min_stats_general + 1, 20, st.session_state.k_max_stats_general, 1, key="k_max_pop_general")
                    if st.button("Calculate K-Means Stats (Elbow/Silhouette)", key="run_kmeans_stats_pop_general"):
                        k_stats_df, error_k_stats = get_kmeans_elbow_silhouette_data(
                            features_df_cleaned, range(st.session_state.k_min_stats_general, st.session_state.k_max_stats_general + 1), scale_features=scale_data_for_clustering
                        )
                        if error_k_stats: st.error(error_k_stats)
                        else: st.session_state.kmeans_stats_df = k_stats_df

                    if st.session_state.get("kmeans_stats_df") is not None:
                        st.write("Elbow Method (Inertia):")
                        st.line_chart(st.session_state.kmeans_stats_df.set_index('K')['Inertia'])
                        st.write("Silhouette Scores:")
                        st.line_chart(st.session_state.kmeans_stats_df.set_index('K')['Silhouette Score'])

                    st.session_state.kmeans_k_final_general = st.number_input("Number of Clusters (K)", 2, 20, st.session_state.kmeans_k_final_general, 1, key="k_final_pop_general")
                    if st.button("Run K-Means Clustering", key="run_kmeans_pop_general"):
                        labels, centers, error = perform_kmeans_clustering(features_df_cleaned, k=st.session_state.kmeans_k_final_general, scale_features=scale_data_for_clustering)
                        st.session_state.clustering_results = {"labels": labels, "centers": centers, "error": error, "method": "K-Means", "scaled":scale_data_for_clustering}

                elif clustering_method_pop == "DBSCAN":
                    st.session_state.dbscan_eps_general = st.slider("Epsilon (DBSCAN)", 0.1, 5.0, st.session_state.dbscan_eps_general, 0.1, key="dbscan_eps_pop_general")
                    st.session_state.dbscan_min_samples_general = st.number_input("Min Samples (DBSCAN)", 1, 20, st.session_state.dbscan_min_samples_general, 1, key="dbscan_min_samples_pop_general")
                    if st.button("Run DBSCAN Clustering", key="run_dbscan_pop_general"):
                        labels, error = perform_dbscan_clustering(features_df_cleaned, eps=st.session_state.dbscan_eps_general, min_samples=st.session_state.dbscan_min_samples_general, scale_features=scale_data_for_clustering)
                        st.session_state.clustering_results = {"labels": labels, "error": error, "method": "DBSCAN", "scaled":scale_data_for_clustering}

                # Display clustering results
                pop_cluster_res = st.session_state.get("clustering_results", {})
                if pop_cluster_res:
                    if pop_cluster_res.get("error"):
                        st.error(f"Error during {pop_cluster_res.get('method', '')} clustering: {pop_cluster_res['error']}")
                    elif pop_cluster_res.get("labels") is not None:
                        labels_c = pop_cluster_res['labels']
                        n_clusters_ = len(set(labels_c)) - (1 if -1 in labels_c else 0) # Number of clusters, excluding noise if present
                        st.success(f"{pop_cluster_res.get('method', '')} completed. Found {n_clusters_} clusters (excluding noise).")

                        # Store labels with original feature data for explanation
                        clustered_df = features_df_cleaned.copy() # Use cleaned (NaN-dropped) features
                        clustered_df['cluster_label'] = labels_c
                        st.session_state.clustered_features_df_for_explain = clustered_df # Save for explainability

                        st.write("Cluster Counts:")
                        st.dataframe(pd.Series(labels_c).value_counts().rename("Count"))
                        if pop_cluster_res.get("method") == "K-Means" and pop_cluster_res.get("centers") is not None:
                            st.write("Cluster Centers:")
                            st.dataframe(pop_cluster_res["centers"])

                        # --- Explain Clusters Sub-section ---
                        st.markdown("---")
                        st.subheader("Explain Clusters")

                        # Feature Importance (ANOVA)
                        anova_res_clusters, err_anova_c = get_feature_importance_for_clusters_anova(clustered_df, 'cluster_label')
                        if err_anova_c: st.warning(f"Could not compute ANOVA for cluster explanation: {err_anova_c}")
                        elif not anova_res_clusters.empty:
                            st.write("Feature Importance for Distinguishing Clusters (ANOVA F-value, lower p-value is better):")
                            st.dataframe(anova_res_clusters.sort_values(by='F-statistic', ascending=False).head(10))

                        # Cluster Feature Summary (Mean values)
                        cluster_summary_df, overall_means_df, err_summary_c = get_cluster_feature_summary(clustered_df, 'cluster_label')
                        if err_summary_c: st.warning(f"Could not compute cluster feature summary: {err_summary_c}")
                        elif not cluster_summary_df.empty:
                            st.write("Mean Feature Values per Cluster:")
                            st.dataframe(cluster_summary_df)

                            # Textual Summary for each cluster
                            for cluster_id in sorted(cluster_summary_df.index.tolist()):
                                if cluster_id == -1 and pop_cluster_res.get("method") == "DBSCAN": continue # Skip noise for DBSCAN summary text
                                summary_text_c = generate_cluster_summary_text(
                                    cluster_id, cluster_summary_df, overall_means_df,
                                    top_n_features=5, anova_results=anova_res_clusters
                                )
                                with st.expander(f"Detailed Summary for Cluster {cluster_id}"):
                                    st.markdown(summary_text_c)

                        # Event Correlation with Clusters
                        if not st.session_state.event_df.empty and id_cols: # Check if event data and ID cols exist
                            st.markdown("##### Event Correlation with Clusters")
                            event_corr_cluster_df, err_corr_cluster = analyze_event_correlations(
                                st.session_state.event_df,
                                labels_c, # Cluster labels
                                features_df_cleaned.index, # Device IDs from feature matrix
                                ts_specs.get('event_device_id_col', 'device_id'),
                                ts_specs.get('event_event_type_col', 'event_type'),
                                st.session_state.get("global_top_event_types_cleaned", []) # Use globally defined event types
                            )
                            if err_corr_cluster: st.error(f"Event correlation error for clusters: {err_corr_cluster}")
                            elif not event_corr_cluster_df.empty:
                                st.write("Mean Event Counts per Cluster:")
                                st.dataframe(event_corr_cluster_df)
                            else: st.info("No event correlation data generated for clusters.")

    # --- Validate Findings with Known Failures ---
    if not st.session_state.get('all_device_features_df', pd.DataFrame()).empty:
        st.markdown("---")
        st.header("🔬 Validate Findings with Known Failures")
        st.markdown("""
        If you have a list of Device/Entity IDs that are known to have experienced failures or specific issues,
        you can input them here to see how they align with the anomaly detection and clustering results.
        This helps assess the relevance of the automated findings.
        """)

        if 'failed_ids_text_area_general' not in st.session_state:
            st.session_state.failed_ids_text_area_general = ""

        failed_ids_input_general = st.text_area(
            "Enter known problematic Device/Entity IDs (comma, semicolon, or newline separated):",
            value=st.session_state.failed_ids_text_area_general,
            key="failed_ids_text_area_general_widget", # Ensure a unique key for the widget itself
            height=100
        )

        if st.button("Run Validation Analysis", key="run_validation_button_general"):
            if failed_ids_input_general.strip():
                # Parse IDs: handles comma, semicolon, or newline, and strips whitespace
                parsed_known_failed_ids = [
                    item.strip() for item in
                    failed_ids_input_general.replace(',', ' ').replace(';', ' ').split()
                    if item.strip()
                ]
                st.session_state.parsed_known_failed_ids_general = parsed_known_failed_ids # Save for potential reuse
                st.write(f"Found {len(parsed_known_failed_ids)} unique known failed IDs for validation.")

                # Validate against Anomaly Detection Results
                pop_anom_res_val = st.session_state.get("population_anomaly_results", {})
                res_df_anomalies_sorted_val = st.session_state.get("res_df_anomalies_sorted", pd.DataFrame())

                if pop_anom_res_val and not res_df_anomalies_sorted_val.empty:
                    st.subheader("Validation Against Anomaly Detection")
                    anomalous_flagged_known_failures = res_df_anomalies_sorted_val[
                        res_df_anomalies_sorted_val.index.isin(parsed_known_failed_ids) &
                        (res_df_anomalies_sorted_val['anomaly_label'] == -1)
                    ]
                    num_caught = len(anomalous_flagged_known_failures)
                    st.metric(
                        label=f"Known Failures Flagged as Anomalous (by {pop_anom_res_val.get('method', 'N/A')})",
                        value=f"{num_caught} / {len(parsed_known_failed_ids)}"
                    )
                    if num_caught > 0:
                        st.write("Details of flagged known failures:")
                        st.dataframe(anomalous_flagged_known_failures[['anomaly_label', 'anomaly_score']])
                    else:
                        st.info("None of the provided known failed IDs were flagged as anomalous by the current settings.")
                else:
                    st.warning("Anomaly detection results not available for validation. Please run population-level anomaly detection first.")

                # Validate against Clustering Results
                pop_cluster_res_val = st.session_state.get("clustering_results", {})
                clustered_features_df_val = st.session_state.get("clustered_features_df_for_explain", pd.DataFrame())

                if pop_cluster_res_val and not clustered_features_df_val.empty:
                    st.subheader("Validation Against Clustering Results")
                    known_failures_in_clusters = clustered_features_df_val[
                        clustered_features_df_val.index.isin(parsed_known_failed_ids)
                    ]
                    if not known_failures_in_clusters.empty:
                        st.write("Distribution of Known Failures Across Clusters:")
                        st.dataframe(known_failures_in_clusters['cluster_label'].value_counts().rename("Count of Known Failures"))

                        # Show full details of known failures and their assigned clusters
                        st.write("Details of known failures and their assigned clusters:")
                        st.dataframe(known_failures_in_clusters[['cluster_label'] + [col for col in known_failures_in_clusters.columns if col not in ['cluster_label']][:3]]) # Show label + first 3 features
                    else:
                        st.info("None of the provided known failed IDs were found in the clustered devices (they might have been filtered out due to NaN features).")
                else:
                    st.warning("Clustering results not available for validation. Please run population-level clustering first.")
            else:
                st.warning("Please enter some known failed Device/Entity IDs.")
        st.markdown("---")


# === GUIDED WORKFLOWS MODE ===
elif app_mode == "Guided Workflows":
    st.header("Guided Workflows Mode")
    # Sidebar for workflow selection
    available_workflows = ["Potential Failure Investigation"]
    # Initialize selected_workflow in session_state if not present
    if 'selected_workflow' not in st.session_state:
        st.session_state.selected_workflow = available_workflows[0]

    # Use on_change to reset workflow-specific state if workflow changes
    def reset_workflow_state():
        # Example: clear selected clusters if workflow changes or is re-selected
        if 'at_risk_clusters_wf' in st.session_state:
            del st.session_state['at_risk_clusters_wf']
        # Add other workflow-specific state variables here to reset

    selected_workflow = st.sidebar.selectbox(
        "Choose a Guided Workflow:",
        available_workflows,
        key='selected_workflow_dropdown', # Main key for the widget
        on_change=reset_workflow_state
    )
    st.session_state.selected_workflow = selected_workflow # Keep it synced with session_state for logic

    # --- Potential Failure Investigation Workflow ---
    if st.session_state.selected_workflow == "Potential Failure Investigation":
        st.subheader("Guided Workflow: Potential Failure Investigation")
        st.markdown("""
        This workflow helps identify devices that might be at higher risk of failure by:
        1.  Leveraging existing clustering results (from General Analysis mode).
        2.  Allowing you to select 'at-risk' clusters.
        3.  Showing what makes these clusters different.
        4.  Listing devices in these at-risk clusters for further monitoring.
        5.  Optionally cross-referencing with a list of already known failed devices.
        """)

        # Step 1: Prerequisites Check
        st.markdown("---")
        st.subheader("Step 1: Prerequisites")
        prereq_features_df = st.session_state.get('all_device_features_df', pd.DataFrame())
        prereq_clustering_results = st.session_state.get('clustering_results', {})
        prereq_clustered_df_for_explain = st.session_state.get('clustered_features_df_for_explain', pd.DataFrame())

        prereq_met = True
        if prereq_features_df.empty:
            st.error("Prerequisite Not Met: Device features have not been computed. Please go to 'General Analysis' mode, compute features for all devices, and run clustering.")
            prereq_met = False
        if not prereq_clustering_results or 'labels' not in prereq_clustering_results:
            st.error("Prerequisite Not Met: Clustering has not been performed or results are unavailable. Please run clustering in 'General Analysis' mode.")
            prereq_met = False
        if prereq_clustered_df_for_explain.empty:
            st.error("Prerequisite Not Met: Clustered feature data for explanation is missing. Ensure clustering was run successfully.")
            prereq_met = False

        if not prereq_met:
            st.stop()
        else:
            st.success("Prerequisites met: Feature data and clustering results are available.")

        # Step 2: Select At-Risk Cluster(s)
        st.markdown("---")
        st.subheader("Step 2: Select At-Risk Cluster(s)")

        # Ensure clustered_df is available from session state (set during clustering in General Analysis)
        clustered_df_wf = st.session_state.get('clustered_features_df_for_explain', pd.DataFrame())
        if 'cluster_label' not in clustered_df_wf.columns:
            st.error("Clustering labels not found in the feature data. Please re-run clustering.")
            st.stop()

        available_clusters_wf = sorted([c for c in clustered_df_wf['cluster_label'].unique() if c != -1]) # Exclude noise points for selection
        if not available_clusters_wf:
            st.warning("No valid clusters found (excluding noise points if any). Cannot proceed.")
            st.stop()

        # Initialize selected at_risk_clusters in session_state if not present
        if 'at_risk_clusters_wf' not in st.session_state:
             st.session_state.at_risk_clusters_wf = []

        # Ensure selections are valid if clusters change (e.g. re-clustering in General mode)
        valid_at_risk_selection = [c for c in st.session_state.at_risk_clusters_wf if c in available_clusters_wf]


        selected_at_risk_clusters = st.multiselect(
            "Select cluster(s) you consider 'at-risk' based on previous analysis or domain knowledge:",
            options=available_clusters_wf,
            default=valid_at_risk_selection, # Use the validated list
            key='at_risk_clusters_multiselect_wf' # Unique key for this widget
        )
        st.session_state.at_risk_clusters_wf = selected_at_risk_clusters # Update session state

        if not selected_at_risk_clusters:
            st.info("Please select one or more at-risk clusters to proceed.")
            st.stop()

        st.success(f"Selected at-risk clusters: {selected_at_risk_clusters}")

        # Step 3: Key Characteristics of Selected At-Risk Cluster(s)
        st.markdown("---")
        st.subheader("Step 3: Distinguishing Characteristics of At-Risk Group")

        # Create a temporary 'at_risk_group' column: True if in selected_at_risk_clusters, False otherwise
        # Only consider non-noise points for this comparison
        comparison_df_wf = clustered_df_wf[clustered_df_wf['cluster_label'] != -1].copy()
        comparison_df_wf['at_risk_group'] = comparison_df_wf['cluster_label'].isin(selected_at_risk_clusters)

        # ANOVA for at-risk vs not-at-risk
        anova_at_risk_vs_others, err_anova_wf = get_feature_importance_for_clusters_anova(comparison_df_wf, 'at_risk_group', group_true_label=True, group_false_label=False)
        if err_anova_wf:
            st.warning(f"Could not compute ANOVA for at-risk group comparison: {err_anova_wf}")
        elif not anova_at_risk_vs_others.empty:
            st.write("Top Features Distinguishing At-Risk Group (True) from Other Non-Noise Clusters (False):")
            st.dataframe(anova_at_risk_vs_others.sort_values(by='F-statistic', ascending=False).head(10))

        # Mean feature values for at-risk vs not-at-risk
        at_risk_summary_df, others_summary_df, err_summary_wf = get_cluster_feature_summary(
            comparison_df_wf, 'at_risk_group', group_true_label=True, group_false_label=False
        )

        if err_summary_wf:
            st.warning(f"Could not compute feature summary for at-risk group comparison: {err_summary_wf}")
        elif not at_risk_summary_df.empty and not others_summary_df.empty:
            st.write("Mean Feature Values for At-Risk Group vs. Other Non-Noise Clusters:")
            # Combine for easier side-by-side view - at_risk_summary_df and others_summary_df are Series
            combined_summary_wf = pd.DataFrame({
                "At-Risk Group (Mean)": at_risk_summary_df,
                "Other Non-Noise Clusters (Mean)": others_summary_df
            })
            st.dataframe(combined_summary_wf)

        # Step 4: Devices to Monitor
        st.markdown("---")
        st.subheader("Step 4: Devices to Monitor")

        devices_in_at_risk_clusters = clustered_df_wf[clustered_df_wf['cluster_label'].isin(selected_at_risk_clusters)].index.tolist()
        st.write(f"Found {len(devices_in_at_risk_clusters)} devices in the selected at-risk cluster(s): {', '.join(map(str,selected_at_risk_clusters))}")

        if 'failed_ids_input_wf' not in st.session_state:
            st.session_state.failed_ids_input_wf = ""

        known_failed_ids_input_wf = st.text_area(
            "Optional: Enter known failed Device/Entity IDs (comma, semicolon, or newline separated) to cross-reference:",
            value=st.session_state.failed_ids_input_wf,
            key="failed_ids_text_area_wf_widget", # Unique key
            height=100
        )

        parsed_known_failed_ids_wf = []
        if known_failed_ids_input_wf.strip():
            parsed_known_failed_ids_wf = [
                item.strip() for item in
                known_failed_ids_input_wf.replace(',', ' ').replace(';', ' ').split()
                if item.strip()
            ]

        devices_to_monitor_df_data = []
        for device_id in devices_in_at_risk_clusters:
            status = "Potentially At-Risk"
            if device_id in parsed_known_failed_ids_wf:
                status = "Known Failed & In At-Risk Cluster"
            devices_to_monitor_df_data.append({"Device ID": device_id, "Status": status, "Assigned At-Risk Cluster(s)": str(clustered_df_wf.loc[device_id, 'cluster_label'])})

        if devices_to_monitor_df_data:
            devices_to_monitor_df = pd.DataFrame(devices_to_monitor_df_data).set_index("Device ID")
            st.dataframe(devices_to_monitor_df)

            # Highlight those not on the known list but in at-risk clusters
            newly_identified_at_risk = devices_to_monitor_df[devices_to_monitor_df["Status"] == "Potentially At-Risk"]
            if not newly_identified_at_risk.empty:
                st.write("Newly Identified Potentially At-Risk Devices (not on your known failed list):")
                st.dataframe(newly_identified_at_risk)
        else:
            st.info("No devices found in the selected at-risk clusters.")

    # Add placeholders for other workflows if any
    # elif st.session_state.selected_workflow == "Another Workflow":
    #    st.write("Another workflow placeholder")


# (The rest of src/main.py logic for General Analysis tabs will be added in subsequent parts)
