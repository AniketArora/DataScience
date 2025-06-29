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

# --- Global Constants / Configs ---
TEMP_ID_COL_NAME = "_temp_unique_id_" # Define temp_id_col_name globally

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
from database import connect_postgres, get_schemas_postgres, get_tables_postgres, fetch_data_postgres
from analysis_modules.profiling import (
    get_series_summary_stats,
    get_missing_values_summary,
    perform_stationarity_test
)
from analysis_modules.decomposition import decompose_time_series
from analysis_modules.anomalies import (
    detect_anomalies_zscore,
    detect_anomalies_iqr,
    # detect_anomalies_isolation_forest, # Now part of AnomalyDetectionAnalysisModule
    # detect_anomalies_one_class_svm, # Now part of AnomalyDetectionAnalysisModule
    AnomalyDetectionAnalysisModule
)
from analysis_modules.feature_engineering import generate_all_features_for_series, run_feature_engineering_for_all_devices
from analysis_modules.clustering import (
    perform_kmeans_clustering,
    # perform_kmeans_clustering, # Now part of ClusteringAnalysisModule
    # get_kmeans_elbow_silhouette_data, # Now part of ClusteringAnalysisModule
    # perform_dbscan_clustering, # Now part of ClusteringAnalysisModule
    ClusteringAnalysisModule
)
from analysis_modules.explainability import (
    get_cluster_feature_summary,
    get_feature_importance_for_clusters_anova,
    compare_anomalous_vs_normal_features,
    generate_cluster_summary_text,
    generate_anomaly_summary_text,
    explain_anomalies_with_surrogate_model
)
from sklearn.tree import plot_tree # For visualizing surrogate tree
from config_utils import (
    gather_settings_for_save,
    apply_loaded_settings_to_session_state,
    # Assuming these key lists are defined in config_utils as per previous setup
    # If not, they would need to be defined here or the functions adapted
)

# --- Main App Title ---
st.title("Universal Data Analyzer 📊")

# --- Global State Initialization ---
if 'db_conn' not in st.session_state: st.session_state.db_conn = None
if 'data_df_original' not in st.session_state: st.session_state.data_df_original = pd.DataFrame()
if 'data_df' not in st.session_state: st.session_state.data_df = pd.DataFrame()
if 'event_df' not in st.session_state: st.session_state.event_df = pd.DataFrame() # New for event data
if 'active_filters' not in st.session_state: st.session_state.active_filters = {}
if 'time_series_specs' not in st.session_state:
    st.session_state.time_series_specs = {"id_cols": [], "timestamp_col": "None", "value_cols": [], "selected_id": "None", "selected_value_col_for_analysis": "None", "processed_series": None}
if 'all_device_features_df' not in st.session_state: st.session_state.all_device_features_df = pd.DataFrame()
if 'single_series_features_display' not in st.session_state: st.session_state.single_series_features_display = None
if 'running_all_features_computation' not in st.session_state: st.session_state.running_all_features_computation = False
if 'population_anomaly_results' not in st.session_state: st.session_state.population_anomaly_results = {}
if 'clustering_results' not in st.session_state: st.session_state.clustering_results = {}
if 'kmeans_stats_df' not in st.session_state: st.session_state.kmeans_stats_df = None
if 'last_pop_anomaly_df_id' not in st.session_state: st.session_state.last_pop_anomaly_df_id = None
if 'last_clustering_df_id' not in st.session_state: st.session_state.last_clustering_df_id = None
if 'res_df_anomalies_sorted' not in st.session_state: st.session_state.res_df_anomalies_sorted = pd.DataFrame()
if 'surrogate_tree_explainer' not in st.session_state: st.session_state.surrogate_tree_explainer = None

# --- Utility for resetting states ---
def reset_ts_and_temp_cols():
    # temp_id_col_name is now global (TEMP_ID_COL_NAME)
    if TEMP_ID_COL_NAME in st.session_state.data_df_original.columns:
        try: del st.session_state.data_df_original[TEMP_ID_COL_NAME]
        except KeyError: pass
    if TEMP_ID_COL_NAME in st.session_state.data_df.columns:
         try: del st.session_state.data_df[TEMP_ID_COL_NAME]
         except KeyError: pass
    st.session_state.time_series_specs = {"id_cols": [], "timestamp_col": "None", "value_cols": [], "selected_id": "None", "selected_value_col_for_analysis": "None", "processed_series": None}
    st.session_state.active_filters = {}
    st.session_state.all_device_features_df = pd.DataFrame()
    st.session_state.single_series_features_display = None
    st.session_state.population_anomaly_results = {}
    st.session_state.clustering_results = {}
    st.session_state.kmeans_stats_df = None
    st.session_state.last_pop_anomaly_df_id = None
    st.session_state.last_clustering_df_id = None
    st.session_state.res_df_anomalies_sorted = pd.DataFrame()
    st.session_state.surrogate_tree_explainer = None
    st.session_state.event_df = pd.DataFrame() # Reset event_df
    st.session_state.global_top_event_types_cleaned = [] # Reset global event types


# --- APP MODE SELECTION (SIDEBAR) ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Choose Mode:",
    ("General Analysis", "Guided Workflows"),
    key="app_mode_selector",
    on_change=reset_all_dependent_states, args=(True,) # Reset all on mode change for safety
)
st.sidebar.markdown("---")

# --- CONDITIONAL UI BASED ON APP MODE ---

if app_mode == "Guided Workflows":
    # ... (Guided Workflow logic as defined in previous step) ...
    st.sidebar.header("Available Workflows")
    selected_workflow = st.sidebar.selectbox( "Select a Guided Workflow:", ["Potential Failure Investigation"], key="guided_workflow_selector")
    st.sidebar.markdown("---")
    st.header(f"Guided Workflow: {selected_workflow}")
    if selected_workflow == "Potential Failure Investigation":
        st.subheader("Step 1: Prerequisites")
        prereq_met = True
        if st.session_state.get('all_device_features_df', pd.DataFrame()).empty:
            st.warning("Compute 'All Device Features' first in 'General Analysis' mode."); prereq_met = False
        clustering_results_wf = st.session_state.get('clustering_results', {})
        if not clustering_results_wf or not isinstance(clustering_results_wf.get('labels'), pd.Series) or clustering_results_wf['labels'].empty:
            st.warning("Run Device Behavior Clustering first in 'General Analysis' mode."); prereq_met = False
        if not prereq_met: st.stop()
        st.success("Prerequisites met.")
        st.subheader("Step 2: Select At-Risk Cluster(s)")
        cluster_labels_series_wf = st.session_state.clustering_results['labels']
        available_clusters_wf = sorted(cluster_labels_series_wf.unique()); available_clusters_str_wf = [str(c) for c in available_clusters_wf]
        selected_at_risk_clusters_str_wf = st.multiselect("Select 'at-risk' cluster(s):", options=available_clusters_str_wf, key="at_risk_cluster_multiselect_wf")
        selected_at_risk_clusters_wf = []
        for s_cluster in selected_at_risk_clusters_str_wf:
            try: selected_at_risk_clusters_wf.append(int(s_cluster))
            except ValueError:
                try: selected_at_risk_clusters_wf.append(float(s_cluster))
                except ValueError: st.error(f"Could not parse cluster label: {s_cluster}")
        if not selected_at_risk_clusters_wf: st.info("Select cluster(s) to proceed."); st.stop()
        st.subheader("Step 3: Key Characteristics of Selected At-Risk Cluster(s)")
        features_df_cleaned_wf = st.session_state.all_device_features_df.dropna()
        if features_df_cleaned_wf.empty: st.error("Cleaned features empty."); st.stop()
        combined_risk_labels_wf = cluster_labels_series_wf.isin(selected_at_risk_clusters_wf).map({True: 'At-Risk', False: 'Not At-Risk'})
        if combined_risk_labels_wf.nunique() > 1:
            importance_df_wf, error_imp_wf = get_feature_importance_for_clusters_anova(features_df_cleaned_wf, combined_risk_labels_wf)
            if error_imp_wf: st.warning(f"ANOVA error: {error_imp_wf}")
            elif importance_df_wf is not None and not importance_df_wf.empty: st.write("Top Distinguishing Features (At-Risk vs. Others):"); st.dataframe(importance_df_wf.head(5))
            else: st.info("No significant distinguishing features (ANOVA).")
        else: st.info("Only one group for ANOVA.")
        mean_summary_df_wf, error_means_wf = get_cluster_feature_summary(features_df_cleaned_wf, combined_risk_labels_wf)
        if error_means_wf: st.warning(f"Mean summary error: {error_means_wf}")
        elif mean_summary_df_wf is not None and not mean_summary_df_wf.empty: st.write("Mean Features (At-Risk vs. Others):"); st.dataframe(mean_summary_df_wf)
        st.subheader("Step 4: Devices to Monitor")
        devices_in_at_risk_clusters_wf = cluster_labels_series_wf[cluster_labels_series_wf.isin(selected_at_risk_clusters_wf)].index.tolist()
        if not devices_in_at_risk_clusters_wf: st.info("No devices in selected cluster(s).")
        else:
            # st.session_state.db_conn = conn # conn is not defined here, commenting out
            st.sidebar.warning("DB connection for Guided Workflows needs setup (conn variable).")
            # logger.info("PostgreSQL connection successful.") # Only log if connection attempt is made
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

            # Corrected indentation for this block
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
            st.sidebar.error(f"Error parsing settings file: {e}")

    # This needs to be structured correctly: the following elif and else are for the "if not st.session_state.data_df_original.empty:" block
    elif not st.session_state.data_df_original.empty and TEMP_ID_COL_NAME in st.session_state.data_df_original.columns: # Use global TEMP_ID_COL_NAME
         if (st.session_state.time_series_specs.get("timestamp_col","None")=="None" or not st.session_state.time_series_specs.get("value_cols",[])):
                try: del st.session_state.data_df_original[TEMP_ID_COL_NAME] # Use global TEMP_ID_COL_NAME
                except KeyError:pass
                if TEMP_ID_COL_NAME in st.session_state.data_df.columns: # Use global TEMP_ID_COL_NAME
                    try: del st.session_state.data_df[TEMP_ID_COL_NAME] # Use global TEMP_ID_COL_NAME
                    except KeyError:pass
    else: st.sidebar.info("Load data for TS settings.")
    st.sidebar.markdown("---"); st.sidebar.header("Population Analysis Settings")
    compute_all_button_disabled_general = not (st.session_state.time_series_specs.get("timestamp_col")!="None" and st.session_state.time_series_specs.get("selected_value_col_for_analysis")!="None" and not st.session_state.data_df_original.empty)
    if compute_all_button_disabled_general: st.sidebar.warning("Load data & select Timestamp & Value in 'Time Series Settings' for population features.")
    if st.sidebar.button("Compute Features for ALL Devices", key="compute_all_features_button_general", disabled=compute_all_button_disabled_general):
        st.session_state.running_all_features_computation=True; st.session_state.all_device_features_df=pd.DataFrame(); st.session_state.population_anomaly_results={}; st.session_state.clustering_results={}; st.session_state.kmeans_stats_df=None; st.session_state.res_df_anomalies_sorted=pd.DataFrame(); st.session_state.surrogate_tree_explainer = None
        # Determine global top event types if event_df is loaded
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
            tab_pop_anomalies, tab_pop_clustering = st.tabs(["🕵️ Anomaly Detection (All Devices)", "🧩 Device Behavior Clustering"])
            with tab_pop_anomalies:
                st.subheader("Unsupervised Anomaly Detection on Device Features")
                if "last_pop_anomaly_df_id" not in st.session_state or id(features_df_cleaned) != st.session_state.get("last_pop_anomaly_df_id"): st.session_state.population_anomaly_results = {}; st.session_state.last_pop_anomaly_df_id = id(features_df_cleaned); st.session_state.surrogate_tree_explainer = None
                anomaly_method_pop = st.selectbox("Method", ["Isolation Forest", "One-Class SVM"], key="pop_an_method_general")
                if anomaly_method_pop == "Isolation Forest":
                    contam_if = st.slider("Contamination", 0.01, 0.5, 0.1, 0.01, key="if_contam_general")
                    # Removed individual save/load placeholder for Isolation Forest
                    if st.button("Run Isolation Forest", key="run_if_pop_general"):
                        labels, scores, error_msg = detect_anomalies_isolation_forest(features_df_cleaned, contamination=contam_if)
                        if error_msg:
                            st.error(error_msg)
                        else:
                            st.session_state.population_anomaly_results = {'method':'Isolation Forest', 'labels':labels, 'scores':scores}
                            st.success("IForest complete.")
                elif anomaly_method_pop == "One-Class SVM":
                    nu_ocsvm = st.slider("Nu", 0.01, 0.5, 0.05, 0.01, key="ocsvm_nu_general"); kernel_ocsvm = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="ocsvm_kernel_general"); gamma_ocsvm_text = st.text_input("Gamma", value='scale', key="ocsvm_gamma_general");
                    try: gamma_val = float(gamma_ocsvm_text)
                    except ValueError: gamma_val = gamma_ocsvm_text
                    # Removed individual save/load placeholder for One-Class SVM
                    if st.button("Run One-Class SVM", key="run_ocsvm_pop_general"):
                        labels, scores, error_msg = detect_anomalies_one_class_svm(features_df_cleaned, nu=nu_ocsvm, kernel=kernel_ocsvm, gamma=gamma_val)
                        if error_msg:
                            st.error(error_msg)
                        else:
                            st.session_state.population_anomaly_results = {'method':'One-Class SVM', 'labels':labels, 'scores':scores}
                            st.success("OC-SVM complete.")
                pop_results = st.session_state.get('population_anomaly_results')
                if pop_results and 'labels' in pop_results:
                    st.write(f"Results: {pop_results['method']}"); res_df = pd.DataFrame({'label':pop_results['labels'], 'score':pop_results['scores']}).sort_values(by='score'); st.session_state.res_df_anomalies_sorted = res_df
                    st.write(f"Found {(res_df['label'] == -1).sum()} anomalies."); st.dataframe(res_df.head())
                    fig, ax = plt.subplots(); scores_sorted = pop_results['scores'].sort_values(); scores_sorted.plot(kind='bar', ax=ax, title=f"Scores ({pop_results['method']})"); ax.set_xticks([]); ax.set_ylabel("Score"); st.pyplot(fig)
                    st.subheader("Explain Anomalies"); comparison_df, error_comp = compare_anomalous_vs_normal_features(features_df_cleaned, pop_results["labels"], anomalous_label_val=-1)
                    if error_comp: st.error(f"Comparison error: {error_comp}")
                    elif comparison_df is not None and not comparison_df.empty: st.write("Mean Feature Comparison (Anomalous vs. Normal):"); st.dataframe(comparison_df.head(10))

                    # Updated call to generate_anomaly_summary_text
                    if not st.session_state.res_df_anomalies_sorted.empty:
                        top_anomaly_id = st.session_state.res_df_anomalies_sorted.index[0]
                        top_anomaly_score = st.session_state.res_df_anomalies_sorted.iloc[0]["score"]
                        surrogate_importances_for_summary = st.session_state.get("surrogate_tree_explainer", {}).get("importances")

                        st.markdown("---"); st.markdown(f"**Summary for Top Anomaly ({top_anomaly_id}):**")
                        summary_text = generate_anomaly_summary_text(
                            top_anomaly_id,
                            top_anomaly_score,
                            comparison_df, # This is the compare_anomalous_vs_normal_features output
                            surrogate_tree_importances=surrogate_importances_for_summary
                        ); st.markdown(summary_text)
                    else: st.info("No data for feature comparison (for summary text).")
                    st.markdown("---"); st.subheader("Detailed Anomaly Explanation (Surrogate Decision Tree)"); st.markdown("This trains a Decision Tree to mimic the unsupervised anomaly detector.")
                    if st.session_state.get("population_anomaly_results") and "labels" in st.session_state.population_anomaly_results and not features_df_cleaned.empty:
                        surrogate_max_depth = st.slider("Surrogate Tree Max Depth", 2, 10, 4, 1, key="surrogate_tree_depth_slider_general", help="Complexity of explanation tree.")
                        if st.button("Train & Show Surrogate Explanation Tree", key="train_surrogate_tree_button_general"):
                            current_pop_anomaly_labels = st.session_state.population_anomaly_results["labels"]
                            if current_pop_anomaly_labels.nunique() < 2: st.error("Need at least two classes (anomalous/normal) for surrogate tree."); st.session_state.surrogate_tree_explainer = None
                            else:
                                with st.spinner("Training surrogate tree..."):
                                    tree_model, tree_importances, tree_report, error_tree = explain_anomalies_with_surrogate_model(features_df_cleaned, current_pop_anomaly_labels, max_depth=surrogate_max_depth, test_size=0)
                                    if error_tree: st.error(f"Surrogate tree error: {error_tree}"); st.session_state.surrogate_tree_explainer = None
                                    elif tree_model and tree_importances is not None: st.session_state.surrogate_tree_explainer = {"model": tree_model, "importances": tree_importances, "report": tree_report }; st.success("Surrogate tree trained.")
                                    else: st.warning("Surrogate training produced no model/importances."); st.session_state.surrogate_tree_explainer = None
                        if st.session_state.get("surrogate_tree_explainer"):
                            explainer_results = st.session_state.surrogate_tree_explainer
                            if explainer_results.get("model"):
                                st.write("**Surrogate Tree - Top Feature Importances:**"); st.dataframe(explainer_results["importances"].head(10).rename("Importance"))
                                st.write("**Surrogate Decision Tree Visualization:**"); fig_tree, ax_tree = plt.subplots(figsize=(20, 10));
                                try:
                                    class_names_tree = ["Anomalous", "Normal"];
                                    if set(explainer_results["model"].classes_) != {-1, 1} and len(explainer_results["model"].classes_) == 2: c0, c1 = sorted(explainer_results["model"].classes_); class_names_tree = [f"Class {c0}", f"Class {c1}"]
                                    plot_tree(explainer_results["model"], filled=True, rounded=True, feature_names=features_df_cleaned.columns.tolist(), class_names=class_names_tree, ax=ax_tree, fontsize=10, max_depth=surrogate_max_depth )
                                    st.pyplot(fig_tree)
                                except Exception as e_plot: st.error(f"Error plotting tree: {e_plot}")
                            else: st.info("Surrogate tree model not available.")
                    else: st.info("Run Population Anomaly Detection for surrogate tree explanations.")

                    # Event Correlations for Anomalies
                    event_cols_in_features_df_anom = [col for col in features_df_cleaned.columns if col.startswith(st.session_state.time_series_specs.get("selected_value_col_for_analysis", "value") + "_evt_count_")] # Assuming prefix structure
                    if event_cols_in_features_df_anom: # Check if any event count features exist
                        st.markdown("---")
                        st.subheader("Event Correlations with Anomalies")
                        st.warning("Event correlation analysis for anomalies is temporarily disabled (analyze_event_correlations function not found).")
                        # event_corr_anom_df, error_eca = analyze_event_correlations(
                        #     features_df_cleaned,
                        #     pop_results["labels"], # current_anomaly_labels
                        #     event_feature_prefix=f"{st.session_state.time_series_specs.get('selected_value_col_for_analysis', 'value')}_evt_count_" # Ensure correct prefix
                        # )
                        # if error_eca:
                        #     st.error(f"Error analyzing event correlations for anomalies: {error_eca}")
                        # elif event_corr_anom_df is not None and not event_corr_anom_df.empty:
                        #     st.write("Mean Event Counts (Anomalous vs. Normal Devices vs. Overall):")
                        #     st.dataframe(event_corr_anom_df)
                        # else:
                        #     st.info("No event correlation data to display for anomalies or no event count features found matching the prefix.")
                    # else: st.info("No event count features found in the dataset for anomaly correlation.") # Optional message

            with tab_pop_clustering:
                st.subheader("Device Behavior Clustering")
                if "last_clustering_df_id" not in st.session_state or id(features_df_cleaned) != st.session_state.get("last_clustering_df_id"):
                    st.session_state.clustering_results = {}
                    st.session_state.kmeans_stats_df = None
                    st.session_state.last_clustering_df_id = id(features_df_cleaned)

                cluster_method = st.selectbox("Method", ["K-Means", "DBSCAN"], key="pop_clust_method_general")

                # Specific scaling checkboxes per method
                scale_data_kmeans_specific = False
                scale_data_dbscan_specific = False

                if cluster_method == "K-Means":
                    scale_data_kmeans_specific = st.checkbox("Scale data before K-Means", value=True, key="scale_data_clustering_kmeans_general")
                    st.write("Determine optimal K:")
                    k_min = st.number_input("Min K", 2, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2, 2, key="k_min_kstats_general")
                    k_max_val = min(10, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2)
                    k_max = st.number_input("Max K", k_min, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2, k_max_val if k_max_val >= k_min else k_min, key="k_max_kstats_general")
                    if st.button("Calc K-Means Stats", key="calc_km_stats_btn_general"):
                        if k_max >= k_min:
                            k_stats_df, err_msg = get_kmeans_elbow_silhouette_data(features_df_cleaned, k_range=range(k_min, k_max+1), scale_data=scale_data_kmeans_specific); # Use specific scale
                            if err_msg: st.error(err_msg)
                            else: st.session_state.kmeans_stats_df = k_stats_df
                        else: st.warning("Max K >= Min K.")
                    if st.session_state.get("kmeans_stats_df") is not None: k_stats_df_display = st.session_state.kmeans_stats_df; st.line_chart(k_stats_df_display.set_index('K')['Inertia']); st.line_chart(k_stats_df_display.set_index('K')['Silhouette Score'].dropna())
                    k_final = st.number_input("Num Clusters (K)", 2, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2, 3, key="km_k_final_general");
                # Removed individual save/load placeholder
                    if st.button("Run K-Means", key="run_km_pop_btn_general"):
                        labels, model, err_msg = perform_kmeans_clustering(features_df_cleaned, n_clusters=k_final, scale_data=scale_data_kmeans_specific) # Use specific scale
                        if err_msg:
                            st.error(err_msg)
                        else:
                            st.session_state.clustering_results = {'method':'K-Means', 'labels':labels, 'model':model, 'k':k_final}
                            st.success(f"K-Means complete (K={k_final}).")
                elif cluster_method == "DBSCAN":
                    scale_data_dbscan_specific = st.checkbox("Scale data before DBSCAN", value=True, key="scale_data_clustering_dbscan_general")
                    eps_dbscan = st.number_input("Epsilon", 0.01, 10.0, 0.5, 0.01, key="db_eps_general")
                    min_samples_dbscan = st.number_input("Min Samples", 1, 100, 5, 1, key="db_min_samples_general")
                    # Removed individual save/load placeholder
                    if st.button("Run DBSCAN", key="run_db_pop_btn_general"):
                        labels, model, err_msg = perform_dbscan_clustering(features_df_cleaned, eps=eps_dbscan, min_samples=min_samples_dbscan, scale_data=scale_data_dbscan_specific) # Use specific scale
                        if err_msg:
                            st.error(err_msg)
                        else:
                            st.session_state.clustering_results = {'method':'DBSCAN', 'labels':labels, 'model':model}
                            st.success("DBSCAN complete.")

                clust_results = st.session_state.get('clustering_results')
                if clust_results and 'labels' in clust_results:
                    st.write(f"Results: {clust_results['method']}"); clust_summary = clust_results['labels'].value_counts().rename("Device Count").to_frame(); st.dataframe(clust_summary)
                    if clust_results['method'] == 'K-Means' and clust_results.get('model'): st.write("Cluster Centers:"); centers_df = pd.DataFrame(clust_results['model'].cluster_centers_, columns=features_df_cleaned.columns); st.dataframe(centers_df)
                    st.subheader("Explain Clusters"); importance_df, err_imp = get_feature_importance_for_clusters_anova(features_df_cleaned, clust_results["labels"])
                    if err_imp: st.error(f"Importance error: {err_imp}")
                    elif importance_df is not None and not importance_df.empty: st.write("Top Differentiating Features (ANOVA):"); st.dataframe(importance_df.head(10))
                    else: st.info("Could not get feature importance.")
                    summary_means_df, err_means = get_cluster_feature_summary(features_df_cleaned, clust_results["labels"])
                    if err_means: st.error(f"Mean summary error: {err_means}")
                    elif summary_means_df is not None and not summary_means_df.empty:
                        st.write("Mean Feature Values per Cluster:"); st.dataframe(summary_means_df)
                        overall_mean_features_for_summary = features_df_cleaned.mean() # Calculate overall means
                        st.markdown("---"); st.markdown("**Cluster Summaries:**")
                        for cluster_id_val in sorted(clust_results["labels"].unique()):
                            cluster_name_disp = f"Cluster {cluster_id_val}" if not (cluster_id_val == -1 and clust_results.get("method") == "DBSCAN") else "Noise Points (DBSCAN Cluster -1)"
                            cluster_size_val = (clust_results["labels"] == cluster_id_val).sum()
                            cluster_means_for_this_one = summary_means_df.loc[cluster_id_val] if cluster_id_val in summary_means_df.index else pd.Series(dtype=float) # Handle if cluster_id not in summary (e.g. noise)

                            summary_text = generate_cluster_summary_text(
                                cluster_name_disp,
                                cluster_size_val,
                                len(features_df_cleaned),
                                cluster_mean_features_for_this_cluster=cluster_means_for_this_one,
                                overall_mean_features=overall_mean_features_for_summary
                            ); st.markdown(f"- {summary_text}")
                    else: st.info("No cluster mean summary.")

                    # Event Correlations for Clusters
                    event_cols_in_features_df_clust = [col for col in features_df_cleaned.columns if col.startswith(st.session_state.time_series_specs.get("selected_value_col_for_analysis", "value") + "_evt_count_")]
                    if event_cols_in_features_df_clust:
                        st.markdown("---")
                        st.subheader("Event Correlations with Clusters")
                        st.warning("Event correlation analysis for clusters is temporarily disabled (analyze_event_correlations function not found).")
                        # event_corr_clust_df, error_ecc = analyze_event_correlations(
                        #     features_df_cleaned,
                        #     clust_results["labels"], # current_cluster_labels
                        #     event_feature_prefix=f"{st.session_state.time_series_specs.get('selected_value_col_for_analysis', 'value')}_evt_count_"
                        # )
                        # if error_ecc:
                        #     st.error(f"Error analyzing event correlations for clusters: {error_ecc}")
                        # elif event_corr_clust_df is not None and not event_corr_clust_df.empty:
                        #     st.write("Mean Event Counts per Cluster (vs. Overall):")
                        #     st.dataframe(event_corr_clust_df)
                        # else:
                        #     st.info("No event correlation data to display for clusters or no event count features found matching the prefix.")
                    # else: st.info("No event count features found in the dataset for cluster correlation.") # Optional

                    if not features_df_cleaned.empty: # This is for the feature distribution plot by cluster
                        feat_to_plot = st.selectbox("Select feature to visualize by cluster", options=features_df_cleaned.columns.tolist(), key="clust_feat_plot_sel_general")
                        if feat_to_plot: plot_df = features_df_cleaned.copy(); plot_df['cluster'] = clust_results['labels']; fig, ax = plt.subplots(); plot_df.boxplot(column=feat_to_plot, by='cluster', ax=ax, grid=False); ax.set_title(f"Distribution of '{feat_to_plot}' by Cluster"); ax.set_xlabel("Cluster"); ax.set_ylabel(feat_to_plot); plt.suptitle(''); st.pyplot(fig)
    elif not st.session_state.get("running_all_features_computation") and st.session_state.get("compute_all_features_button_general"): st.warning("Feature computation for all devices resulted in an empty dataset.")
    elif not st.session_state.data_df_original.empty : st.info("Compute 'All Device Features' from sidebar for population analysis.")

    if not st.session_state.get("all_device_features_df", pd.DataFrame()).empty:
        st.header("🔬 Validate Findings with Known Failures")
        st.markdown("Provide known failed Device IDs (one per line or comma-separated) to see how they map to detected anomalies or clusters.")
        failed_ids_input = st.text_area("Enter known failed Device/Entity IDs", height=100, key="failed_ids_text_area_general")
        if st.button("Run Validation Analysis", key="run_validation_btn_general") and failed_ids_input.strip():
            parsed_ids = set(item.strip() for line in failed_ids_input.strip().split("\n") for item in line.split(",") if item.strip())
            if not parsed_ids: st.warning("No valid failed IDs entered.")
            else:
                st.write(f"Validating against {len(parsed_ids)} unique known failed IDs.")
                available_ids_in_features = set(st.session_state.all_device_features_df.index.tolist())
                valid_failed_ids_in_dataset = parsed_ids.intersection(available_ids_in_features)
                if not valid_failed_ids_in_dataset: st.error("None of the provided failed IDs are in the current dataset's entities.")
                else:
                    st.info(f"{len(valid_failed_ids_in_dataset)} of your failed IDs are in the dataset and used for validation.")
                    pop_anom_results = st.session_state.get("population_anomaly_results")
                    if pop_anom_results and "labels" in pop_anom_results:
                        st.subheader(f"Validation: Population Anomaly Detection ({pop_anom_results.get('method', 'N/A')})")
                        anomalous_detected_ids = set(pop_anom_results["labels"][pop_anom_results["labels"] == -1].index.tolist())
                        identified_failed_anomalies = valid_failed_ids_in_dataset.intersection(anomalous_detected_ids)
                        st.metric(label="Known Failed Devices Flagged as Anomalous", value=f"{len(identified_failed_anomalies)} / {len(valid_failed_ids_in_dataset)}")
                        if identified_failed_anomalies: st.write("IDs of known failures flagged as anomalous:", sorted(list(identified_failed_anomalies)))
                    else: st.info("Run Population Anomaly Detection first for its validation.")
                    clust_results_val = st.session_state.get("clustering_results")
                    if clust_results_val and "labels" in clust_results_val:
                        st.subheader(f"Validation: Device Behavior Clustering ({clust_results_val.get('method', 'N/A')})")
                        cluster_labels_val = clust_results_val["labels"]; validation_summary_list = []
                        for cluster_id_iter in sorted(cluster_labels_val.unique()):
                            cluster_device_ids = set(cluster_labels_val[cluster_labels_val == cluster_id_iter].index.tolist())
                            identified_failed_in_cluster = valid_failed_ids_in_dataset.intersection(cluster_device_ids)
                            percentage_of_cluster = (len(identified_failed_in_cluster)/len(cluster_device_ids)*100) if len(cluster_device_ids)>0 else 0
                            percentage_of_total_failed = (len(identified_failed_in_cluster)/len(valid_failed_ids_in_dataset)*100) if len(valid_failed_ids_in_dataset)>0 else 0
                            cluster_name_val = f"Cluster {cluster_id_iter}" if not (cluster_id_iter == -1 and clust_results_val.get("method") == "DBSCAN") else "Noise (-1)"
                            validation_summary_list.append({"Cluster ID": cluster_name_val, "Total Devices": len(cluster_device_ids), "Known Failed in Cluster": len(identified_failed_in_cluster), "% of Cluster (Failures)": f"{percentage_of_cluster:.2f}%", "% of Total Failures in Cluster": f"{percentage_of_total_failed:.2f}%"})
                        if validation_summary_list: st.dataframe(pd.DataFrame(validation_summary_list).set_index("Cluster ID"))
                        else: st.info("Could not generate cluster validation summary.")
                    else: st.info("Run Clustering first for its validation.")
    elif not st.session_state.data_df_original.empty : st.info("Compute 'All Device Features' from sidebar for validation.")

    st.header("General Data Table Tools")
    if not st.session_state.data_df.empty:
        if st.checkbox("Show Summary Statistics for Full Loaded Data Preview", key="general_stats_cb_general"):
            st.subheader("Summary Statistics (Loaded Data Preview)"); st.write(st.session_state.data_df.describe(include='all'))
    else: st.info("Load data using the sidebar to enable general data tools.")

# To run this app: streamlit run src/main.py