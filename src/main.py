import streamlit as st
import pandas as pd
import numpy as np
from database import (
    connect_postgres, fetch_postgres_data,
    connect_elasticsearch, fetch_elasticsearch_data
)
from src.analysis_modules.profiling import (
    get_series_summary_stats,
    get_missing_values_summary,
    perform_stationarity_test
)
from src.analysis_modules.decomposition import decompose_time_series
from src.analysis_modules.anomalies import (
    detect_anomalies_zscore,
    detect_anomalies_iqr,
    # detect_anomalies_isolation_forest, # Now part of AnomalyDetectionAnalysisModule
    # detect_anomalies_one_class_svm, # Now part of AnomalyDetectionAnalysisModule
    AnomalyDetectionAnalysisModule
)
from src.analysis_modules.clustering import (
    perform_kmeans_clustering,
    # perform_kmeans_clustering, # Now part of ClusteringAnalysisModule
    # get_kmeans_elbow_silhouette_data, # Now part of ClusteringAnalysisModule
    # perform_dbscan_clustering, # Now part of ClusteringAnalysisModule
    ClusteringAnalysisModule
)
from src.analysis_modules.feature_engineering import generate_all_features_for_series
from src.analysis_modules.explainability import (
    get_cluster_feature_summary,
    get_feature_importance_for_clusters_anova,
    compare_anomalous_vs_normal_features,
    generate_cluster_summary_text,
    generate_anomaly_summary_text,
    explain_anomalies_with_surrogate_model,
    analyze_event_correlations, # Added for context, ensure it's there
    analyze_significant_event_types # New import
)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import json # New import
from src.config_utils import ( # New imports
    gather_settings_for_save,
    apply_loaded_settings_to_session_state,
    TS_SETTINGS_KEYS, IFOREST_PARAMS_KEYS, OCSVM_PARAMS_KEYS,
    KMEANS_PARAMS_KEYS, DBSCAN_PARAMS_KEYS # Import key dicts if needed directly in main
)


# --- Page Configuration ---
st.set_page_config(page_title="Data Analyzer", layout="wide")
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
# if 'population_anomaly_results' not in st.session_state: st.session_state.population_anomaly_results = {} # Old state
# Old clustering state - will be replaced by module-specific state
# if 'clustering_results' not in st.session_state: st.session_state.clustering_results = {}
# if 'kmeans_stats_df' not in st.session_state: st.session_state.kmeans_stats_df = None
if 'last_pop_anomaly_df_id' not in st.session_state: st.session_state.last_pop_anomaly_df_id = None
# if 'last_clustering_df_id' not in st.session_state: st.session_state.last_clustering_df_id = None # Will be managed by module or module's results scope
if 'res_df_anomalies_sorted' not in st.session_state: st.session_state.res_df_anomalies_sorted = pd.DataFrame()
if 'surrogate_tree_explainer' not in st.session_state: st.session_state.surrogate_tree_explainer = None

# --- Utility for resetting states ---
def reset_ts_and_temp_cols():
    temp_id_col_name = "_temp_unique_id_"
    if temp_id_col_name in st.session_state.data_df_original.columns:
        try: del st.session_state.data_df_original[temp_id_col_name]
        except KeyError: pass
    if temp_id_col_name in st.session_state.data_df.columns:
         try: del st.session_state.data_df[temp_id_col_name]
         except KeyError: pass
    st.session_state.time_series_specs = {"id_cols": [], "timestamp_col": "None", "value_cols": [], "selected_id": "None", "selected_value_col_for_analysis": "None", "processed_series": None}
    st.session_state.active_filters = {}
    st.session_state.all_device_features_df = pd.DataFrame()
    st.session_state.single_series_features_display = None
    # st.session_state.population_anomaly_results = {} # Old state, replaced by module state
    # st.session_state.clustering_results = {} # Old state
    # st.session_state.kmeans_stats_df = None # Old state
    st.session_state.last_pop_anomaly_df_id = None
    # st.session_state.last_clustering_df_id = None # Old state

    # New module-specific state reset
    if 'clustering_module_params' in st.session_state:
        del st.session_state.clustering_module_params
    if 'clustering_module_results' in st.session_state:
        del st.session_state.clustering_module_results

    # New anomaly module state reset
    if 'anomaly_module_params' in st.session_state:
        del st.session_state.anomaly_module_params
    if 'anomaly_module_results' in st.session_state:
        del st.session_state.anomaly_module_results

    st.session_state.res_df_anomalies_sorted = pd.DataFrame() # This might need to be sourced from anomaly_module_results
    st.session_state.surrogate_tree_explainer = None
    st.session_state.event_df = pd.DataFrame() # Reset event_df
    st.session_state.global_top_event_types_cleaned = [] # Reset global event types


# --- APP MODE SELECTION (SIDEBAR) ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.radio(
    "Choose Mode:",
    ("General Analysis", "Guided Workflows"),
    key="app_mode_selector"
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
        # Updated to use new module state
        clustering_module_results_wf = st.session_state.get('clustering_module_results', {})
        if not clustering_module_results_wf or \
           clustering_module_results_wf.get("status") != "clustering_done" or \
           not isinstance(clustering_module_results_wf.get('labels'), pd.Series) or \
           clustering_module_results_wf['labels'].empty:
            st.warning("Run Device Behavior Clustering first in 'General Analysis' mode (and ensure it completed successfully)."); prereq_met = False
        if not prereq_met: st.stop()
        st.success("Prerequisites met.")
        st.subheader("Step 2: Select At-Risk Cluster(s)")
        cluster_labels_series_wf = st.session_state.clustering_module_results['labels'] # Use new state
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
            st.write(f"Found {len(devices_in_at_risk_clusters_wf)} devices in at-risk cluster(s):")
            known_failed_ids_input_wf = st.text_area("Enter known failed IDs (comma/newline separated) for cross-referencing:", height=100, key="failed_ids_input_workflow").strip()
            devices_to_monitor_df_wf = pd.DataFrame(devices_in_at_risk_clusters_wf, columns=["Device ID"])
            if known_failed_ids_input_wf:
                parsed_ids_wf = set(item.strip() for line in known_failed_ids_input_wf.split("\n") for item in line.split(",") if item.strip())
                devices_to_monitor_df_wf['Is_Known_Failure'] = devices_to_monitor_df_wf['Device ID'].isin(parsed_ids_wf)
                st.dataframe(devices_to_monitor_df_wf.sort_values(by=['Is_Known_Failure', 'Device ID']))
                not_yet_failed_in_at_risk_wf = devices_to_monitor_df_wf[~devices_to_monitor_df_wf['Is_Known_Failure']]
                st.write(f"**{len(not_yet_failed_in_at_risk_wf)} devices in at-risk clusters are NOT in your provided known failures list.**")
                if not not_yet_failed_in_at_risk_wf.empty: st.dataframe(not_yet_failed_in_at_risk_wf[['Device ID']].reset_index(drop=True))
            else: st.dataframe(devices_to_monitor_df_wf); st.info("Provide known failed IDs to highlight new concerns.")

elif app_mode == "General Analysis":
    # --- EXISTING UI LOGIC INDENTED HERE ---
    st.sidebar.header("Database Connection")
    db_type_general = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "Elasticsearch"], key="db_type_select_general")
    if db_type_general == "PostgreSQL":
        st.sidebar.subheader("PostgreSQL Details"); pg_host = st.sidebar.text_input("Host", "localhost", key="pg_host_general"); pg_port = st.sidebar.text_input("Port", "5432", key="pg_port_general"); pg_dbname = st.sidebar.text_input("Database Name", "mydatabase", key="pg_dbname_general"); pg_user = st.sidebar.text_input("User", "myuser", key="pg_user_general"); pg_password = st.sidebar.text_input("Password", type="password", key="pg_password_general"); pg_query = st.sidebar.text_area("SQL Query", "SELECT * FROM your_table_name LIMIT 100;", key="pg_query_general")
        if st.sidebar.button("Connect to PostgreSQL", key="pg_connect_btn_general"):
            st.session_state.db_conn = connect_postgres(pg_host, pg_port, pg_dbname, pg_user, pg_password);
            if st.session_state.db_conn: st.sidebar.success("Connected!"); st.session_state.data_df_original = pd.DataFrame(); st.session_state.data_df = pd.DataFrame(); reset_ts_and_temp_cols()
        if st.sidebar.button("Fetch Data from PostgreSQL", key="pg_fetch_btn_general"):
            if st.session_state.db_conn: reset_ts_and_temp_cols(); st.session_state.data_df_original = fetch_postgres_data(st.session_state.db_conn, pg_query); st.session_state.data_df = st.session_state.data_df_original.copy();
            if not st.session_state.data_df_original.empty: st.sidebar.success("Data fetched!")
            else: st.sidebar.warning("No data returned.")
            else: st.sidebar.error("Not connected.")
    elif db_type_general == "Elasticsearch":
        st.sidebar.subheader("Elasticsearch Details"); es_hosts_str = st.sidebar.text_input("Host URL(s) (comma-separated)", "http://localhost:9200", key="es_hosts_general"); es_index = st.sidebar.text_input("Index Name", "my_index", key="es_index_general"); es_query_dsl_str = st.sidebar.text_area("Elasticsearch Query DSL (JSON)", '{\n  "query": {\n    "match_all": {}\n  }\n}', key="es_query_general")
        if st.sidebar.button("Connect to Elasticsearch", key="es_connect_btn_general"):
            es_hosts_list = [h.strip() for h in es_hosts_str.split(',') if h.strip()]
            if not es_hosts_list: st.sidebar.error("Enter Host URL(s)."); st.session_state.db_conn = None
            else: st.session_state.db_conn = connect_elasticsearch(hosts=es_hosts_list)
            if st.session_state.db_conn: st.session_state.data_df_original = pd.DataFrame(); st.session_state.data_df = pd.DataFrame(); reset_ts_and_temp_cols()
        if st.sidebar.button("Fetch Data from Elasticsearch", key="es_fetch_btn_general"):
            if st.session_state.db_conn:
                import json
                try: query_body = json.loads(es_query_dsl_str); reset_ts_and_temp_cols(); st.session_state.data_df_original = fetch_elasticsearch_data(st.session_state.db_conn, es_index, query_body); st.session_state.data_df = st.session_state.data_df_original.copy()
                if not st.session_state.data_df_original.empty: st.sidebar.success("Data fetched!")
                else: st.sidebar.warning("No data returned.")
                except json.JSONDecodeError: st.sidebar.error("Invalid JSON in Query DSL.")
                except Exception as e: st.sidebar.error(f"Fetch error: {e}")
            else: st.sidebar.error("Not connected.")

    st.sidebar.markdown("---") # Event Data Uploader
    st.sidebar.subheader("Optional: Load Event Data")
    event_file = st.sidebar.file_uploader("Upload Event Data (CSV or Excel)", type=['csv', 'xlsx'], key="event_file_uploader_general")
    if event_file:
        try:
            if event_file.name.endswith('.csv'): df_events = pd.read_csv(event_file)
            else: df_events = pd.read_excel(event_file)
            required_event_cols = ['device_id', 'timestamp', 'event_type']
            if not all(col in df_events.columns for col in required_event_cols):
                st.sidebar.error(f"Event data must contain: {', '.join(required_event_cols)}")
                st.session_state.event_df = pd.DataFrame()
            else:
                df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], errors='coerce')
                df_events = df_events.dropna(subset=['timestamp'])
                st.session_state.event_df = df_events
                st.sidebar.success(f"Loaded event data: {len(df_events)} records.")
        except Exception as e: st.sidebar.error(f"Error loading event data: {e}"); st.session_state.event_df = pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.header("Time Series Settings"); temp_id_col_name = "_temp_unique_id_"
    if not st.session_state.data_df_original.empty:
        df_columns = st.session_state.data_df_original.columns.tolist()
        current_id_cols = st.session_state.time_series_specs.get("id_cols", []); st.session_state.time_series_specs["id_cols"] = st.sidebar.multiselect("ID Column(s) (Optional)", options=df_columns, default=[col for col in current_id_cols if col in df_columns], key="ts_id_cols_general")
        current_ts_col = st.session_state.time_series_specs.get("timestamp_col", "None"); ts_col_options = ["None"] + df_columns; st.session_state.time_series_specs["timestamp_col"] = st.sidebar.selectbox("Timestamp Column", options=ts_col_options, index=ts_col_options.index(current_ts_col) if current_ts_col in ts_col_options else 0, key="ts_timestamp_col_general")
        current_value_cols = st.session_state.time_series_specs.get("value_cols", []); st.session_state.time_series_specs["value_cols"] = st.sidebar.multiselect("Value/Metric Column(s)", options=df_columns, default=[col for col in current_value_cols if col in df_columns], key="ts_value_cols_general")
        if st.session_state.time_series_specs["timestamp_col"] != "None" and st.session_state.time_series_specs["value_cols"]:
            unique_ids_display = ["Default Time Series"]
            if st.session_state.time_series_specs["id_cols"]:
                try:
                    df_for_ids = st.session_state.data_df_original; id_col_data_check = df_for_ids[st.session_state.time_series_specs["id_cols"]].astype(str).agg('-'.join, axis=1)
                    if temp_id_col_name not in df_for_ids.columns or not df_for_ids[temp_id_col_name].equals(id_col_data_check): st.session_state.data_df_original[temp_id_col_name] = id_col_data_check
                    unique_ids_list = sorted(st.session_state.data_df_original[temp_id_col_name].unique().tolist()); unique_ids_display = ["None"] + unique_ids_list
                except KeyError as e: st.sidebar.warning(f"ID column error: {e}. Reselect."); st.session_state.time_series_specs["id_cols"] = []
            current_selected_id = st.session_state.time_series_specs.get("selected_id", "None"); st.session_state.time_series_specs["selected_id"] = st.sidebar.selectbox("Select Device/Entity ID", options=unique_ids_display, index=unique_ids_display.index(current_selected_id) if current_selected_id in unique_ids_display else 0, key="ts_selected_id_general")
            valid_value_cols_for_selection = [col for col in st.session_state.time_series_specs["value_cols"] if col in df_columns]; value_col_options = ["None"] + valid_value_cols_for_selection; current_selected_value_col = st.session_state.time_series_specs.get("selected_value_col_for_analysis", "None"); st.session_state.time_series_specs["selected_value_col_for_analysis"] = st.sidebar.selectbox("Select Value/Metric for Analysis", options=value_col_options, index=value_col_options.index(current_selected_value_col) if current_selected_value_col in value_col_options else 0, key="ts_selected_value_col_general")
            if st.sidebar.button("Prepare Time Series for Analysis", key="ts_prepare_btn_general"):
                ts_col, val_col, selected_entity_id = st.session_state.time_series_specs["timestamp_col"], st.session_state.time_series_specs["selected_value_col_for_analysis"], st.session_state.time_series_specs["selected_id"]
                if selected_entity_id!="None" and ts_col!="None" and val_col!="None":
                    current_df_copy = st.session_state.data_df_original.copy(); entity_series_df = current_df_copy
                    if selected_entity_id!="Default Time Series":
                        if temp_id_col_name in current_df_copy.columns: entity_series_df = current_df_copy[current_df_copy[temp_id_col_name] == selected_entity_id]
                        else: st.sidebar.error("ID column error."); st.session_state.time_series_specs["processed_series"]=None; entity_series_df=pd.DataFrame()
                    if not entity_series_df.empty:
                        try: entity_series_df[ts_col]=pd.to_datetime(entity_series_df[ts_col],errors='coerce'); entity_series_df=entity_series_df.dropna(subset=[ts_col,val_col])
                        if entity_series_df.empty: st.sidebar.error(f"No valid data after NA drop."); st.session_state.time_series_specs["processed_series"]=None
                        else: entity_series_df=entity_series_df.sort_values(by=ts_col); processed_series=entity_series_df.groupby(ts_col)[val_col].mean().rename(val_col); st.session_state.time_series_specs["processed_series"]=processed_series; st.sidebar.success(f"Prepared '{val_col}' for '{selected_entity_id}'."); st.session_state.single_series_features_display=None
                        except Exception as e: st.sidebar.error(f"Error: {e}"); st.session_state.time_series_specs["processed_series"]=None
                    elif st.session_state.time_series_specs["processed_series"] is not None : st.sidebar.error(f"No data for ID: {selected_entity_id}."); st.session_state.time_series_specs["processed_series"]=None
                else: st.sidebar.warning("Select Entity, Timestamp, and Value."); st.session_state.time_series_specs["processed_series"]=None
        else:
            st.sidebar.info("Select Timestamp & Value columns for TS analysis.")
            if st.session_state.time_series_specs["processed_series"] is not None: st.session_state.time_series_specs["processed_series"]=None
            if temp_id_col_name in st.session_state.data_df_original.columns and (st.session_state.time_series_specs.get("timestamp_col","None")=="None" or not st.session_state.time_series_specs.get("value_cols",[])):
                try:del st.session_state.data_df_original[temp_id_col_name];
                except KeyError:pass
                if temp_id_col_name in st.session_state.data_df.columns:
                    try:del st.session_state.data_df[temp_id_col_name]
                    except KeyError:pass
    st.sidebar.markdown("---")
    st.sidebar.subheader("Save/Load App State")
    if st.sidebar.button("Save App State", key="save_app_state_button_general"):
        app_settings = gather_settings_for_save()
        json_settings = json.dumps(app_settings, indent=4)
        # Use a unique key for download button if it's dynamically created or might re-appear
        st.sidebar.download_button(
            label="Download Settings File (.json)",
            data=json_settings,
            file_name="data_analyzer_settings.json",
            mime="application/json",
            key="download_app_settings_json_button_general"
        )

    uploaded_settings_file = st.sidebar.file_uploader(
        "Load App State (.json)",
        type=['json'],
        key="load_app_settings_uploader_general"
    )
    if uploaded_settings_file is not None:
        try:
            loaded_settings_dict = json.load(uploaded_settings_file)
            success, message = apply_loaded_settings_to_session_state(loaded_settings_dict)
            if success:
                st.sidebar.success(f"Settings loaded! {message}")
                # Clear the uploader to allow re-upload of same file if needed
                st.session_state.load_app_settings_uploader_general = None
                st.experimental_rerun()
            else:
                st.sidebar.error(message)
        except Exception as e:
            st.sidebar.error(f"Error parsing settings file: {e}")

    # This needs to be structured correctly: the following elif and else are for the "if not st.session_state.data_df_original.empty:" block
    elif not st.session_state.data_df_original.empty and temp_id_col_name in st.session_state.data_df_original.columns:
         if (st.session_state.time_series_specs.get("timestamp_col","None")=="None" or not st.session_state.time_series_specs.get("value_cols",[])):
                try: del st.session_state.data_df_original[temp_id_col_name]
                except KeyError:pass
                if temp_id_col_name in st.session_state.data_df.columns:
                    try: del st.session_state.data_df[temp_id_col_name]
                    except KeyError:pass
    else: st.sidebar.info("Load data for TS settings.")
    st.sidebar.markdown("---"); st.sidebar.header("Population Analysis Settings")
    compute_all_button_disabled_general = not (st.session_state.time_series_specs.get("timestamp_col")!="None" and st.session_state.time_series_specs.get("selected_value_col_for_analysis")!="None" and not st.session_state.data_df_original.empty)
    if compute_all_button_disabled_general: st.sidebar.warning("Load data & select Timestamp & Value in 'Time Series Settings' for population features.")
    if st.sidebar.button("Compute Features for ALL Devices", key="compute_all_features_button_general", disabled=compute_all_button_disabled_general):
        st.session_state.running_all_features_computation=True; st.session_state.all_device_features_df=pd.DataFrame();
        # Reset module states
        if 'population_anomaly_results' in st.session_state: del st.session_state.population_anomaly_results # old key
        if 'anomaly_module_params' in st.session_state: del st.session_state.anomaly_module_params
        if 'anomaly_module_results' in st.session_state: del st.session_state.anomaly_module_results

        if 'clustering_module_params' in st.session_state: del st.session_state.clustering_module_params
        if 'clustering_module_results' in st.session_state: del st.session_state.clustering_module_results

        st.session_state.res_df_anomalies_sorted=pd.DataFrame(); st.session_state.surrogate_tree_explainer = None
        # Determine global top event types if event_df is loaded
        if not st.session_state.event_df.empty:
            try:
                all_event_counts_global = st.session_state.event_df['event_type'].value_counts()
                num_global_event_types_main = 10 # Or make this configurable
                st.session_state.global_top_event_types_cleaned = [
                    str(etype).replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:30]
                    for etype in all_event_counts_global.head(num_global_event_types_main).index.tolist()
                ]
                st.sidebar.info(f"Using top {len(st.session_state.global_top_event_types_cleaned)} global event types for feature consistency.")
            except Exception as e_global_events:
                st.sidebar.warning(f"Could not determine global event types: {e_global_events}")
                st.session_state.global_top_event_types_cleaned = []
        else:
            st.session_state.global_top_event_types_cleaned = []


    st.header("Loaded Data Preview")
    if not st.session_state.data_df.empty: st.dataframe(st.session_state.data_df.head()); st.info(f"Preview: {st.session_state.data_df.shape[0]} rows, {st.session_state.data_df.shape[1]} columns.")
    else: st.info("No data loaded.")

    processed_series_display = st.session_state.time_series_specs.get("processed_series")
    if processed_series_display is not None and not processed_series_display.empty:
        st.header("Prepared Time Series for Analysis")
        # --- Plotting with Events ---
        fig_ts, ax_ts = plt.subplots(figsize=(12,6))
        analysis_series_name_for_plot = processed_series_display.name or "Value"
        ax_ts.plot(processed_series_display.index, processed_series_display.values, label=analysis_series_name_for_plot)
        ax_ts.set_title(f"Time Series: {analysis_series_name_for_plot}")
        ax_ts.set_xlabel("Timestamp"); ax_ts.set_ylabel("Value")

        if not st.session_state.event_df.empty and st.session_state.time_series_specs.get("selected_id") is not None:
            event_df_loaded = st.session_state.event_df
            current_device_id_for_plot = st.session_state.time_series_specs["selected_id"]
            device_id_col_in_event_df = "device_id" # Default
            if st.session_state.time_series_specs.get("id_cols") and st.session_state.time_series_specs["id_cols"][0] in event_df_loaded.columns:
                device_id_col_in_event_df = st.session_state.time_series_specs["id_cols"][0]

            device_events = pd.DataFrame()
            if current_device_id_for_plot != "Default Time Series" and device_id_col_in_event_df in event_df_loaded.columns:
                device_events = event_df_loaded[event_df_loaded[device_id_col_in_event_df].astype(str) == str(current_device_id_for_plot)]
            elif current_device_id_for_plot == "Default Time Series" and not st.session_state.time_series_specs.get("id_cols"):
                device_events = event_df_loaded # Show all events if no specific device ID context from main data

            if not device_events.empty:
                plotted_event_types = []
                for idx, event_row in device_events.iterrows():
                    event_time = event_row['timestamp']; event_type = event_row['event_type']
                    color_map = {'error': 'red', 'warning': 'orange', 'info': 'blue', 'maintenance': 'green'}
                    event_color = color_map.get(str(event_type).lower(), 'gray')
                    label = f"Event: {event_type}" if event_type not in plotted_event_types else None
                    ax_ts.axvline(x=event_time, color=event_color, linestyle='--', label=label, lw=1.5 if label else 1)
                    if label: plotted_event_types.append(event_type)
                if plotted_event_types: ax_ts.legend()
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig_ts)
        st.write(processed_series_display.describe())

        st.header("Single Time Series Analysis"); analysis_series = processed_series_display; series_name = analysis_series.name if analysis_series.name else "Value"
        tab_profiling, tab_decomposition, tab_anomalies, tab_eng_features = st.tabs(["📊 Profiling", "📉 Decomposition", "❗ Anomaly Detection", "⚙️ Engineered Features"])
        with tab_profiling:
            st.subheader(f"Profiling: {series_name}"); st.write("Summary Statistics (Processed Series):"); summary_df = get_series_summary_stats(analysis_series); st.dataframe(summary_df)
            original_selected_series_for_profiling = pd.Series(dtype=float)
            current_df_orig_profiling = st.session_state.data_df_original.copy()
            selected_entity_id_profiling = st.session_state.time_series_specs["selected_id"]; ts_col_profiling = st.session_state.time_series_specs["timestamp_col"]; val_col_profiling = st.session_state.time_series_specs["selected_value_col_for_analysis"]
            if selected_entity_id_profiling != "None" and ts_col_profiling != "None" and val_col_profiling != "None":
                entity_df_orig = current_df_orig_profiling
                if selected_entity_id_profiling != "Default Time Series":
                    if temp_id_col_name not in entity_df_orig.columns and st.session_state.time_series_specs["id_cols"]: entity_df_orig[temp_id_col_name] = entity_df_orig[st.session_state.time_series_specs["id_cols"]].astype(str).agg('-'.join, axis=1)
                    if temp_id_col_name in entity_df_orig.columns: entity_df_orig = entity_df_orig[entity_df_orig[temp_id_col_name] == selected_entity_id_profiling]
                if not entity_df_orig.empty and val_col_profiling in entity_df_orig.columns and ts_col_profiling in entity_df_orig.columns:
                    entity_df_orig[ts_col_profiling] = pd.to_datetime(entity_df_orig[ts_col_profiling], errors='coerce'); temp_series_for_missing = entity_df_orig.dropna(subset=[ts_col_profiling]).set_index(ts_col_profiling)[val_col_profiling].sort_index()
                    if not temp_series_for_missing.empty: original_selected_series_for_profiling = temp_series_for_missing.groupby(temp_series_for_missing.index).mean()
            if not original_selected_series_for_profiling.empty: st.write("Missing Values (Original Segment for Selected Metric):"); missing_df_orig = get_missing_values_summary(original_selected_series_for_profiling); st.dataframe(missing_df_orig)
            else: st.warning("Original segment for full missing value profiling could not be retrieved.")
            st.write("Stationarity Test (ADF - on processed series):"); adf_results = perform_stationarity_test(analysis_series);
            if "error" in adf_results: st.error(adf_results["error"]); else: st.json(adf_results)
        with tab_decomposition:
            st.subheader(f"Decomposition: {series_name}"); decomp_model = st.selectbox("Model", ["additive", "multiplicative"], key="dc_model_general"); inferred_period = 1
            if analysis_series.index.inferred_freq: freq = analysis_series.index.inferred_freq
            else: freq = pd.infer_freq(analysis_series.index)
            if freq:
                if 'D' in freq: inferred_period = 7; elif 'M' in freq or 'ME' in freq : inferred_period = 12; elif 'Q' in freq or 'QE' in freq: inferred_period = 4; elif 'H' in freq: inferred_period = 24
            period_help_text = (f"Seasonality period. Suggested: {inferred_period}. Ensure series has min 2 full periods."); decomp_period = st.number_input("Period",2, None, inferred_period, key="dc_period_general");
            if st.button("Decompose", key="dc_btn_general"):
                if decomp_period and decomp_period >=2:
                    decomposition_result, error_msg = decompose_time_series(analysis_series, model=decomp_model, period=decomp_period)
                    if error_msg: st.error(error_msg)
                    if decomposition_result: st.success("Decomposition successful."); st.line_chart(decomposition_result.observed.rename("Observed")); st.line_chart(decomposition_result.trend.rename("Trend")); st.line_chart(decomposition_result.seasonal.rename("Seasonal")); st.line_chart(decomposition_result.resid.rename("Residual"))
                else: st.warning("Please provide a valid seasonal period (>=2).")
        with tab_anomalies:
            st.subheader(f"Anomalies: {series_name}"); anomaly_method_single = st.selectbox("Method", ["Z-score", "IQR"], key="an_single_method_general");
            if anomaly_method_single == "Z-score":
                z_threshold = st.number_input("Z-score Threshold", 0.5, 5.0, 3.0, 0.1, key="z_thresh_general"); z_window = st.selectbox("Rolling Window", [None,5,10,15,20,30],0,key="z_window_general")
                if st.button("Detect Z-score Anomalies",key="an_single_z_btn_general"):
                    anomalies, z_scores, error_msg = detect_anomalies_zscore(analysis_series, threshold=z_threshold, window=z_window)
                    if error_msg: st.error(error_msg)
                    elif anomalies is not None: st.success(f"Found {anomalies.sum()} Z-score anomalies."); fig, ax = plt.subplots(); ax.plot(analysis_series.index, analysis_series, label=series_name); ax.scatter(analysis_series.index[anomalies], analysis_series[anomalies], color='red', label='Anomalies', s=50, zorder=5); ax.set_title(f"{series_name} Z-score Anomalies"); ax.legend(); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)
            elif anomaly_method_single == "IQR":
                iqr_multiplier = st.number_input("IQR Multiplier", 0.5, 5.0, 1.5, 0.1, key="iqr_mult_general")
                if st.button("Detect IQR Anomalies",key="an_single_iqr_btn_general"):
                    anomalies, bounds_info, error_msg = detect_anomalies_iqr(analysis_series, multiplier=iqr_multiplier)
                    if error_msg: st.error(error_msg)
                    elif anomalies is not None: st.success(f"Found {anomalies.sum()} IQR anomalies."); fig, ax = plt.subplots(); ax.plot(analysis_series.index, analysis_series, label=series_name); ax.scatter(analysis_series.index[anomalies], analysis_series[anomalies], color='red', label='Anomalies', s=50, zorder=5);
                    if bounds_info is not None: lower_b = bounds_info[bounds_info['Metric']=='Lower Bound']['Value'].iloc[0]; upper_b = bounds_info[bounds_info['Metric']=='Upper Bound']['Value'].iloc[0]; ax.axhline(lower_b,color='orange',ls='--',label=f'Lower ({lower_b:.2f})'); ax.axhline(upper_b,color='orange',ls='--',label=f'Upper ({upper_b:.2f})')
                    ax.set_title(f"{series_name} IQR Anomalies"); ax.legend(); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)
        with tab_eng_features:
            st.subheader(f"Engineered Features for: {series_name}")
            if st.button("Compute/Refresh Features for Selected Series", key="compute_single_eng_features_btn_general"):
                if analysis_series is not None and not analysis_series.empty:
                    with st.spinner("Calculating..."):
                        single_device_events_tab = pd.DataFrame()
                        if not st.session_state.event_df.empty and st.session_state.time_series_specs.get("selected_id"):
                            event_df_loaded_single_tab = st.session_state.event_df
                            selected_id_single_tab = st.session_state.time_series_specs["selected_id"]
                            id_cols_single_tab = st.session_state.time_series_specs.get("id_cols", [])
                            # Determine the device ID column in event_df
                            dev_id_col_event_single_tab = id_cols_single_tab[0] if id_cols_single_tab and id_cols_single_tab[0] in event_df_loaded_single_tab.columns else "device_id"

                            if selected_id_single_tab != "DefaultTimeSeries" and dev_id_col_event_single_tab in event_df_loaded_single_tab.columns:
                                single_device_events_tab = event_df_loaded_single_tab[event_df_loaded_single_tab[dev_id_col_event_single_tab].astype(str) == str(selected_id_single_tab)]
                            elif selected_id_single_tab == "DefaultTimeSeries" and not id_cols_single_tab: # Only use all events if no specific entity context for main series
                                 single_device_events_tab = event_df_loaded_single_tab

                        feature_name_prefix = f"{analysis_series.name}_" if analysis_series.name else "series_"
                        st.session_state.single_series_features_display = generate_all_features_for_series(
                            analysis_series,
                            name=feature_name_prefix,
                            device_event_df=single_device_events_tab if not single_device_events_tab.empty else None,
                            all_possible_event_types=st.session_state.get("global_top_event_types_cleaned", [])
                        )
                else: st.session_state.single_series_features_display = None; st.warning("No processed series.")
            current_single_features = st.session_state.get("single_series_features_display")
            if current_single_features is not None and not current_single_features.empty: st.dataframe(current_single_features.rename("Value"))
            elif st.session_state.get("compute_single_eng_features_btn_general"): st.info("No features to display.")
    elif not st.session_state.data_df_original.empty: st.info("Prepare a time series via sidebar for single series analysis.")

    if st.session_state.get("running_all_features_computation"):
        st.header("Device Behavior Feature Engineering (All Devices)");
        with st.spinner("Processing all devices..."):
            original_df_copy = st.session_state.data_df_original.copy(); all_features_list = []
            id_cols = st.session_state.time_series_specs.get("id_cols", []); ts_col_all = st.session_state.time_series_specs["timestamp_col"]; val_col_to_process_for_all = st.session_state.time_series_specs["selected_value_col_for_analysis"]
            unique_entities_list = ["DefaultTimeSeries"]
            if id_cols:
                if temp_id_col_name not in original_df_copy.columns: original_df_copy[temp_id_col_name] = original_df_copy[id_cols].astype(str).agg('-'.join, axis=1)
                unique_entities_list = original_df_copy[temp_id_col_name].unique()
            total_entities = len(unique_entities_list); progress_bar = st.progress(0)
            for i, entity_id_iter in enumerate(unique_entities_list):
                entity_df_segment = original_df_copy
                if id_cols: entity_df_segment = original_df_copy[original_df_copy[temp_id_col_name] == entity_id_iter]
                if entity_df_segment.empty: continue
                try:
                    series_for_entity_all = entity_df_segment.copy(); series_for_entity_all[ts_col_all] = pd.to_datetime(series_for_entity_all[ts_col_all], errors='coerce'); series_for_entity_all = series_for_entity_all.dropna(subset=[ts_col_all, val_col_to_process_for_all]).sort_values(by=ts_col_all)
                    processed_entity_series_all = series_for_entity_all.groupby(ts_col_all)[val_col_to_process_for_all].mean(); processed_entity_series_all.name = val_col_to_process_for_all
                if not processed_entity_series_all.empty:
                    device_specific_events_all = pd.DataFrame()
                    if not st.session_state.event_df.empty:
                        event_df_loaded_main_loop = st.session_state.event_df
                        dev_id_col_event_main_loop = id_cols[0] if id_cols and id_cols[0] in event_df_loaded_main_loop.columns else "device_id"
                        if entity_id_iter != "DefaultTimeSeries" and dev_id_col_event_main_loop in event_df_loaded_main_loop.columns:
                            device_specific_events_all = event_df_loaded_main_loop[event_df_loaded_main_loop[dev_id_col_event_main_loop].astype(str) == str(entity_id_iter)]
                        elif entity_id_iter == "DefaultTimeSeries" and not id_cols:
                            device_specific_events_all = event_df_loaded_main_loop

                    features = generate_all_features_for_series(
                        processed_entity_series_all,
                        name=f"{val_col_to_process_for_all}_",
                        device_event_df=device_specific_events_all if not device_specific_events_all.empty else None,
                        all_possible_event_types=st.session_state.get("global_top_event_types_cleaned", [])
                    )
                    features['entity_id'] = entity_id_iter; all_features_list.append(features)
                except Exception as e: st.warning(f"Could not process entity {entity_id_iter} for {val_col_to_process_for_all}: {e}")
                progress_bar.progress((i + 1) / total_entities)
            if all_features_list: st.session_state.all_device_features_df = pd.DataFrame(all_features_list).set_index('entity_id'); st.success(f"Computed features for {len(all_features_list)} devices.")
            else: st.session_state.all_device_features_df = pd.DataFrame(); st.warning("No features computed.")
            st.session_state.running_all_features_computation = False; st.experimental_rerun()

    if not st.session_state.get('all_device_features_df', pd.DataFrame()).empty and not st.session_state.get("running_all_features_computation"):
        st.header("Population-Level Analysis"); st.markdown("On 'All Device Features' table.")
        features_df_cleaned = st.session_state.all_device_features_df.dropna()
        if features_df_cleaned.empty or len(features_df_cleaned) < 2 : st.warning("Not enough clean data in 'All Device Features' for population analysis.")
        else:
            tab_pop_anomalies, tab_pop_clustering = st.tabs(["🕵️ Anomaly Detection (All Devices)", "🧩 Device Behavior Clustering"])
            with tab_pop_anomalies:
                anomaly_module = AnomalyDetectionAnalysisModule()
                anom_module_key = "anomaly_detection_module"

                if f'{anom_module_key}_params' not in st.session_state:
                    default_params_anom = {}
                    for k, v_dict in anomaly_module.get_parameter_definitions().items():
                        default_params_anom[k] = v_dict.get("default")
                    st.session_state[f'{anom_module_key}_params'] = default_params_anom

                if f'{anom_module_key}_results' not in st.session_state:
                    st.session_state[f'{anom_module_key}_results'] = {"status": "pending"}

                current_features_df_id = id(features_df_cleaned) if features_df_cleaned is not None else None
                if "last_pop_anomaly_df_id" not in st.session_state or \
                   current_features_df_id != st.session_state.get("last_pop_anomaly_df_id"):
                    st.session_state[f'{anom_module_key}_results'] = {"status": "pending"}
                    st.session_state.last_pop_anomaly_df_id = current_features_df_id
                    st.session_state.surrogate_tree_explainer = None # Reset explainer if data changes
                    st.session_state.res_df_anomalies_sorted = pd.DataFrame()


                with st.expander("Anomaly Detection Parameters", expanded=True):
                    updated_anom_params = anomaly_module.render_parameters_ui(
                        st,
                        st.session_state[f'{anom_module_key}_params'],
                        anom_module_key
                    )
                    st.session_state[f'{anom_module_key}_params'] = updated_anom_params

                    if st.button("Run Population Anomaly Detection", key=f"{anom_module_key}_run_button"):
                        if not features_df_cleaned.empty:
                            with st.spinner("Running anomaly detection..."):
                                results_anom, error_msg_anom = anomaly_module.run_analysis(
                                    features_df_cleaned,
                                    st.session_state[f'{anom_module_key}_params'],
                                    st.session_state
                                )
                                if error_msg_anom:
                                    st.error(error_msg_anom)
                                    st.session_state[f'{anom_module_key}_results'] = {"status": "error", "error_message": error_msg_anom}
                                else:
                                    st.session_state[f'{anom_module_key}_results'] = results_anom
                                    # Populate res_df_anomalies_sorted for downstream use (e.g., summary text)
                                    if results_anom and results_anom.get("status") == "detection_done":
                                        st.session_state.res_df_anomalies_sorted = pd.DataFrame({
                                            'label': results_anom['labels'],
                                            'score': results_anom['scores']
                                        }).sort_values(by='score')
                                    st.success("Population anomaly detection complete.")
                        else:
                            st.warning("No feature data available to run anomaly detection.")
                        st.experimental_rerun() # To reflect new results or errors

                # Display results from the module
                anom_results_to_render = st.session_state.get(f'{anom_module_key}_results')
                if anom_results_to_render and anom_results_to_render.get("status") != "pending":
                    if anom_results_to_render.get("status") == "error":
                        st.error(f"Anomaly Detection Error: {anom_results_to_render.get('error_message', 'Unknown error')}")
                    else:
                        anomaly_module.render_results(st, anom_results_to_render, st.session_state)
                else:
                    st.info("Population anomaly detection not yet run or no results to display.")

                # --- Existing Explainability Logic (adapted for new state) ---
                if anom_results_to_render and anom_results_to_render.get("status") == "detection_done":
                    pop_anom_labels = anom_results_to_render.get('labels')
                    pop_anom_scores = anom_results_to_render.get('scores') # though not directly used by comparison_df

                    if pop_anom_labels is not None:
                        st.subheader("Explain Anomalies (Post-Module Analysis)");
                        comparison_df, error_comp = compare_anomalous_vs_normal_features(
                            features_df_cleaned, pop_anom_labels, anomalous_label_val=-1
                        )
                        if error_comp: st.error(f"Comparison error: {error_comp}")
                        elif comparison_df is not None and not comparison_df.empty:
                            st.write("Mean Feature Comparison (Anomalous vs. Normal):"); st.dataframe(comparison_df.head(10))

                        # Updated call to generate_anomaly_summary_text
                        if not st.session_state.res_df_anomalies_sorted.empty: # This is already populated above
                            top_anomaly_id = st.session_state.res_df_anomalies_sorted.index[0]
                            top_anomaly_score = st.session_state.res_df_anomalies_sorted.iloc[0]["score"]
                            surrogate_importances_for_summary = st.session_state.get("surrogate_tree_explainer", {}).get("importances")
                            st.markdown("---"); st.markdown(f"**Summary for Top Anomaly ({top_anomaly_id}):**")
                            summary_text = generate_anomaly_summary_text(
                                top_anomaly_id, top_anomaly_score, comparison_df,
                                surrogate_tree_importances=surrogate_importances_for_summary
                            ); st.markdown(summary_text)
                        else: st.info("No sorted anomaly data for summary text.")

                        st.markdown("---"); st.subheader("Detailed Anomaly Explanation (Surrogate Decision Tree)")
                        if pop_anom_labels.nunique() < 2:
                            st.info("Not enough distinct classes (anomalous/normal) for surrogate tree after current run.")
                            st.session_state.surrogate_tree_explainer = None # Reset if not enough classes
                        else:
                            surrogate_max_depth = st.slider("Surrogate Tree Max Depth", 2, 10, 4, 1, key=f"{anom_module_key}_surrogate_depth")
                            if st.button("Train & Show Surrogate Explanation Tree", key=f"{anom_module_key}_train_surrogate_button"):
                                with st.spinner("Training surrogate tree..."):
                                    tree_model, tree_importances, tree_report, error_tree = explain_anomalies_with_surrogate_model(
                                        features_df_cleaned, pop_anom_labels, max_depth=surrogate_max_depth, test_size=0.0 # No test split for explanation
                                    )
                                    if error_tree: st.error(f"Surrogate tree error: {error_tree}"); st.session_state.surrogate_tree_explainer = None
                                    elif tree_model and tree_importances is not None:
                                        st.session_state.surrogate_tree_explainer = {"model": tree_model, "importances": tree_importances, "report": tree_report }
                                        st.success("Surrogate tree trained.")
                                    else: st.warning("Surrogate training produced no model/importances."); st.session_state.surrogate_tree_explainer = None
                                st.experimental_rerun()

                        if st.session_state.get("surrogate_tree_explainer"):
                            explainer_results = st.session_state.surrogate_tree_explainer
                            if explainer_results.get("model"):
                                st.write("**Surrogate Tree - Top Feature Importances:**"); st.dataframe(explainer_results["importances"].head(10).rename("Importance"))
                                st.write("**Surrogate Decision Tree Visualization:**"); fig_tree, ax_tree = plt.subplots(figsize=(20, 10));
                                try:
                                    class_names_tree = ["Anomalous", "Normal"]; # Assuming -1 (anom), 1 (normal)
                                    plot_tree(explainer_results["model"], filled=True, rounded=True, feature_names=features_df_cleaned.columns.tolist(), class_names=class_names_tree, ax=ax_tree, fontsize=10, max_depth=surrogate_max_depth )
                                    st.pyplot(fig_tree)
                                except Exception as e_plot: st.error(f"Error plotting tree: {e_plot}")
                            else: st.info("Surrogate tree model not available.")
                        # else: st.info("Train surrogate tree to see explanation.") # Covered by button logic

                        # Event Correlations for Anomalies
                        event_cols_in_features_df_anom = [col for col in features_df_cleaned.columns if col.startswith(st.session_state.time_series_specs.get("selected_value_col_for_analysis", "value") + "_evt_count_")]
                        if event_cols_in_features_df_anom:
                            st.markdown("---"); st.subheader("Event Correlations with Anomalies (Post-Module Analysis)")
                            event_corr_anom_df, error_eca = analyze_event_correlations(
                                features_df_cleaned, pop_anom_labels,
                                event_feature_prefix=f"{st.session_state.time_series_specs.get('selected_value_col_for_analysis', 'value')}_evt_count_"
                            )
                            if error_eca: st.error(f"Error analyzing event correlations for anomalies: {error_eca}")
                            elif event_corr_anom_df is not None and not event_corr_anom_df.empty:
                                st.write("Mean Event Counts (Anomalous vs. Normal Devices vs. Overall):"); st.dataframe(event_corr_anom_df)
                            else: st.info("No event correlation data for anomalies or no event count features found.")

                        # Significant Event Type Analysis for Anomalies
                        event_feature_prefix_anom_explain = f"{st.session_state.time_series_specs.get('selected_value_col_for_analysis', 'value')}_evt_count_"
                        event_cols_for_sig_anom_check_explain = [col for col in features_df_cleaned.columns if col.startswith(event_feature_prefix_anom_explain)]
                        if event_cols_for_sig_anom_check_explain:
                            st.markdown("---"); st.subheader("Significant Event Type Analysis (Anomalies vs. Normals, Post-Module Analysis)")
                            try:
                                sig_event_types_anom_df, error_sig_anom = analyze_significant_event_types(
                                    features_df_cleaned, pop_anom_labels, event_feature_prefix_anom_explain,
                                    at_risk_label=-1, baseline_label=1
                                )
                                if error_sig_anom: st.warning(f"Significant event analysis error for anomalies: {error_sig_anom}")
                                elif sig_event_types_anom_df is not None and not sig_event_types_anom_df.empty:
                                    # ... (markdown for explanation) ...
                                    st.dataframe(sig_event_types_anom_df)
                                else: st.info("No significant event types found for anomalies.")
                            except Exception as e_sig_anom: st.error(f"Unexpected error in significant event analysis for anomalies: {e_sig_anom}")
                    # else: st.info("Run Population Anomaly Detection to generate labels for explanation.") # Covered by outer check

            with tab_pop_clustering:
                clustering_module = ClusteringAnalysisModule()
                module_key = "clustering_module" # Unique key for this module instance

                # Initialize module-specific state if it doesn't exist
                if f'{module_key}_params' not in st.session_state:
                    default_params = {}
                    for k, v_dict in clustering_module.get_parameter_definitions().items():
                        default_params[k] = v_dict.get("default")
                    st.session_state[f'{module_key}_params'] = default_params

                if f'{module_key}_results' not in st.session_state:
                     st.session_state[f'{module_key}_results'] = {"status": "pending"} # Initial status

                # Check if features_df_cleaned has changed since last clustering run
                current_features_df_id_clust = id(features_df_cleaned) if features_df_cleaned is not None else None
                if "last_clustering_df_id" not in st.session_state or \
                   current_features_df_id_clust != st.session_state.get("last_clustering_df_id"):
                    # Reset results if the underlying data has changed.
                    st.session_state[f'{module_key}_results'] = {"status": "pending"}
                    st.session_state.last_clustering_df_id = current_features_df_id

                # UI for parameters
                with st.expander("Clustering Parameters", expanded=True):
                    updated_params = clustering_module.render_parameters_ui(
                        st,
                        st.session_state[f'{module_key}_params'],
                        module_key
                    )
                    st.session_state[f'{module_key}_params'] = updated_params

                    # Centralized run button logic - main.py orchestrates the call
                    # The module's render_parameters_ui might return an 'action' if its internal buttons are clicked
                    action = updated_params.get("action") # e.g., 'calculate_k_stats' or None

                    if st.button("Run Clustering Analysis", key=f"{module_key}_run_main_button"):
                        action = "run_clustering" # Override if main button is pressed

                    if action:
                        if not features_df_cleaned.empty:
                            st.session_state["last_clustering_feature_columns"] = features_df_cleaned.columns.tolist()

                            # Pass the action to run_analysis via params if needed, or handle here
                            # Current module design expects 'action' in params for K-stats
                            run_params_for_module = st.session_state[f'{module_key}_params'].copy()
                            if action == "run_clustering": # Ensure 'action' is not 'calculate_k_stats' if running full clustering
                                if run_params_for_module.get("action") == "calculate_k_stats":
                                     run_params_for_module["action"] = None # Clear it for full run
                            else: # if action is 'calculate_k_stats'
                                run_params_for_module["action"] = "calculate_k_stats"


                            with st.spinner("Running clustering analysis..."):
                                results, error_msg = clustering_module.run_analysis(
                                    features_df_cleaned,
                                    run_params_for_module, # Pass the potentially modified params
                                    st.session_state
                                )
                                if error_msg:
                                    st.error(error_msg)
                                    st.session_state[f'{module_key}_results'] = {"status": "error", "error_message": error_msg}
                                else:
                                    st.session_state[f'{module_key}_results'] = results
                                    st.success("Clustering analysis complete.")
                                    # If k_stats were calculated, they are inside 'results' dict.
                        else:
                            st.warning("No feature data available to run clustering.")
                        # Reset action in params to avoid re-triggering without explicit button press
                        st.session_state[f'{module_key}_params']["action"] = None
                        st.experimental_rerun()


                # Display results using the module's render_results
                results_to_render = st.session_state.get(f'{module_key}_results')
                if results_to_render and results_to_render.get("status") != "pending":
                    if results_to_render.get("status") == "error":
                        st.error(f"Clustering Error: {results_to_render.get('error_message', 'Unknown error')}")
                    else:
                        clustering_module.render_results(st, results_to_render, st.session_state)
                else:
                    st.info("Clustering analysis not yet run or no results to display.")

                # --- Existing Explainability Logic (Needs to use new state keys) ---
                # IMPORTANT: 'clust_results' now needs to come from st.session_state[f'{module_key}_results']
                clust_results_from_module = st.session_state.get(f'{module_key}_results', {})

                # Only proceed if clustering is done and labels are available
                if clust_results_from_module.get("status") == "clustering_done" and \
                   isinstance(clust_results_from_module.get('labels'), pd.Series):

                    st.subheader("Explain Clusters (Post-Module Analysis)");
                    # Pass labels from the module's results
                    labels_for_explain = clust_results_from_module['labels']
                    importance_df, err_imp = get_feature_importance_for_clusters_anova(features_df_cleaned, labels_for_explain)
                    if err_imp: st.error(f"Importance error: {err_imp}")
                    elif importance_df is not None and not importance_df.empty: st.write("Top Differentiating Features (ANOVA):"); st.dataframe(importance_df.head(10))
                    else: st.info("Could not get feature importance for clusters.")

                    summary_means_df, err_means = get_cluster_feature_summary(features_df_cleaned, labels_for_explain)
                    if err_means: st.error(f"Mean summary error: {err_means}")
                    elif summary_means_df is not None and not summary_means_df.empty:
                        st.write("Mean Feature Values per Cluster:"); st.dataframe(summary_means_df)
                        overall_mean_features_for_summary = features_df_cleaned.mean()
                        st.markdown("---"); st.markdown("**Cluster Summaries (Generated by main.py):**")
                        for cluster_id_val in sorted(labels_for_explain.unique()):
                            # Use method from module results for correct naming of noise points
                            method_used = clust_results_from_module.get("method", "Unknown")
                            cluster_name_disp = f"Cluster {cluster_id_val}" if not (cluster_id_val == -1 and method_used == "DBSCAN") else f"Noise Points ({method_used} Cluster -1)"
                            cluster_size_val = (labels_for_explain == cluster_id_val).sum()
                            cluster_means_for_this_one = summary_means_df.loc[cluster_id_val] if cluster_id_val in summary_means_df.index else pd.Series(dtype=float)

                            summary_text = generate_cluster_summary_text(
                                cluster_name_disp, cluster_size_val, len(features_df_cleaned),
                                cluster_mean_features_for_this_cluster=cluster_means_for_this_one,
                                overall_mean_features=overall_mean_features_for_summary
                            ); st.markdown(f"- {summary_text}")
                    else: st.info("No cluster mean summary data available.")

                    # Event Correlations for Clusters (using labels_for_explain)
                    event_cols_in_features_df_clust_explain = [col for col in features_df_cleaned.columns if col.startswith(st.session_state.time_series_specs.get("selected_value_col_for_analysis", "value") + "_evt_count_")]
                    if event_cols_in_features_df_clust_explain:
                        st.markdown("---"); st.subheader("Event Correlations with Clusters (Post-Module Analysis)")
                        event_corr_clust_df, error_ecc = analyze_event_correlations(
                            features_df_cleaned, labels_for_explain,
                            event_feature_prefix=f"{st.session_state.time_series_specs.get('selected_value_col_for_analysis', 'value')}_evt_count_"
                        )
                        if error_ecc: st.error(f"Error analyzing event correlations for clusters: {error_ecc}")
                        elif event_corr_clust_df is not None and not event_corr_clust_df.empty:
                            st.write("Mean Event Counts per Cluster (vs. Overall):"); st.dataframe(event_corr_clust_df)
                        else: st.info("No event correlation data for clusters or no event count features found.")

                    # Significant Event Type Analysis for Clusters (using labels_for_explain)
                    event_feature_prefix_clust_explain = f"{st.session_state.time_series_specs.get('selected_value_col_for_analysis', 'value')}_evt_count_"
                    event_cols_for_sig_clust_check_explain = [col for col in features_df_cleaned.columns if col.startswith(event_feature_prefix_clust_explain)]

                    if event_cols_for_sig_clust_check_explain:
                        st.markdown("---"); st.subheader("Significant Event Type Analysis (Between Clusters, Post-Module Analysis)")
                        available_clusters_sig_explain = sorted([c for c in labels_for_explain.unique() if not (c == -1 and clust_results_from_module.get("method") == "DBSCAN")])
                        if len(available_clusters_sig_explain) >= 1:
                            col_sig_event1_explain, col_sig_event2_explain = st.columns(2)
                            with col_sig_event1_explain:
                                selected_at_risk_cluster_sig_explain = st.selectbox("Select 'At-Risk' Cluster:", options=available_clusters_sig_explain, key="at_risk_cluster_sig_event_explain")
                            baseline_options_sig_explain = ["All Other Clusters (Combined)"] + [c for c in available_clusters_sig_explain if c != selected_at_risk_cluster_sig_explain]
                            with col_sig_event2_explain:
                                selected_baseline_option_sig_explain = st.selectbox("Select Baseline Group:", options=baseline_options_sig_explain, key="baseline_option_sig_event_explain")

                            baseline_label_for_func_sig_explain = None
                            if selected_baseline_option_sig_explain != "All Other Clusters (Combined)": baseline_label_for_func_sig_explain = selected_baseline_option_sig_explain

                            if selected_at_risk_cluster_sig_explain == baseline_label_for_func_sig_explain:
                                st.warning("At-risk and baseline clusters cannot be the same.")
                            else:
                                try:
                                    sig_event_types_clust_df_explain, error_sig_clust_explain = analyze_significant_event_types(
                                        features_df_cleaned, labels_for_explain, event_feature_prefix_clust_explain,
                                        selected_at_risk_cluster_sig_explain, baseline_label_for_func_sig_explain
                                    )
                                    if error_sig_clust_explain: st.warning(f"Significant event analysis error: {error_sig_clust_explain}")
                                    elif sig_event_types_clust_df_explain is not None and not sig_event_types_clust_df_explain.empty:
                                        # ... (markdown for explanation) ...
                                        st.dataframe(sig_event_types_clust_df_explain)
                                    else: st.info("No significant event types found for selected clusters.")
                                except Exception as e_sig_clust_explain: st.error(f"Unexpected error in significant event analysis: {e_sig_clust_explain}")
                        else: st.info("Not enough clusters for significant event type comparison.")

                    # Feature distribution plot
                    if not features_df_cleaned.empty:
                        feat_to_plot_explain = st.selectbox("Select feature to visualize by cluster (Post-Module)", options=features_df_cleaned.columns.tolist(), key="clust_feat_plot_sel_explain")
                        if feat_to_plot_explain:
                            plot_df_explain = features_df_cleaned.copy(); plot_df_explain['cluster'] = labels_for_explain
                            fig_explain, ax_explain = plt.subplots(); plot_df_explain.boxplot(column=feat_to_plot_explain, by='cluster', ax=ax_explain, grid=False)
                            ax_explain.set_title(f"Distribution of '{feat_to_plot_explain}' by Cluster"); ax_explain.set_xlabel("Cluster"); ax_explain.set_ylabel(feat_to_plot_explain); plt.suptitle(''); st.pyplot(fig_explain)
                # else: st.info("Run clustering analysis to see cluster explanations.") # This is covered by the outer check for results_to_render

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
                available_ids_in_features = set(st.session_state.all_device_features_df.index.tolist()) # Assumes all_device_features_df is the source for features_df_cleaned
                valid_failed_ids_in_dataset = parsed_ids.intersection(available_ids_in_features)
                if not valid_failed_ids_in_dataset: st.error("None of the provided failed IDs are in the current dataset's entities.")
                else:
                    st.info(f"{len(valid_failed_ids_in_dataset)} of your failed IDs are in the dataset and used for validation.")

                    # Validation for Anomaly Detection (using new module state)
                    anom_module_results_val = st.session_state.get(f'{anom_module_key}_results', {})
                    if anom_module_results_val.get("status") == "detection_done" and "labels" in anom_module_results_val:
                        st.subheader(f"Validation: Population Anomaly Detection ({anom_module_results_val.get('method', 'N/A')})")
                        anomalous_detected_ids_val = set(anom_module_results_val["labels"][anom_module_results_val["labels"] == -1].index.tolist())
                        identified_failed_anomalies_val = valid_failed_ids_in_dataset.intersection(anomalous_detected_ids_val)
                        st.metric(label="Known Failed Devices Flagged as Anomalous", value=f"{len(identified_failed_anomalies_val)} / {len(valid_failed_ids_in_dataset)}")
                        if identified_failed_anomalies_val: st.write("IDs of known failures flagged as anomalous:", sorted(list(identified_failed_anomalies_val)))
                    else: st.info("Run Population Anomaly Detection first for its validation (using the new module).")

                    # Validation for Clustering (already updated in previous step, ensure it's consistent)
                    clust_module_results_val = st.session_state.get("clustering_module_results", {}) # Using 'clustering_module' as key
                    if clust_module_results_val.get("status") == "clustering_done" and "labels" in clust_module_results_val:
                        st.subheader(f"Validation: Device Behavior Clustering ({clust_module_results_val.get('method', 'N/A')})")
                        cluster_labels_val = clust_module_results_val["labels"]; validation_summary_list = []
                        for cluster_id_iter in sorted(cluster_labels_val.unique()):
                            cluster_device_ids = set(cluster_labels_val[cluster_labels_val == cluster_id_iter].index.tolist())
                            identified_failed_in_cluster = valid_failed_ids_in_dataset.intersection(cluster_device_ids)
                            percentage_of_cluster = (len(identified_failed_in_cluster)/len(cluster_device_ids)*100) if len(cluster_device_ids)>0 else 0
                            percentage_of_total_failed = (len(identified_failed_in_cluster)/len(valid_failed_ids_in_dataset)*100) if len(valid_failed_ids_in_dataset)>0 else 0
                            method_val = clust_module_results_val.get("method", "Unknown")
                            cluster_name_val = f"Cluster {cluster_id_iter}" if not (cluster_id_iter == -1 and method_val == "DBSCAN") else f"Noise (-1, {method_val})"
                            validation_summary_list.append({"Cluster ID": cluster_name_val, "Total Devices": len(cluster_device_ids), "Known Failed in Cluster": len(identified_failed_in_cluster), "% of Cluster (Failures)": f"{percentage_of_cluster:.2f}%", "% of Total Failures in Cluster": f"{percentage_of_total_failed:.2f}%"})
                        if validation_summary_list: st.dataframe(pd.DataFrame(validation_summary_list).set_index("Cluster ID"))
                        else: st.info("Could not generate cluster validation summary.")
                    else: st.info("Run Clustering first for its validation (using the new module).")
    elif not st.session_state.data_df_original.empty : st.info("Compute 'All Device Features' from sidebar for validation.")

    st.header("General Data Table Tools")
    if not st.session_state.data_df.empty:
        if st.checkbox("Show Summary Statistics for Full Loaded Data Preview", key="general_stats_cb_general"):
            st.subheader("Summary Statistics (Loaded Data Preview)"); st.write(st.session_state.data_df.describe(include='all'))
    else: st.info("Load data using the sidebar to enable general data tools.")

# To run this app: streamlit run src/main.py

[end of src/main.py]
