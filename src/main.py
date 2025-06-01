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
    detect_anomalies_isolation_forest,
    detect_anomalies_one_class_svm
)
from src.analysis_modules.clustering import (
    perform_kmeans_clustering,
    get_kmeans_elbow_silhouette_data,
    perform_dbscan_clustering
)
from src.analysis_modules.feature_engineering import generate_all_features_for_series
from src.analysis_modules.explainability import (
    get_cluster_feature_summary,
    get_feature_importance_for_clusters_anova,
    compare_anomalous_vs_normal_features,
    generate_cluster_summary_text,
    generate_anomaly_summary_text,
    explain_anomalies_with_surrogate_model
)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# --- Page Configuration ---
st.set_page_config(page_title="Data Analyzer", layout="wide")
st.title("Universal Data Analyzer 📊")

# --- Global State Initialization ---
if 'db_conn' not in st.session_state: st.session_state.db_conn = None
if 'data_df_original' not in st.session_state: st.session_state.data_df_original = pd.DataFrame()
if 'data_df' not in st.session_state: st.session_state.data_df = pd.DataFrame()
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
    st.session_state.population_anomaly_results = {}
    st.session_state.clustering_results = {}
    st.session_state.kmeans_stats_df = None
    st.session_state.last_pop_anomaly_df_id = None
    st.session_state.last_clustering_df_id = None
    st.session_state.res_df_anomalies_sorted = pd.DataFrame()
    st.session_state.surrogate_tree_explainer = None


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
    st.sidebar.header("Available Workflows")
    selected_workflow = st.sidebar.selectbox(
        "Select a Guided Workflow:",
        ["Potential Failure Investigation"],
        key="guided_workflow_selector"
    )
    st.sidebar.markdown("---")

    st.header(f"Guided Workflow: {selected_workflow}")

    if selected_workflow == "Potential Failure Investigation":
        st.subheader("Step 1: Prerequisites")
        prereq_met = True
        if st.session_state.get('all_device_features_df', pd.DataFrame()).empty:
            st.warning("Please compute 'All Device Features' first in 'General Analysis' mode (Sidebar -> Population Analysis Settings -> Compute Features for ALL Devices).")
            prereq_met = False

        clustering_results_wf = st.session_state.get('clustering_results', {})
        if not clustering_results_wf or not isinstance(clustering_results_wf.get('labels'), pd.Series) or clustering_results_wf['labels'].empty:
            st.warning("Please run Device Behavior Clustering first in 'General Analysis' mode (Population-Level Analysis -> Device Behavior Clustering tab).")
            prereq_met = False

        if not prereq_met:
            st.stop()

        st.success("Prerequisites met: Device features and clustering results are available.")

        st.subheader("Step 2: Select At-Risk Cluster(s)")
        cluster_labels_series_wf = st.session_state.clustering_results['labels']
        available_clusters_wf = sorted(cluster_labels_series_wf.unique())
        available_clusters_str_wf = [str(c) for c in available_clusters_wf]

        selected_at_risk_clusters_str_wf = st.multiselect(
            "Select cluster(s) you consider 'at-risk' (based on your domain knowledge or previous validation):",
            options=available_clusters_str_wf,
            key="at_risk_cluster_multiselect_wf"
        )

        selected_at_risk_clusters_wf = []
        for s_cluster in selected_at_risk_clusters_str_wf:
            try: selected_at_risk_clusters_wf.append(int(s_cluster))
            except ValueError:
                try: selected_at_risk_clusters_wf.append(float(s_cluster))
                except ValueError: st.error(f"Could not parse cluster label: {s_cluster}")

        if not selected_at_risk_clusters_wf:
            st.info("Select one or more clusters to proceed."); st.stop()

        st.subheader("Step 3: Key Characteristics of Selected At-Risk Cluster(s)")
        features_df_cleaned_wf = st.session_state.all_device_features_df.dropna()
        if features_df_cleaned_wf.empty: st.error("Cleaned feature DataFrame is empty. Cannot proceed."); st.stop()

        combined_risk_labels_wf = cluster_labels_series_wf.isin(selected_at_risk_clusters_wf).map({True: 'At-Risk', False: 'Not At-Risk'})

        if combined_risk_labels_wf.nunique() > 1:
            importance_df_wf, error_imp_wf = get_feature_importance_for_clusters_anova(features_df_cleaned_wf, combined_risk_labels_wf)
            if error_imp_wf: st.warning(f"Could not determine distinguishing features (ANOVA): {error_imp_wf}")
            elif importance_df_wf is not None and not importance_df_wf.empty:
                st.write("Top Features Distinguishing Selected At-Risk Group vs. Others:"); st.dataframe(importance_df_wf.head(5))
            else: st.info("No significant distinguishing features found via ANOVA for the selected group(s).")
        else: st.info("Only one group ('At-Risk' or 'Not At-Risk') present; cannot perform comparative ANOVA.")

        mean_summary_df_wf, error_means_wf = get_cluster_feature_summary(features_df_cleaned_wf, combined_risk_labels_wf)
        if error_means_wf: st.warning(f"Could not calculate mean feature values: {error_means_wf}")
        elif mean_summary_df_wf is not None and not mean_summary_df_wf.empty:
            st.write("Mean Feature Values (Selected At-Risk Group vs. Others):"); st.dataframe(mean_summary_df_wf)

        st.subheader("Step 4: Devices to Monitor")
        devices_in_at_risk_clusters_wf = cluster_labels_series_wf[cluster_labels_series_wf.isin(selected_at_risk_clusters_wf)].index.tolist()

        if not devices_in_at_risk_clusters_wf: st.info("No devices found in the selected at-risk cluster(s).")
        else:
            st.write(f"Found {len(devices_in_at_risk_clusters_wf)} devices in the selected at-risk cluster(s):")
            known_failed_ids_input_wf = st.text_area(
                "Enter known failed Device/Entity IDs (one per line or comma-separated) for cross-referencing:",
                height=100, key="failed_ids_input_workflow"
            ).strip()
            devices_to_monitor_df_wf = pd.DataFrame(devices_in_at_risk_clusters_wf, columns=["Device ID"])

            if known_failed_ids_input_wf:
                parsed_ids_wf = set(item.strip() for line in known_failed_ids_input_wf.split("\n") for item in line.split(",") if item.strip())
                devices_to_monitor_df_wf['Is_Known_Failure'] = devices_to_monitor_df_wf['Device ID'].isin(parsed_ids_wf)
                st.dataframe(devices_to_monitor_df_wf.sort_values(by=['Is_Known_Failure', 'Device ID']))
                not_yet_failed_in_at_risk_wf = devices_to_monitor_df_wf[~devices_to_monitor_df_wf['Is_Known_Failure']]
                st.write(f"**{len(not_yet_failed_in_at_risk_wf)} devices in at-risk clusters are NOT in your provided known failures list.**")
                if not not_yet_failed_in_at_risk_wf.empty: st.dataframe(not_yet_failed_in_at_risk_wf[['Device ID']].reset_index(drop=True))
            else:
                st.dataframe(devices_to_monitor_df_wf)
                st.info("Provide known failed IDs above to see which of these at-risk devices are new potential concerns.")

elif app_mode == "General Analysis":
    # --- ALL EXISTING UI LOGIC INDENTED HERE ---
    st.sidebar.header("Database Connection")
    db_type = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "Elasticsearch"], key="db_type_select_general") # Changed key
    if db_type == "PostgreSQL":
        st.sidebar.subheader("PostgreSQL Details"); pg_host = st.sidebar.text_input("Host", "localhost", key="pg_host_general"); pg_port = st.sidebar.text_input("Port", "5432", key="pg_port_general"); pg_dbname = st.sidebar.text_input("Database Name", "mydatabase", key="pg_dbname_general"); pg_user = st.sidebar.text_input("User", "myuser", key="pg_user_general"); pg_password = st.sidebar.text_input("Password", type="password", key="pg_password_general"); pg_query = st.sidebar.text_area("SQL Query", "SELECT * FROM your_table_name LIMIT 100;", key="pg_query_general")
        if st.sidebar.button("Connect to PostgreSQL", key="pg_connect_btn_general"):
            st.session_state.db_conn = connect_postgres(pg_host, pg_port, pg_dbname, pg_user, pg_password)
            if st.session_state.db_conn: st.sidebar.success("Connected!"); st.session_state.data_df_original = pd.DataFrame(); st.session_state.data_df = pd.DataFrame(); reset_ts_and_temp_cols()
        if st.sidebar.button("Fetch Data from PostgreSQL", key="pg_fetch_btn_general"):
            if st.session_state.db_conn: reset_ts_and_temp_cols(); st.session_state.data_df_original = fetch_postgres_data(st.session_state.db_conn, pg_query); st.session_state.data_df = st.session_state.data_df_original.copy();
            if not st.session_state.data_df_original.empty: st.sidebar.success("Data fetched!")
            else: st.sidebar.warning("No data returned.")
            else: st.sidebar.error("Not connected.")
    elif db_type == "Elasticsearch":
        st.sidebar.subheader("Elasticsearch Details"); es_hosts_str = st.sidebar.text_input("Host URL(s) (comma-separated)", "http://localhost:9200", key="es_hosts_general"); es_index = st.sidebar.text_input("Index Name", "my_index", key="es_index_general"); es_query_dsl_str = st.sidebar.text_area("Elasticsearch Query DSL (JSON)", '{\n  "query": {\n    "match_all": {}\n  }\n}', key="es_query_general")
        if st.sidebar.button("Connect to Elasticsearch", key="es_connect_btn_general"):
            es_hosts_list = [h.strip() for h in es_hosts_str.split(',') if h.strip()]
            if not es_hosts_list: st.sidebar.error("Enter Host URL(s)."); st.session_state.db_conn = None
            else:
                st.session_state.db_conn = connect_elasticsearch(hosts=es_hosts_list)
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
                    df_for_ids = st.session_state.data_df_original
                    id_col_data_check = df_for_ids[st.session_state.time_series_specs["id_cols"]].astype(str).agg('-'.join, axis=1)
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
    elif not st.session_state.data_df_original.empty and temp_id_col_name in st.session_state.data_df_original.columns:
         if (st.session_state.time_series_specs.get("timestamp_col","None")=="None" or not st.session_state.time_series_specs.get("value_cols",[])):
                try: del st.session_state.data_df_original[temp_id_col_name]
                except KeyError:pass
            if temp_id_col_name in st.session_state.data_df.columns:
                try: del st.session_state.data_df[temp_id_col_name]
                except KeyError:pass
    else: # If data_df_original is empty
        st.sidebar.info("Load data to configure Time Series Settings.")

    st.sidebar.markdown("---") # Visual separator
    st.sidebar.subheader("Analysis Configuration (Future)")
    st.sidebar.button("Save Time Series Settings", disabled=True, key="save_ts_settings_button_placeholder_general")
    st.sidebar.info("Future: Load/Save selected columns and preparation settings.")

elif not st.session_state.data_df_original.empty and temp_id_col_name in st.session_state.data_df_original.columns: # This elif is part of the outer if not data_df_original.empty block
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
        st.session_state.running_all_features_computation=True; st.session_state.all_device_features_df=pd.DataFrame(); st.session_state.population_anomaly_results={}; st.session_state.clustering_results={}; st.session_state.kmeans_stats_df=None; st.session_state.res_df_anomalies_sorted=pd.DataFrame()

    st.header("Loaded Data Preview")
    if not st.session_state.data_df.empty: st.dataframe(st.session_state.data_df.head()); st.info(f"Preview: {st.session_state.data_df.shape[0]} rows, {st.session_state.data_df.shape[1]} columns.")
    else: st.info("No data loaded.")

    processed_series_display = st.session_state.time_series_specs.get("processed_series")
    if processed_series_display is not None and not processed_series_display.empty:
        st.header("Prepared Time Series for Analysis"); st.line_chart(processed_series_display); st.write(processed_series_display.describe())
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
            st.subheader(f"Decomposition: {series_name}"); decomp_model = st.selectbox("Model", ["additive", "multiplicative"], key="dc_model_general"); decomp_period = st.number_input("Period",2,None,7,key="dc_period_general");
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
                    with st.spinner("Calculating..."): feature_name_prefix = f"{analysis_series.name}_" if analysis_series.name else "series_"; st.session_state.single_series_features_display = generate_all_features_for_series(analysis_series, name=feature_name_prefix)
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
                    if not processed_entity_series_all.empty: features = generate_all_features_for_series(processed_entity_series_all, name=f"{val_col_to_process_for_all}_"); features['entity_id'] = entity_id_iter; all_features_list.append(features)
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
                st.subheader("Unsupervised Anomaly Detection on Device Features")
                if "last_pop_anomaly_df_id" not in st.session_state or id(features_df_cleaned) != st.session_state.get("last_pop_anomaly_df_id"): st.session_state.population_anomaly_results = {}; st.session_state.last_pop_anomaly_df_id = id(features_df_cleaned); st.session_state.surrogate_tree_explainer = None # Reset surrogate on new data
                anomaly_method_pop = st.selectbox("Method", ["Isolation Forest", "One-Class SVM"], key="pop_an_method_general")
                if anomaly_method_pop == "Isolation Forest":
                    contam_if = st.slider("Contamination", 0.01, 0.5, 0.1, 0.01, key="if_contam_general");
                st.markdown("---")
                st.button("Save Isolation Forest Settings", disabled=True, key="save_if_settings_button_placeholder_general")
                st.info("Future: Load/Save Isolation Forest parameters.")
                    if st.button("Run Isolation Forest", key="run_if_pop_general"):
                        labels, scores, error_msg = detect_anomalies_isolation_forest(features_df_cleaned, contamination=contam_if)
                        if error_msg: st.error(error_msg); else: st.session_state.population_anomaly_results = {'method':'Isolation Forest', 'labels':labels, 'scores':scores}; st.success("IForest complete.")
                elif anomaly_method_pop == "One-Class SVM":
                    nu_ocsvm = st.slider("Nu", 0.01, 0.5, 0.05, 0.01, key="ocsvm_nu_general"); kernel_ocsvm = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="ocsvm_kernel_general"); gamma_ocsvm_text = st.text_input("Gamma", value='scale', key="ocsvm_gamma_general");
                    try: gamma_val = float(gamma_ocsvm_text)
                    except ValueError: gamma_val = gamma_ocsvm_text

                st.markdown("---")
                st.button("Save One-Class SVM Settings", disabled=True, key="save_ocsvm_settings_button_placeholder_general")
                st.info("Future: Load/Save One-Class SVM parameters.")

                    if st.button("Run One-Class SVM", key="run_ocsvm_pop_general"):
                        labels, scores, error_msg = detect_anomalies_one_class_svm(features_df_cleaned, nu=nu_ocsvm, kernel=kernel_ocsvm, gamma=gamma_val)
                        if error_msg: st.error(error_msg); else: st.session_state.population_anomaly_results = {'method':'One-Class SVM', 'labels':labels, 'scores':scores}; st.success("OC-SVM complete.")
                pop_results = st.session_state.get('population_anomaly_results')
                if pop_results and 'labels' in pop_results:
                    st.write(f"Results: {pop_results['method']}"); res_df = pd.DataFrame({'label':pop_results['labels'], 'score':pop_results['scores']}).sort_values(by='score'); st.session_state.res_df_anomalies_sorted = res_df
                    st.write(f"Found {(res_df['label'] == -1).sum()} anomalies."); st.dataframe(res_df.head())
                    fig, ax = plt.subplots(); scores_sorted = pop_results['scores'].sort_values(); scores_sorted.plot(kind='bar', ax=ax, title=f"Scores ({pop_results['method']})"); ax.set_xticks([]); ax.set_ylabel("Score"); st.pyplot(fig)
                    st.subheader("Explain Anomalies"); comparison_df, error_comp = compare_anomalous_vs_normal_features(features_df_cleaned, pop_results["labels"], anomalous_label_val=-1)
                    if error_comp: st.error(f"Comparison error: {error_comp}")
                    elif comparison_df is not None and not comparison_df.empty: st.write("Mean Feature Comparison (Anomalous vs. Normal):"); st.dataframe(comparison_df.head(10))
                    if not st.session_state.res_df_anomalies_sorted.empty: top_anomaly_id = st.session_state.res_df_anomalies_sorted.index[0]; top_anomaly_score = st.session_state.res_df_anomalies_sorted.iloc[0]["score"]; st.markdown("---"); st.markdown(f"**Summary for Top Anomaly ({top_anomaly_id}):**"); summary_text = generate_anomaly_summary_text(top_anomaly_id, top_anomaly_score, comparison_df); st.markdown(summary_text)
                    else: st.info("No data for feature comparison.")
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
            with tab_pop_clustering:
                st.subheader("Device Behavior Clustering")
                if "last_clustering_df_id" not in st.session_state or id(features_df_cleaned) != st.session_state.get("last_clustering_df_id"): st.session_state.clustering_results = {}; st.session_state.kmeans_stats_df = None; st.session_state.last_clustering_df_id = id(features_df_cleaned)
                cluster_method = st.selectbox("Method", ["K-Means", "DBSCAN"], key="pop_clust_method_general"); scale_data_clustering = st.checkbox("Scale data", value=True, key="scale_data_clust_cb_general")
                if cluster_method == "K-Means":
                    st.write("Optimal K:"); k_min = st.number_input("Min K", 2, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2, 2, key="k_min_kstats_general"); k_max_val = min(10, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2); k_max = st.number_input("Max K", k_min, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2, k_max_val if k_max_val >= k_min else k_min, key="k_max_kstats_general")
                    if st.button("Calc K-Means Stats", key="calc_km_stats_btn_general"):
                        if k_max >= k_min: k_stats_df, err_msg = get_kmeans_elbow_silhouette_data(features_df_cleaned, k_range=range(k_min, k_max+1), scale_data=scale_data_clustering);
                        if err_msg: st.error(err_msg); else: st.session_state.kmeans_stats_df = k_stats_df
                        else: st.warning("Max K >= Min K.")
                    if st.session_state.get("kmeans_stats_df") is not None: k_stats_df_display = st.session_state.kmeans_stats_df; st.line_chart(k_stats_df_display.set_index('K')['Inertia']); st.line_chart(k_stats_df_display.set_index('K')['Silhouette Score'].dropna())
                    k_final = st.number_input("Num Clusters (K)", 2, len(features_df_cleaned)-1 if len(features_df_cleaned)>2 else 2, 3, key="km_k_final_general");

                st.markdown("---")
                st.button("Save K-Means Settings", disabled=True, key="save_kmeans_settings_button_placeholder_general")
                st.info("Future: Load/Save K-Means parameters.")

                    if st.button("Run K-Means", key="run_km_pop_btn_general"):
                        labels, model, err_msg = perform_kmeans_clustering(features_df_cleaned, n_clusters=k_final, scale_data=scale_data_clustering)
                        if err_msg: st.error(err_msg); else: st.session_state.clustering_results = {'method':'K-Means', 'labels':labels, 'model':model, 'k':k_final}; st.success(f"K-Means complete (K={k_final}).")
                elif cluster_method == "DBSCAN":
                    eps_dbscan = st.number_input("Epsilon", 0.01, 10.0, 0.5, 0.01, key="db_eps_general"); min_samples_dbscan = st.number_input("Min Samples", 1, 100, 5, 1, key="db_min_samples_general")

                st.markdown("---")
                st.button("Save DBSCAN Settings", disabled=True, key="save_dbscan_settings_button_placeholder_general")
                st.info("Future: Load/Save DBSCAN parameters.")

                    if st.button("Run DBSCAN", key="run_db_pop_btn_general"):
                        labels, model, err_msg = perform_dbscan_clustering(features_df_cleaned, eps=eps_dbscan, min_samples=min_samples_dbscan, scale_data=scale_data_clustering)
                        if err_msg: st.error(err_msg); else: st.session_state.clustering_results = {'method':'DBSCAN', 'labels':labels, 'model':model}; st.success("DBSCAN complete.")
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
                    elif summary_means_df is not None and not summary_means_df.empty: st.write("Mean Feature Values per Cluster:"); st.dataframe(summary_means_df); st.markdown("---"); st.markdown("**Cluster Summaries:**")
                    for cluster_id_val in sorted(clust_results["labels"].unique()):
                        cluster_name_disp = f"Cluster {cluster_id_val}" if not (cluster_id_val == -1 and clust_results.get("method") == "DBSCAN") else "Noise Points (DBSCAN Cluster -1)"
                        summary_text = generate_cluster_summary_text(cluster_name_disp, (clust_results["labels"] == cluster_id_val).sum(), len(features_df_cleaned), importance_df); st.markdown(f"- {summary_text}")
                    else: st.info("No cluster mean summary.")
                    if not features_df_cleaned.empty:
                        feat_to_plot = st.selectbox("Select feature for cluster visualization", options=features_df_cleaned.columns.tolist(), key="clust_feat_plot_sel_general")
                        if feat_to_plot: plot_df = features_df_cleaned.copy(); plot_df['cluster'] = clust_results['labels']; fig, ax = plt.subplots(); plot_df.boxplot(column=feat_to_plot, by='cluster', ax=ax, grid=False); ax.set_title(f"Distribution of '{feat_to_plot}' by Cluster"); ax.set_xlabel("Cluster"); ax.set_ylabel(feat_to_plot); plt.suptitle(''); st.pyplot(fig)
    elif not st.session_state.get("running_all_features_computation") and st.session_state.get("compute_all_features_button_general"): st.warning("Feature computation for all devices resulted in an empty dataset.") # Changed key
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
