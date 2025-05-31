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
    detect_anomalies_iqr
)
import matplotlib.pyplot as plt # Ensure this is imported for plotting


# --- Page Configuration ---
st.set_page_config(page_title="Data Analyzer", layout="wide")
st.title("Universal Data Analyzer 📊")

# --- Global State Initialization ---
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None
if 'data_df_original' not in st.session_state:
    st.session_state.data_df_original = pd.DataFrame()
if 'data_df' not in st.session_state:
    st.session_state.data_df = pd.DataFrame()
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {}
if 'time_series_specs' not in st.session_state:
    st.session_state.time_series_specs = {
        "id_cols": [], "timestamp_col": "None", "value_cols": [],
        "selected_id": "None", "selected_value_col_for_analysis": "None",
        "processed_series": None
    }

# --- Utility for resetting TS specs and temp columns ---
def reset_ts_and_temp_cols():
    temp_id_col_name = "_temp_unique_id_"
    if temp_id_col_name in st.session_state.data_df_original.columns:
        try:
            del st.session_state.data_df_original[temp_id_col_name]
        except KeyError: pass # Might have already been deleted by another path
    if temp_id_col_name in st.session_state.data_df.columns:
         try:
            del st.session_state.data_df[temp_id_col_name]
         except KeyError: pass


    st.session_state.time_series_specs = {
        "id_cols": [], "timestamp_col": "None", "value_cols": [],
        "selected_id": "None", "selected_value_col_for_analysis": "None",
        "processed_series": None
    }
    st.session_state.active_filters = {}

# --- Sidebar for Database Connection ---
st.sidebar.header("Database Connection")
db_type = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "Elasticsearch"], key="db_type_select")

if db_type == "PostgreSQL":
    st.sidebar.subheader("PostgreSQL Details")
    pg_host = st.sidebar.text_input("Host", "localhost", key="pg_host")
    pg_port = st.sidebar.text_input("Port", "5432", key="pg_port")
    pg_dbname = st.sidebar.text_input("Database Name", "mydatabase", key="pg_dbname")
    pg_user = st.sidebar.text_input("User", "myuser", key="pg_user")
    pg_password = st.sidebar.text_input("Password", type="password", key="pg_password")
    pg_query = st.sidebar.text_area("SQL Query", "SELECT * FROM your_table_name LIMIT 100;", key="pg_query")

    if st.sidebar.button("Connect to PostgreSQL", key="pg_connect_btn"):
        st.session_state.db_conn = connect_postgres(pg_host, pg_port, pg_dbname, pg_user, pg_password)
        if st.session_state.db_conn:
            st.sidebar.success("Connected to PostgreSQL!")
            st.session_state.data_df_original = pd.DataFrame()
            st.session_state.data_df = pd.DataFrame()
            reset_ts_and_temp_cols()

    if st.sidebar.button("Fetch Data from PostgreSQL", key="pg_fetch_btn"):
        if st.session_state.db_conn:
            reset_ts_and_temp_cols()
            st.session_state.data_df_original = fetch_postgres_data(st.session_state.db_conn, pg_query)
            st.session_state.data_df = st.session_state.data_df_original.copy()
            if not st.session_state.data_df_original.empty:
                st.sidebar.success("Data fetched successfully!")
            else:
                st.sidebar.warning("No data returned from PostgreSQL.")
        else:
            st.sidebar.error("Not connected to PostgreSQL. Please connect first.")

elif db_type == "Elasticsearch":
    st.sidebar.subheader("Elasticsearch Details")
    es_hosts_str = st.sidebar.text_input("Host URL(s) (comma-separated)", "http://localhost:9200", key="es_hosts")
    es_index = st.sidebar.text_input("Index Name", "my_index", key="es_index")
    es_query_dsl_str = st.sidebar.text_area("Elasticsearch Query DSL (JSON)", '{\n  "query": {\n    "match_all": {}\n  }\n}', key="es_query")

    if st.sidebar.button("Connect to Elasticsearch", key="es_connect_btn"):
        es_hosts_list = [h.strip() for h in es_hosts_str.split(',') if h.strip()]
        if not es_hosts_list:
            st.sidebar.error("Please enter valid Elasticsearch Host URL(s).")
            st.session_state.db_conn = None
        else:
            st.session_state.db_conn = connect_elasticsearch(hosts=es_hosts_list)
            if st.session_state.db_conn:
                st.session_state.data_df_original = pd.DataFrame()
                st.session_state.data_df = pd.DataFrame()
                reset_ts_and_temp_cols()

    if st.sidebar.button("Fetch Data from Elasticsearch", key="es_fetch_btn"):
        if st.session_state.db_conn:
            import json
            try:
                query_body = json.loads(es_query_dsl_str)
                reset_ts_and_temp_cols()
                st.session_state.data_df_original = fetch_elasticsearch_data(st.session_state.db_conn, es_index, query_body)
                st.session_state.data_df = st.session_state.data_df_original.copy()
                if not st.session_state.data_df_original.empty:
                    st.sidebar.success("Data fetched successfully!")
                else:
                    st.sidebar.warning("No data returned from Elasticsearch.")
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON in Elasticsearch Query DSL.")
            except Exception as e:
                st.sidebar.error(f"Error during Elasticsearch data fetch: {e}")
        else:
            st.sidebar.error("Not connected to Elasticsearch. Please connect first.")

# --- Time Series Settings (Sidebar) ---
st.sidebar.header("Time Series Settings")
temp_id_col_name = "_temp_unique_id_"

if not st.session_state.data_df_original.empty:
    df_columns = st.session_state.data_df_original.columns.tolist()

    current_id_cols = st.session_state.time_series_specs.get("id_cols", [])
    st.session_state.time_series_specs["id_cols"] = st.sidebar.multiselect(
        "Select Device/Entity ID Column(s) (Optional)",
        options=df_columns,
        default=[col for col in current_id_cols if col in df_columns],
        key="ts_id_cols"
    )

    current_ts_col = st.session_state.time_series_specs.get("timestamp_col", "None")
    ts_col_options = ["None"] + df_columns
    st.session_state.time_series_specs["timestamp_col"] = st.sidebar.selectbox(
        "Select Timestamp Column",
        options=ts_col_options,
        index=ts_col_options.index(current_ts_col) if current_ts_col in ts_col_options else 0,
        key="ts_timestamp_col"
    )

    current_value_cols = st.session_state.time_series_specs.get("value_cols", [])
    st.session_state.time_series_specs["value_cols"] = st.sidebar.multiselect(
        "Select Value/Metric Column(s) to Analyze",
        options=df_columns,
        default=[col for col in current_value_cols if col in df_columns],
        key="ts_value_cols"
    )

    if st.session_state.time_series_specs["timestamp_col"] != "None" and \
       st.session_state.time_series_specs["value_cols"]:

        unique_ids_display = ["Default Time Series"]

        if st.session_state.time_series_specs["id_cols"]:
            try:
                df_for_ids = st.session_state.data_df_original
                # Ensure temp_id_col_name is created on data_df_original if not present
                if temp_id_col_name not in df_for_ids.columns or \
                   not df_for_ids[temp_id_col_name].equals(df_for_ids[st.session_state.time_series_specs["id_cols"]].astype(str).agg('-'.join, axis=1)):
                    st.session_state.data_df_original[temp_id_col_name] = df_for_ids[st.session_state.time_series_specs["id_cols"]].astype(str).agg('-'.join, axis=1)

                unique_ids_list = sorted(st.session_state.data_df_original[temp_id_col_name].unique().tolist())
                unique_ids_display = ["None"] + unique_ids_list
            except KeyError as e:
                st.sidebar.warning(f"One or more ID columns not found: {e}. Please reselect.")
                st.session_state.time_series_specs["id_cols"] = []

        current_selected_id = st.session_state.time_series_specs.get("selected_id", "None")
        st.session_state.time_series_specs["selected_id"] = st.sidebar.selectbox(
            "Select a specific Device/Entity ID to analyze",
            options=unique_ids_display,
            index=unique_ids_display.index(current_selected_id) if current_selected_id in unique_ids_display else 0,
            key="ts_selected_id"
        )

        valid_value_cols_for_selection = [col for col in st.session_state.time_series_specs["value_cols"] if col in df_columns]
        value_col_options = ["None"] + valid_value_cols_for_selection
        current_selected_value_col = st.session_state.time_series_specs.get("selected_value_col_for_analysis", "None")

        st.session_state.time_series_specs["selected_value_col_for_analysis"] = st.sidebar.selectbox(
            "Select a specific Value/Metric to analyze",
            options=value_col_options,
            index=value_col_options.index(current_selected_value_col) if current_selected_value_col in value_col_options else 0,
            key="ts_selected_value_col"
        )

        if st.sidebar.button("Prepare Time Series for Analysis", key="ts_prepare_btn"):
            ts_col = st.session_state.time_series_specs["timestamp_col"]
            val_col = st.session_state.time_series_specs["selected_value_col_for_analysis"]
            selected_entity_id = st.session_state.time_series_specs["selected_id"]

            if selected_entity_id != "None" and ts_col != "None" and val_col != "None":
                current_df_copy = st.session_state.data_df_original.copy()
                entity_series_df = current_df_copy

                if selected_entity_id != "Default Time Series":
                    if temp_id_col_name in current_df_copy.columns:
                        entity_series_df = current_df_copy[current_df_copy[temp_id_col_name] == selected_entity_id]
                    else:
                        st.sidebar.error("ID column for filtering not found. Reselect ID columns if changed.")
                        st.session_state.time_series_specs["processed_series"] = None
                        entity_series_df = pd.DataFrame() # Empty df to stop processing

                if not entity_series_df.empty:
                    try:
                        entity_series_df[ts_col] = pd.to_datetime(entity_series_df[ts_col], errors='coerce')
                        entity_series_df = entity_series_df.dropna(subset=[ts_col, val_col])

                        if entity_series_df.empty:
                            st.sidebar.error(f"No valid data after NA drop for {ts_col} or {val_col}.")
                            st.session_state.time_series_specs["processed_series"] = None
                        else:
                            entity_series_df = entity_series_df.sort_values(by=ts_col)
                            processed_series = entity_series_df.groupby(ts_col)[val_col].mean().rename(val_col)
                            st.session_state.time_series_specs["processed_series"] = processed_series
                            st.sidebar.success(f"Prepared '{val_col}' for '{selected_entity_id}'. Length: {len(processed_series)}")
                    except Exception as e:
                        st.sidebar.error(f"Error preparing series: {e}")
                        st.session_state.time_series_specs["processed_series"] = None
                elif st.session_state.time_series_specs["processed_series"] is not None : # only show if not already handled by inner empty check
                    st.sidebar.error(f"No data found for ID: {selected_entity_id} with column {val_col}.")
                    st.session_state.time_series_specs["processed_series"] = None
            else:
                st.sidebar.warning("Please select a valid Device/Entity ID, Timestamp, and Value/Metric.")
                st.session_state.time_series_specs["processed_series"] = None
    else:
        st.sidebar.info("Select Timestamp and Value columns to enable detailed time series analysis options.")
        if st.session_state.time_series_specs["processed_series"] is not None: # Clear if previously set
            st.session_state.time_series_specs["processed_series"] = None

        if temp_id_col_name in st.session_state.data_df_original.columns and \
           (st.session_state.time_series_specs.get("timestamp_col", "None") == "None" or \
            not st.session_state.time_series_specs.get("value_cols", [])):
            try:
                del st.session_state.data_df_original[temp_id_col_name]
                if temp_id_col_name in st.session_state.data_df.columns:
                     del st.session_state.data_df[temp_id_col_name]
            except KeyError: pass


elif not st.session_state.data_df_original.empty and temp_id_col_name in st.session_state.data_df_original.columns:
     if (st.session_state.time_series_specs.get("timestamp_col", "None") == "None" or \
        not st.session_state.time_series_specs.get("value_cols", [])):
            try:
                del st.session_state.data_df_original[temp_id_col_name]
                if temp_id_col_name in st.session_state.data_df.columns:
                     del st.session_state.data_df[temp_id_col_name]
            except KeyError: pass
else:
    st.sidebar.info("Load data to configure time series settings.")

# --- Main Area for Data Display ---
st.header("Loaded Data Preview")
if not st.session_state.data_df.empty:
    st.dataframe(st.session_state.data_df.head())
    st.info(f"Displaying preview of {st.session_state.data_df.shape[0]} rows and {st.session_state.data_df.shape[1]} columns.")
else:
    st.info("No data loaded yet. Use the sidebar to connect to a database and fetch data.")

# --- Display Processed Time Series (if available) ---
processed_series_display = st.session_state.time_series_specs.get("processed_series")
if processed_series_display is not None and not processed_series_display.empty:
    st.header("Prepared Time Series for Analysis")
    st.line_chart(processed_series_display)
    st.write(processed_series_display.describe())

# --- Time Series Analysis Modules ---
if processed_series_display is not None and not processed_series_display.empty:
    st.header("Time Series Analysis")
    analysis_series = processed_series_display
    series_name = analysis_series.name if analysis_series.name else "Value" # Ensure name exists

    tab_profiling, tab_decomposition, tab_anomalies = st.tabs(["📊 Profiling", "📉 Decomposition", "❗ Anomaly Detection"])

    with tab_profiling:
        st.subheader(f"Profiling: {series_name}")

        st.write("Summary Statistics (Processed Series):")
        summary_df = get_series_summary_stats(analysis_series)
        st.dataframe(summary_df)

        original_selected_series_for_profiling = pd.Series(dtype=float)
        current_df_orig = st.session_state.data_df_original.copy()
        selected_entity_id = st.session_state.time_series_specs["selected_id"]
        ts_col_profiling = st.session_state.time_series_specs["timestamp_col"]
        val_col_profiling = st.session_state.time_series_specs["selected_value_col_for_analysis"]

        if selected_entity_id != "None" and ts_col_profiling != "None" and val_col_profiling != "None":
            entity_df_orig = current_df_orig
            if selected_entity_id != "Default Time Series":
                if temp_id_col_name not in entity_df_orig.columns and st.session_state.time_series_specs["id_cols"]:
                    entity_df_orig[temp_id_col_name] = entity_df_orig[st.session_state.time_series_specs["id_cols"]].astype(str).agg('-'.join, axis=1)
                if temp_id_col_name in entity_df_orig.columns:
                    entity_df_orig = entity_df_orig[entity_df_orig[temp_id_col_name] == selected_entity_id]

            if not entity_df_orig.empty and val_col_profiling in entity_df_orig.columns and ts_col_profiling in entity_df_orig.columns:
                entity_df_orig[ts_col_profiling] = pd.to_datetime(entity_df_orig[ts_col_profiling], errors='coerce')
                # Important: For missing value analysis, we use the series *before* extensive NA dropping on value_col
                # but after timestamp conversion and potential ID filtering.
                temp_series_for_missing = entity_df_orig.dropna(subset=[ts_col_profiling]).set_index(ts_col_profiling)[val_col_profiling].sort_index()
                original_selected_series_for_profiling = temp_series_for_missing.groupby(temp_series_for_missing.index).mean()


        if not original_selected_series_for_profiling.empty:
            st.write("Missing Values (Original Segment for Selected Metric):")
            missing_df_orig = get_missing_values_summary(original_selected_series_for_profiling)
            st.dataframe(missing_df_orig)
        else:
             st.warning("Original segment for full missing value profiling could not be retrieved. Processed series is already NA-handled for its values.")


        st.write("Stationarity Test (ADF - on processed series):")
        # analysis_series is the processed series, which had its values NA-dropped during groupby().mean()
        adf_results = perform_stationarity_test(analysis_series)
        if "error" in adf_results:
            st.error(adf_results["error"])
        else:
            st.json(adf_results)

    with tab_decomposition:
        st.subheader(f"Decomposition: {series_name}")
        decomp_model = st.selectbox("Decomposition Model", ["additive", "multiplicative"], key="decomp_model_select")

        inferred_period = 1
        if analysis_series.index.inferred_freq:
            freq = analysis_series.index.inferred_freq
            if 'D' in freq: inferred_period = 7
            elif 'M' in freq or 'ME' in freq : inferred_period = 12
            elif 'Q' in freq or 'QE' in freq: inferred_period = 4
            elif 'H' in freq: inferred_period = 24

        period_help_text = (f"Seasonality period. Suggested: {inferred_period}. Ensure series has min 2 full periods.")
        decomp_period = st.number_input("Seasonal Period", min_value=2, value=inferred_period, help=period_help_text, key="decomp_period_num_input")

        if st.button("Decompose Series", key="decomp_run_btn"):
            if decomp_period and decomp_period >=2:
                # Use analysis_series which is already cleaned of NaNs in values
                decomposition_result, error_msg = decompose_time_series(analysis_series, model=decomp_model, period=decomp_period)
                if error_msg:
                    st.error(error_msg)
                if decomposition_result:
                    st.success("Decomposition successful.")
                    st.line_chart(decomposition_result.observed.rename("Observed"))
                    st.line_chart(decomposition_result.trend.rename("Trend"))
                    st.line_chart(decomposition_result.seasonal.rename("Seasonal"))
                    st.line_chart(decomposition_result.resid.rename("Residual"))
            else:
                st.warning("Please provide a valid seasonal period (>=2).")

    with tab_anomalies:
        st.subheader(f"Anomaly Detection: {series_name}")
        anomaly_method = st.selectbox("Detection Method", ["Z-score", "IQR"], key="anomaly_method_select")

        if anomaly_method == "Z-score":
            z_threshold = st.number_input("Z-score Threshold", min_value=0.5, value=3.0, step=0.1, key="z_threshold_num_input")
            z_window_options = [None, 5, 10, 15, 20, 30]
            z_window = st.selectbox("Rolling Window (optional)", options=z_window_options, index=0, key="z_window_select", help="If None, global mean/std is used.")

            if st.button("Detect Z-score Anomalies", key="zscore_run_btn"):
                # analysis_series is already cleaned of NaNs in values
                anomalies, z_scores, error_msg = detect_anomalies_zscore(analysis_series, threshold=z_threshold, window=z_window)
                if error_msg:
                    st.error(error_msg)
                elif anomalies is not None:
                    st.success(f"Z-score analysis complete. Found {anomalies.sum()} anomalies.")
                    fig, ax = plt.subplots()
                    ax.plot(analysis_series.index, analysis_series, label=series_name)
                    ax.scatter(analysis_series.index[anomalies], analysis_series[anomalies], color='red', label='Anomalies', s=50, zorder=5)
                    ax.set_title(f"{series_name} with Z-score Anomalies")
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    if st.checkbox("Show Z-scores table for anomalies", key="zscore_table_cb"):
                        st.dataframe(z_scores[anomalies].rename("Z-score of Anomaly"))

        elif anomaly_method == "IQR":
            iqr_multiplier = st.number_input("IQR Multiplier", min_value=0.5, value=1.5, step=0.1, key="iqr_multiplier_num_input")
            if st.button("Detect IQR Anomalies", key="iqr_run_btn"):
                # analysis_series is already cleaned of NaNs in values
                anomalies, bounds_info, error_msg = detect_anomalies_iqr(analysis_series, multiplier=iqr_multiplier)
                if error_msg:
                    st.error(error_msg)
                elif anomalies is not None:
                    st.success(f"IQR analysis complete. Found {anomalies.sum()} anomalies.")
                    fig, ax = plt.subplots()
                    ax.plot(analysis_series.index, analysis_series, label=series_name)
                    ax.scatter(analysis_series.index[anomalies], analysis_series[anomalies], color='red', label='Anomalies', s=50, zorder=5)

                    if bounds_info is not None:
                        lower_b = bounds_info[bounds_info['Metric'] == 'Lower Bound']['Value'].iloc[0]
                        upper_b = bounds_info[bounds_info['Metric'] == 'Upper Bound']['Value'].iloc[0]
                        ax.axhline(lower_b, color='orange', linestyle='--', label=f'Lower Bound ({lower_b:.2f})')
                        ax.axhline(upper_b, color='orange', linestyle='--', label=f'Upper Bound ({upper_b:.2f})')

                    ax.set_title(f"{series_name} with IQR Anomalies")
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    if st.checkbox("Show IQR Bounds Info", key="iqr_bounds_table_cb"):
                        st.dataframe(bounds_info)
else:
    if not st.session_state.data_df_original.empty: # Data is loaded but no series prepared
        st.info("Prepare a time series using the 'Time Series Settings' in the sidebar to enable specific time series analyses.")
    # If data_df_original is empty, the message "Load data..." from "General Data Analysis Tools" section is sufficient.

# --- General Data Analysis Tools (Existing Filters, etc.) ---
# Kept minimal as focus shifts to TS, but could be expanded or kept as is.
st.header("General Data Table Tools")

if not st.session_state.data_df.empty:
    # Simplified: only showing stats for the general table, filtering was complex and less focus now.
    if st.checkbox("Show Summary Statistics for Full Loaded Data Preview", key="general_stats_cb"):
        st.subheader("Summary Statistics (Loaded Data Preview)")
        st.write(st.session_state.data_df.describe(include='all'))
else:
    st.info("Load data using the sidebar to enable general data tools.")

# To run this app: streamlit run src/main.py
