import streamlit as st
import pandas as pd
import numpy as np # Added numpy
from database import (
    connect_postgres, fetch_postgres_data,
    connect_elasticsearch, fetch_elasticsearch_data
)

# --- Page Configuration ---
st.set_page_config(page_title="Data Analyzer", layout="wide")

st.title("Universal Data Analyzer 📊")

# --- Global State (if needed, e.g. for caching connections or data) ---
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = None
if 'data_df' not in st.session_state:
    st.session_state.data_df = pd.DataFrame()
if 'active_filters' not in st.session_state: # Initialize active_filters
    st.session_state.active_filters = {}

# --- Sidebar for Database Connection ---
st.sidebar.header("Database Connection")
db_type = st.sidebar.selectbox("Select Database Type", ["PostgreSQL", "Elasticsearch"])

if db_type == "PostgreSQL":
    st.sidebar.subheader("PostgreSQL Details")
    pg_host = st.sidebar.text_input("Host", "localhost")
    pg_port = st.sidebar.text_input("Port", "5432")
    pg_dbname = st.sidebar.text_input("Database Name", "mydatabase")
    pg_user = st.sidebar.text_input("User", "myuser")
    pg_password = st.sidebar.text_input("Password", type="password")
    pg_query = st.sidebar.text_area("SQL Query", "SELECT * FROM your_table_name LIMIT 100;")

    if st.sidebar.button("Connect to PostgreSQL"):
        st.session_state.db_conn = connect_postgres(pg_host, pg_port, pg_dbname, pg_user, pg_password)
        if st.session_state.db_conn:
            st.sidebar.success("Connected to PostgreSQL!")
            st.session_state.active_filters = {} # Reset filters on new connection
            st.session_state.data_df = pd.DataFrame() # Clear old data

    if st.sidebar.button("Fetch Data from PostgreSQL"):
        if st.session_state.db_conn:
            st.session_state.data_df = fetch_postgres_data(st.session_state.db_conn, pg_query)
            st.session_state.active_filters = {} # Reset filters on new data
            if not st.session_state.data_df.empty:
                st.sidebar.success("Data fetched successfully!")
        else:
            st.sidebar.error("Not connected to PostgreSQL. Please connect first.")

elif db_type == "Elasticsearch":
    st.sidebar.subheader("Elasticsearch Details")

    # Directly ask for Host URL(s)
    es_hosts_str = st.sidebar.text_input("Host URL(s) (comma-separated)", "http://localhost:9200")

    es_index = st.sidebar.text_input("Index Name", "my_index")
    es_query_dsl_str = st.sidebar.text_area("Elasticsearch Query DSL (JSON)", '{\n  "query": {\n    "match_all": {}\n  }\n}')

    if st.sidebar.button("Connect to Elasticsearch"):
        es_hosts_list = [h.strip() for h in es_hosts_str.split(',') if h.strip()] # Ensure no empty strings from multiple commas

        if not es_hosts_list: # Check if the list is empty after stripping and filtering
            st.sidebar.error("Please enter valid Elasticsearch Host URL(s).")
            st.session_state.db_conn = None # Ensure connection is reset
        else:
            st.session_state.db_conn = connect_elasticsearch(hosts=es_hosts_list)
            # connect_elasticsearch in database.py shows success/error messages via st.success/st.error
            if st.session_state.db_conn: # If connection was successful
                 st.session_state.active_filters = {}
                 st.session_state.data_df = pd.DataFrame()
            # No explicit else here because connect_elasticsearch handles st.error for failures

    if st.sidebar.button("Fetch Data from Elasticsearch"):
        if st.session_state.db_conn:
            import json
            try:
                query_body = json.loads(es_query_dsl_str)
                st.session_state.data_df = fetch_elasticsearch_data(st.session_state.db_conn, es_index, query_body)
                st.session_state.active_filters = {} # Reset filters
                if not st.session_state.data_df.empty:
                    st.sidebar.success("Data fetched successfully!")
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON in Elasticsearch Query DSL.")
            except Exception as e:
                st.sidebar.error(f"Error during Elasticsearch data fetch: {e}")
        else:
            st.sidebar.error("Not connected to Elasticsearch. Please connect first.")

# --- Main Area for Data Display ---
st.header("Loaded Data")
df_display = st.session_state.data_df.copy() # Work with a copy for display and analysis

if not df_display.empty:
    st.dataframe(df_display)
    st.info(f"Displaying {df_display.shape[0]} rows and {df_display.shape[1]} columns.")
else:
    st.info("No data loaded yet. Use the sidebar to connect to a database and fetch data.")

# --- Data Analysis Section ---
st.header("Data Analysis")

if not st.session_state.data_df.empty: # Base analysis on original loaded data state for consistency
    # df_display is already a copy from st.session_state.data_df

    # 1. Descriptive Statistics
    if st.checkbox("Show Summary Statistics"):
        st.subheader("Summary Statistics")
        st.write(st.session_state.data_df.describe(include='all')) # Show stats on original data

    # 2. Data Filtering
    st.subheader("Filter Data")
    # active_filters is already initialized in session state

    filter_cols_list = ["None"] + st.session_state.data_df.columns.tolist()
    col_to_filter = st.selectbox("Select column to filter", filter_cols_list,
                                 index=0,
                                 key="filter_column_selector")

    if col_to_filter != "None":
        selected_col_data = st.session_state.data_df[col_to_filter] # Filter based on original data

        if pd.api.types.is_numeric_dtype(selected_col_data):
            min_v, max_v = float(selected_col_data.min()), float(selected_col_data.max())
            current_range = st.session_state.active_filters.get(col_to_filter, {}).get('range', (min_v, max_v))
            if min_v < max_v:
                chosen_range = st.slider(f"Filter {col_to_filter} between", min_v, max_v, current_range, key=f"filt_num_{col_to_filter}")
                st.session_state.active_filters[col_to_filter] = {'type': 'numeric', 'range': chosen_range}
            else:
                st.info(f"Column {col_to_filter} has a single unique numeric value ({min_v}) or is empty. No range filter applicable.")
                if col_to_filter in st.session_state.active_filters:
                     del st.session_state.active_filters[col_to_filter]

        elif pd.api.types.is_bool_dtype(selected_col_data.dropna()): # Drop NA for bool checks
            # Ensure correct handling if all are NA then it's not bool
            if selected_col_data.dropna().empty:
                 st.warning(f"Column {col_to_filter} contains only NA values after dropping. Cannot apply boolean filter.")
                 if col_to_filter in st.session_state.active_filters: del st.session_state.active_filters[col_to_filter]
            else:
                options = ["Any", "True", "False"]
                current_val_str = str(st.session_state.active_filters.get(col_to_filter, {}).get('value', "Any"))
                idx = options.index(current_val_str) if current_val_str in options else 0

                choice = st.radio(f"Filter {col_to_filter}", options, index=idx, key=f"filt_bool_{col_to_filter}")
                if choice == "Any":
                    if col_to_filter in st.session_state.active_filters: del st.session_state.active_filters[col_to_filter]
                else:
                    st.session_state.active_filters[col_to_filter] = {'type': 'boolean', 'value': choice == "True"}

        else: # Categorical (object/string or other types treated as categorical)
            unique_vals = selected_col_data.dropna().unique().tolist()
            if not unique_vals:
                st.warning(f"Column {col_to_filter} has no filterable unique values (all NaN or empty).")
                if col_to_filter in st.session_state.active_filters: del st.session_state.active_filters[col_to_filter]
            else:
                # Convert all unique values to string to ensure type consistency for multiselect
                unique_vals_str = sorted([str(v) for v in unique_vals])

                current_selection_raw = st.session_state.active_filters.get(col_to_filter, {}).get('values', [])
                # Ensure current_selection is list of strings and valid
                current_selection_str = [str(v) for v in current_selection_raw if str(v) in unique_vals_str]


                chosen_values_str = st.multiselect(f"Filter {col_to_filter} by values", unique_vals_str, default=current_selection_str, key=f"filt_cat_{col_to_filter}")
                # Store chosen values, potentially converting back to original types if strictly needed, but usually string comparison is fine.
                # For simplicity, we'll store them as strings as chosen.
                st.session_state.active_filters[col_to_filter] = {'type': 'categorical', 'values': chosen_values_str}

    # Apply all active filters to df_display
    if st.session_state.active_filters:
        # df_display is already a copy of st.session_state.data_df
        # We re-assign it here based on filtering the original st.session_state.data_df
        df_filtered_display = st.session_state.data_df.copy()

        for col, filt in st.session_state.active_filters.items():
            if col not in df_filtered_display.columns: continue

            if filt['type'] == 'numeric':
                df_filtered_display = df_filtered_display[
                    (df_filtered_display[col] >= filt['range'][0]) & (df_filtered_display[col] <= filt['range'][1])
                ]
            elif filt['type'] == 'boolean':
                 # Handle potential NA values in boolean column if not dropped before filtering
                df_filtered_display = df_filtered_display[df_filtered_display[col].fillna(False) == filt['value']]
            elif filt['type'] == 'categorical':
                # Ensure comparison is between strings if values are stored as strings
                df_filtered_display = df_filtered_display[df_filtered_display[col].astype(str).isin(filt['values'])]

        # Update df_display only if filters are active and change the data
        if not df_filtered_display.equals(st.session_state.data_df):
            st.write("Filtered Data Preview:")
            st.dataframe(df_filtered_display.head())
            st.info(f"Filtered data has {df_filtered_display.shape[0]} rows and {df_filtered_display.shape[1]} columns.")
            df_display = df_filtered_display # This is the df that visualizations will use
        elif df_filtered_display.empty and st.session_state.active_filters : # Filters resulted in empty
             st.warning("Filters resulted in an empty dataset.")
             df_display = df_filtered_display # Update df_display to be empty
        # else: df_display remains the full original data_df copy if filters don't change anything or no filters active

    if st.button("Clear All Filters"):
        st.session_state.active_filters = {}
        # No need to manually rerun, Streamlit reruns on button click.
        # If state needs to be reset before other widgets are drawn, experimental_rerun can be useful.
        # For this case, the natural rerun should be sufficient.
        st.experimental_rerun()


    # 3. Basic Visualizations (operates on df_display, which is potentially filtered)
    st.subheader("Visualize Data")
    if df_display.empty:
        st.info("No data to visualize (either not loaded, or filtered to empty).")
    else:
        vis_cols_list = ["None"] + df_display.columns.tolist()
        col_to_visualize = st.selectbox("Select column to visualize", vis_cols_list,
                                        index=0,
                                        key="visualization_column_selector")

        if col_to_visualize != "None":
            st.write(f"Visualization for {col_to_visualize}:")
            vis_series = df_display[col_to_visualize].dropna()

            if vis_series.empty:
                st.warning(f"Column '{col_to_visualize}' has no data after dropping NA values (or all values were filtered out).")
            elif pd.api.types.is_numeric_dtype(vis_series):
                st.write("Histogram:")
                try:
                    # np.histogram might fail for some datatypes like timedelta if not handled
                    if pd.api.types.is_timedelta64_dtype(vis_series):
                        vis_series_numeric = vis_series.dt.total_seconds()
                        st.info("Timedelta data converted to total seconds for histogram.")
                    else:
                        vis_series_numeric = vis_series

                    hist_counts, hist_bins = np.histogram(vis_series_numeric, bins='auto')
                    hist_df = pd.DataFrame({'count': hist_counts, 'bin_start': hist_bins[:-1]})
                    st.bar_chart(hist_df.set_index('bin_start'))
                except Exception as e:
                    st.error(f"Could not generate histogram for {col_to_visualize}: {e}")
            # Check for boolean or low-cardinality categorical
            elif vis_series.dtype == 'bool' or vis_series.nunique() < 30 :
                st.write("Value Counts (Bar Chart):")
                val_counts = vis_series.value_counts().reset_index()
                # Ensure column names are strings for st.bar_chart
                val_counts.columns = [str(col_to_visualize), 'count']
                st.bar_chart(val_counts.set_index(str(col_to_visualize)))
            else: # High cardinality strings/objects
                st.warning(f"Column '{col_to_visualize}' has high cardinality ({vis_series.nunique()} unique values). A simple bar chart might not be informative. Top N values could be shown as an improvement.")
else:
    st.info("Load data using the sidebar to enable analysis features.")

# To run this app: streamlit run src/main.py
