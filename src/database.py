import streamlit as st
import pandas as pd
import psycopg2
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan # Added scan import

# --- PostgreSQL Connection ---
def connect_postgres(host, port, dbname, user, password):
    """Connects to a PostgreSQL database and returns a connection object."""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        return conn
    except psycopg2.Error as e:
        st.error(f"Error connecting to PostgreSQL: {e}")
        return None

def fetch_postgres_data(conn, query):
    """Fetches data from PostgreSQL using the provided connection and query."""
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data from PostgreSQL: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Elasticsearch Connection ---
def connect_elasticsearch(hosts):
    """
    Connects to an Elasticsearch cluster via a list of host URLs.
    """
    try:
        if not hosts:
            st.error("Elasticsearch connection requires host URL(s).")
            return None

        es = Elasticsearch(hosts)

        if es.ping():
            st.success("Successfully connected to Elasticsearch!")
            return es
        else:
            st.error("Failed to ping Elasticsearch. Check connection details.")
            return None
    except Exception as e:
        st.error(f"Error connecting to Elasticsearch: {e}")
        return None

def fetch_elasticsearch_data(es_conn, index_name, query_body=None):
    """
    Fetches data from Elasticsearch using the helpers.scan utility for efficiency.
    If query_body is None, it fetches all documents.
    """
    if not es_conn:
        return pd.DataFrame()

    if query_body is None:
        query_body = {"query": {"match_all": {}}}

    documents = []
    try:
        # scan is a generator, iterate through it to get all documents
        for hit in scan(client=es_conn, index=index_name, query=query_body):
            documents.append(hit['_source'])

        if not documents:
            st.warning(f"No documents found in index '{index_name}' for the given query using scan.")
            return pd.DataFrame()

        df = pd.DataFrame(documents)
        return df
    except Exception as e:
        # More specific exception handling for scan errors might be useful if identifiable
        st.error(f"Error fetching data from Elasticsearch index '{index_name}' using scan: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example Usage (primarily for testing purposes, not part of the Streamlit app flow)

    # --- PostgreSQL Example (requires a running PostgreSQL instance) ---
    # print("Testing PostgreSQL connection...")
    # pg_conn = connect_postgres("localhost", "5432", "mydatabase", "myuser", "mypassword")
    # if pg_conn:
    #     print("PostgreSQL connection successful.")
    #     # Example: list tables (adjust query for your DB)
    #     # df_tables = fetch_postgres_data(pg_conn, "SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    #     # print("Tables:", df_tables)
    #     pg_conn.close()
    # else:
    #     print("PostgreSQL connection failed.")

    # --- Elasticsearch Example (requires a running Elasticsearch instance) ---
    # print("\nTesting Elasticsearch connection...")
    # # Option 1: Elastic Cloud (replace with your cloud_id and api_key)
    # # es_instance = connect_elasticsearch(cloud_id="YOUR_CLOUD_ID", api_key=("YOUR_API_KEY_ID", "YOUR_API_KEY"))

    # # Option 2: Self-managed (replace with your host)
    # es_instance = connect_elasticsearch(hosts=["http://localhost:9200"])

    # if es_instance:
    #     print("Elasticsearch connection successful.")
    #     # Example: fetch data from an index (replace 'my_index' with your index name)
    #     # df_es_data = fetch_elasticsearch_data(es_instance, "my_index")
    #     # if not df_es_data.empty:
    #     #     print(f"Data from Elasticsearch index 'my_index':\n{df_es_data.head()}")
    #     # else:
    #     #     print(f"No data found or error fetching from 'my_index'.")
    # else:
    #     print("Elasticsearch connection failed.")
    pass
