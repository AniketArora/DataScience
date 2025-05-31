import streamlit as st
import pandas as pd
import psycopg2
from elasticsearch import Elasticsearch

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
def connect_elasticsearch(cloud_id=None, api_key=None, hosts=None):
    """
    Connects to an Elasticsearch cluster.
    Supports connection via cloud_id and api_key (recommended for Elastic Cloud)
    or via a list of host URLs for self-managed clusters.
    """
    try:
        if cloud_id and api_key:
            es = Elasticsearch(cloud_id=cloud_id, api_key=api_key)
        elif hosts:
            es = Elasticsearch(hosts)
        else:
            st.error("Elasticsearch connection requires either cloud_id & api_key or hosts.")
            return None

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
    Fetches data from Elasticsearch using the provided connection, index, and query.
    If query_body is None, it fetches all documents (up to 10000 by default).
    """
    if not es_conn:
        return pd.DataFrame()

    if query_body is None:
        query_body = {"query": {"match_all": {}}}

    try:
        # The 'scroll' API is better for large datasets, but 'search' is simpler for now.
        # Setting a large size to get more documents, default is 10.
        res = es_conn.search(index=index_name, body=query_body, size=1000)
        hits = res['hits']['hits']

        if not hits:
            st.warning(f"No documents found in index '{index_name}' for the given query.")
            return pd.DataFrame()

        # Process hits into a DataFrame
        # For simplicity, we'll just take the _source from each hit.
        # More complex processing might be needed depending on the data structure.
        data = [hit['_source'] for hit in hits]
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error fetching data from Elasticsearch index '{index_name}': {e}")
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
