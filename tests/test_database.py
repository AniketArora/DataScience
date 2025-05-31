import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import psycopg2 # Import for psycopg2.Error
from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError # Import for ES errors

# Assuming your database.py file is in src directory and src is in PYTHONPATH
from src.database import (
    connect_postgres, fetch_postgres_data,
    connect_elasticsearch, fetch_elasticsearch_data
)

# --- PostgreSQL Tests ---
@patch('src.database.psycopg2.connect')
def test_connect_postgres_success(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    conn = connect_postgres("host", "port", "dbname", "user", "password")
    mock_connect.assert_called_once_with(
        host="host", port="port", dbname="dbname", user="user", password="password"
    )
    assert conn == mock_conn

@patch('src.database.psycopg2.connect')
@patch('src.database.st') # Mock Streamlit
def test_connect_postgres_failure(mock_st, mock_connect):
    mock_connect.side_effect = psycopg2.Error("Connection failed")
    conn = connect_postgres("host", "port", "dbname", "user", "password")
    assert conn is None
    mock_st.error.assert_called_once_with("Error connecting to PostgreSQL: Connection failed")

@patch('src.database.pd.read_sql_query')
def test_fetch_postgres_data_success(mock_read_sql):
    mock_conn = MagicMock()
    expected_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    mock_read_sql.return_value = expected_df
    query = "SELECT * FROM test_table"

    df = fetch_postgres_data(mock_conn, query)

    mock_read_sql.assert_called_once_with(query, mock_conn)
    pd.testing.assert_frame_equal(df, expected_df)

@patch('src.database.pd.read_sql_query')
@patch('src.database.st') # Mock Streamlit
def test_fetch_postgres_data_failure(mock_st, mock_read_sql):
    mock_conn = MagicMock()
    mock_read_sql.side_effect = Exception("Query failed")
    query = "SELECT * FROM test_table"

    df = fetch_postgres_data(mock_conn, query)

    assert df.empty
    mock_st.error.assert_called_once_with("Error fetching data from PostgreSQL: Query failed")

# --- Elasticsearch Tests ---
@patch('src.database.Elasticsearch')
@patch('src.database.st') # Mock Streamlit
def test_connect_elasticsearch_cloud_id_success(mock_st, mock_es_constructor):
    mock_es_instance = MagicMock()
    mock_es_instance.ping.return_value = True
    mock_es_constructor.return_value = mock_es_instance

    es_conn = connect_elasticsearch(cloud_id="test_cloud_id", api_key=("key_id", "api_key_val"))

    mock_es_constructor.assert_called_once_with(cloud_id="test_cloud_id", api_key=("key_id", "api_key_val"))
    assert es_conn == mock_es_instance
    mock_st.success.assert_called_once_with("Successfully connected to Elasticsearch!")

@patch('src.database.Elasticsearch')
@patch('src.database.st') # Mock Streamlit
def test_connect_elasticsearch_hosts_success(mock_st, mock_es_constructor):
    mock_es_instance = MagicMock()
    mock_es_instance.ping.return_value = True
    mock_es_constructor.return_value = mock_es_instance

    es_conn = connect_elasticsearch(hosts=["http://localhost:9200"])

    mock_es_constructor.assert_called_once_with(["http://localhost:9200"])
    assert es_conn == mock_es_instance
    mock_st.success.assert_called_once_with("Successfully connected to Elasticsearch!")

@patch('src.database.Elasticsearch')
@patch('src.database.st') # Mock Streamlit
def test_connect_elasticsearch_ping_false(mock_st, mock_es_constructor):
    mock_es_instance = MagicMock()
    mock_es_instance.ping.return_value = False
    mock_es_constructor.return_value = mock_es_instance

    es_conn = connect_elasticsearch(hosts=["http://localhost:9200"])

    assert es_conn is None
    mock_st.error.assert_called_once_with("Failed to ping Elasticsearch. Check connection details.")

@patch('src.database.Elasticsearch')
@patch('src.database.st') # Mock Streamlit
def test_connect_elasticsearch_connection_error(mock_st, mock_es_constructor):
    # Define a custom exception that derives from ESConnectionError
    # and has a specific __str__ representation for the test.
    class TestMockESConnectionError(ESConnectionError):
        def __init__(self, message="Custom Mocked ES Connection Error"):
            # Initialize the base ESConnectionError with args that won't cause IndexErrors
            # For ESConnectionError, just a message arg is fine.
            super().__init__(message)
            self._message = message # Store custom message if needed, super already stores it in args

        def __str__(self):
            return self._message

    # Use an instance of this custom exception as the side effect
    mock_es_constructor.side_effect = TestMockESConnectionError("Mocked ES Connection Error From Custom Class")

    es_conn = connect_elasticsearch(hosts=["http://localhost:9200"])

    assert es_conn is None
    expected_error_msg = "Error connecting to Elasticsearch: Mocked ES Connection Error From Custom Class"
    mock_st.error.assert_called_once_with(expected_error_msg)

@patch('src.database.st') # Mock Streamlit
def test_connect_elasticsearch_no_params(mock_st):
    es_conn = connect_elasticsearch()
    assert es_conn is None
    mock_st.error.assert_called_once_with("Elasticsearch connection requires either cloud_id & api_key or hosts.")


@patch('src.database.st') # Mock Streamlit
def test_fetch_elasticsearch_data_success(mock_st):
    mock_es_conn = MagicMock()
    mock_response = {
        'hits': {
            'hits': [
                {'_source': {'col1': 1, 'col2': 'a'}},
                {'_source': {'col1': 2, 'col2': 'b'}}
            ]
        }
    }
    mock_es_conn.search.return_value = mock_response
    expected_df = pd.DataFrame([
        {'col1': 1, 'col2': 'a'},
        {'col1': 2, 'col2': 'b'}
    ])

    df = fetch_elasticsearch_data(mock_es_conn, "test_index")

    mock_es_conn.search.assert_called_once_with(index="test_index", body={"query": {"match_all": {}}}, size=1000)
    pd.testing.assert_frame_equal(df, expected_df)

@patch('src.database.st') # Mock Streamlit
def test_fetch_elasticsearch_data_no_hits(mock_st):
    mock_es_conn = MagicMock()
    mock_response = {'hits': {'hits': []}}
    mock_es_conn.search.return_value = mock_response

    df = fetch_elasticsearch_data(mock_es_conn, "test_index")

    assert df.empty
    mock_st.warning.assert_called_once_with("No documents found in index 'test_index' for the given query.")

@patch('src.database.st') # Mock Streamlit
def test_fetch_elasticsearch_data_search_error(mock_st):
    mock_es_conn = MagicMock()
    mock_es_conn.search.side_effect = Exception("Search failed")

    df = fetch_elasticsearch_data(mock_es_conn, "test_index")

    assert df.empty
    mock_st.error.assert_called_once_with("Error fetching data from Elasticsearch index 'test_index': Search failed")

def test_fetch_elasticsearch_data_no_connection():
    df = fetch_elasticsearch_data(None, "test_index")
    assert df.empty
