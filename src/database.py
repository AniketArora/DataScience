import psycopg2
import pandas as pd
import streamlit as st
import logging
from typing import Optional, List, Tuple, Union # Added for type hinting

logger = logging.getLogger(__name__)

@st.cache_resource(max_entries=5)
def connect_postgres(db_config: dict) -> Tuple[Optional[psycopg2.extensions.connection], Optional[str]]:
    """
    Connects to a PostgreSQL database.

    Args:
        db_config: A dictionary with connection parameters (host, port, dbname, user, password).

    Returns:
        A tuple containing the connection object and None on success,
        or None and an error message on failure.
    """
    try:
        conn = psycopg2.connect(**db_config)
        return conn, None
    except psycopg2.Error as e:
        logger.error("Error connecting to PostgreSQL: %s", e, exc_info=True)
        return None, f"Error connecting to PostgreSQL: {e}"

@st.cache_data(max_entries=10)
def get_schemas_postgres(conn: psycopg2.extensions.connection) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Retrieves a list of schema names from the PostgreSQL database.

    Args:
        conn: A psycopg2 connection object.

    Returns:
        A tuple containing a list of schema names and None on success,
        or None and an error message on failure.
    """
    if not conn:
        return None, "No database connection provided."
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT schema_name FROM information_schema.schemata;")
            schemas = [row[0] for row in cur.fetchall()]
        return schemas, None
    except psycopg2.Error as e:
        logger.error("Error retrieving schemas: %s", e, exc_info=True)
        return None, f"Error retrieving schemas: {e}"

@st.cache_data(max_entries=10)
def get_tables_postgres(conn: psycopg2.extensions.connection, schema_name: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Retrieves a list of table names from a specific schema in the PostgreSQL database.

    Args:
        conn: A psycopg2 connection object.
        schema_name: The name of the schema.

    Returns:
        A tuple containing a list of table names and None on success,
        or None and an error message on failure.
    """
    if not conn:
        return None, "No database connection provided."
    if not schema_name or not isinstance(schema_name, str):
        return None, "Invalid schema name provided."
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema_name,),
            )
            tables = [row[0] for row in cur.fetchall()]
        return tables, None
    except psycopg2.Error as e:
        logger.error("Error retrieving tables for schema '%s': %s", schema_name, e, exc_info=True)
        return None, f"Error retrieving tables for schema '{schema_name}': {e}"

@st.cache_data(max_entries=10)
def fetch_data_postgres(
    conn: psycopg2.extensions.connection, schema_name: str, table_name: str, limit: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetches data from a table in a specific schema in the PostgreSQL database.

    Args:
        conn: A psycopg2 connection object.
        schema_name: The name of the schema.
        table_name: The name of the table.
        limit: Optional limit for the number of rows to fetch.

    Returns:
        A tuple containing a Pandas DataFrame with the table data and None on success,
        or None and an error message on failure.
    """
    if not conn:
        return None, "No database connection provided."
    if not schema_name or not isinstance(schema_name, str):
        return None, "Invalid schema name provided."
    if not table_name or not isinstance(table_name, str):
        return None, "Invalid table name provided."

    try:
        with conn.cursor() as cur:
            query = f'SELECT * FROM "{schema_name}"."{table_name}"'
            if limit is not None:
                if not isinstance(limit, int) or limit < 0:
                    return None, "Invalid limit provided. Limit must be a non-negative integer."
                query += f" LIMIT {limit}"
            query += ";"
            cur.execute(query)
            colnames = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            df = pd.DataFrame(data, columns=colnames)
        return df, None
    except psycopg2.Error as e:
        logger.error("Error fetching data from table '%s.%s': %s", schema_name, table_name, e, exc_info=True)
        return None, f"Error fetching data from table '{schema_name}.{table_name}': {e}"
