import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import psycopg2  # Import for psycopg2.Error

# Assuming your database.py file is in src directory and src is in PYTHONPATH
from src.database import (
    connect_postgres,
    get_schemas_postgres,
    get_tables_postgres,
    fetch_data_postgres,
)


class TestDatabaseUtils(unittest.TestCase):
    def setUp(self):
        # Clear caches for functions that use Streamlit caching before each test
        # to ensure mocks are hit and not cached results.
        connect_postgres.clear()
        # If other functions were also causing issues, clear them here too:
        # get_schemas_postgres.clear()
        # get_tables_postgres.clear()
        # fetch_data_postgres.clear()

    def test_connect_postgres_success(self):
        connect_postgres.clear()  # Explicit clear for this specific test too, just in case
        with patch("src.database.psycopg2.connect") as mock_psycopg2_connect:
            mock_conn_obj = MagicMock()
            mock_psycopg2_connect.return_value = mock_conn_obj
            db_config = {
                "host": "localhost",
                "port": "5432",
                "dbname": "testdb",
                "user": "testuser",
                "password": "testpassword",
            }

            conn, err = connect_postgres(db_config)

            mock_psycopg2_connect.assert_called_once_with(**db_config)
            self.assertEqual(conn, mock_conn_obj)
            self.assertIsNone(err)

    def test_connect_postgres_failure(self):
        connect_postgres.clear()  # Explicit clear for this specific test too
        with patch("src.database.psycopg2.connect") as mock_psycopg2_connect:
            mock_psycopg2_connect.side_effect = psycopg2.Error("Connection failed")
            db_config = {
                "host": "localhost",
                "port": "5432",
                "dbname": "testdb",
                "user": "testuser",
                "password": "testpassword",
            }

            conn, err = connect_postgres(db_config)

            mock_psycopg2_connect.assert_called_once_with(**db_config)
            self.assertIsNone(conn)
            self.assertTrue("Error connecting to PostgreSQL: Connection failed" in err)

    def test_get_schemas_success(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("public",),
            ("information_schema",),
            ("custom_schema",),
        ]

        schemas, err = get_schemas_postgres(mock_conn)

        mock_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with(
            "SELECT schema_name FROM information_schema.schemata;"
        )
        mock_cursor.fetchall.assert_called_once()
        self.assertEqual(schemas, ["public", "information_schema", "custom_schema"])
        self.assertIsNone(err)

    def test_get_schemas_failure(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = psycopg2.Error("Failed to get schemas")

        schemas, err = get_schemas_postgres(mock_conn)

        self.assertIsNone(schemas)
        self.assertTrue("Error retrieving schemas: Failed to get schemas" in err)

    def test_get_schemas_no_connection(self):
        schemas, err = get_schemas_postgres(None)
        self.assertIsNone(schemas)
        self.assertEqual(err, "No database connection provided.")

    def test_get_tables_success(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("table1",), ("table2",)]
        schema_name = "public"

        tables, err = get_tables_postgres(mock_conn, schema_name)

        mock_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
            (schema_name,),
        )
        mock_cursor.fetchall.assert_called_once()
        self.assertEqual(tables, ["table1", "table2"])
        self.assertIsNone(err)

    def test_get_tables_failure(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        schema_name = "public"
        mock_cursor.execute.side_effect = psycopg2.Error("Failed to get tables")

        tables, err = get_tables_postgres(mock_conn, schema_name)

        self.assertIsNone(tables)
        self.assertTrue(
            f"Error retrieving tables for schema '{schema_name}': Failed to get tables"
            in err
        )

    def test_get_tables_no_connection(self):
        tables, err = get_tables_postgres(None, "public")
        self.assertIsNone(tables)
        self.assertEqual(err, "No database connection provided.")

    def test_get_tables_invalid_schema_name(self):
        mock_conn = (
            MagicMock()
        )  # Connection object is needed for the check to pass to schema name validation
        tables, err = get_tables_postgres(mock_conn, None)  # Invalid schema name
        self.assertIsNone(tables)
        self.assertEqual(err, "Invalid schema name provided.")

        tables, err = get_tables_postgres(mock_conn, "")  # Invalid schema name
        self.assertIsNone(tables)
        self.assertEqual(err, "Invalid schema name provided.")

    def test_fetch_data_success(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.description = [("col1", None), ("col2", None)]
        mock_cursor.fetchall.return_value = [(1, "data1"), (2, "data2")]
        schema_name = "public"
        table_name = "test_table"

        df, err = fetch_data_postgres(mock_conn, schema_name, table_name)

        mock_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once()
        # expected_query = f'SELECT * FROM "{schema_name}"."{table_name}";'
        # mock_cursor.execute.assert_called_once_with(expected_query)
        mock_cursor.fetchall.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["col1", "col2"])
        pd.testing.assert_frame_equal(
            df, pd.DataFrame([(1, "data1"), (2, "data2")], columns=["col1", "col2"])
        )
        self.assertIsNone(err)

    def test_fetch_data_with_limit(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.description = [("col1", None)]
        mock_cursor.fetchall.return_value = [(1,)]
        schema_name = "public"
        table_name = "test_table"
        limit = 100

        df, err = fetch_data_postgres(mock_conn, schema_name, table_name, limit=limit)

        # expected_query = f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT {limit};'
        mock_cursor.execute.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNone(err)

    def test_fetch_data_failure(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        schema_name = "public"
        table_name = "test_table"
        mock_cursor.execute.side_effect = psycopg2.Error("Fetch failed")

        df, err = fetch_data_postgres(mock_conn, schema_name, table_name)

        self.assertIsNone(df)
        self.assertTrue(
            f"Error fetching data from table '{schema_name}.{table_name}': Fetch failed"
            in err
        )

    def test_fetch_data_no_data(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.description = [
            ("col1", None),
            ("col2", None),
        ]  # Column names are still returned
        mock_cursor.fetchall.return_value = []  # No data rows
        schema_name = "public"
        table_name = "test_table"

        df, err = fetch_data_postgres(mock_conn, schema_name, table_name)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["col1", "col2"])
        self.assertIsNone(err)

    def test_fetch_data_no_connection(self):
        df, err = fetch_data_postgres(None, "public", "test_table")
        self.assertIsNone(df)
        self.assertEqual(err, "No database connection provided.")

    def test_fetch_data_invalid_schema_or_table_name(self):
        mock_conn = MagicMock()
        df, err = fetch_data_postgres(mock_conn, None, "test_table")
        self.assertIsNone(df)
        self.assertEqual(err, "Invalid schema name provided.")

        df, err = fetch_data_postgres(mock_conn, "public", "")
        self.assertIsNone(df)
        self.assertEqual(err, "Invalid table name provided.")

    def test_fetch_data_invalid_limit(self):
        mock_conn = MagicMock()
        df, err = fetch_data_postgres(
            mock_conn, "public", "test_table", limit="invalid"
        )
        self.assertIsNone(df)
        self.assertEqual(
            err, "Invalid limit provided. Limit must be a non-negative integer."
        )

        df, err = fetch_data_postgres(mock_conn, "public", "test_table", limit=-1)
        self.assertIsNone(df)
        self.assertEqual(
            err, "Invalid limit provided. Limit must be a non-negative integer."
        )


if __name__ == "__main__":
    unittest.main()
