import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.app_logic import handle_postgres_connection, handle_event_data_upload, handle_time_series_settings, handle_single_time_series_selection, handle_population_analysis_settings, handle_feature_engineering_settings, handle_save_load_app_state

@pytest.fixture
def mock_st(mocker):
    """A mock for the streamlit module."""
    mock_st = MagicMock()
    mock_st.session_state = MagicMock()
    return mock_st

def test_handle_postgres_connection(mock_st):
    with patch('src.app_logic.st', mock_st):
        db_conn = MagicMock()

        with patch('src.app_logic.get_schemas_postgres') as mock_get_schemas, \
             patch('src.app_logic.get_tables_postgres') as mock_get_tables, \
             patch('src.app_logic.fetch_data_postgres') as mock_fetch_data:

                mock_get_schemas.return_value = (['schema1', 'schema2'], None)
                mock_get_tables.return_value = (['table1', 'table2'], None)
                mock_fetch_data.return_value = (pd.DataFrame({'a': [1]}), None)

                mock_st.sidebar.selectbox.side_effect = ['schema1', 'table1']
                mock_st.sidebar.button.return_value = True

                handle_postgres_connection(db_conn)

                mock_get_schemas.assert_called_once_with(db_conn)
                mock_get_tables.assert_called_once_with(db_conn, 'schema1')
                mock_fetch_data.assert_called_once_with(db_conn, 'schema1', 'table1')

                assert not mock_st.session_state.data_df_original.empty
                assert not mock_st.session_state.data_df.empty
                mock_st.sidebar.success.assert_called_once_with("Data fetched successfully!")

def test_handle_event_data_upload(mock_st):
    with patch('src.app_logic.st', mock_st):
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        mock_st.sidebar.file_uploader.return_value = mock_file

        with patch('src.app_logic.pd.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'device_id': ['a'],
                'timestamp': [pd.to_datetime('2023-01-01')],
                'event_type': ['error']
            })

            mock_st.session_state.time_series_specs = {
                'event_device_id_col': 'device_id',
                'event_timestamp_col': 'timestamp',
                'event_event_type_col': 'event_type'
            }

            handle_event_data_upload()

            mock_read_csv.assert_called_once_with(mock_file)
            assert not mock_st.session_state.event_df.empty
            mock_st.sidebar.success.assert_called_once()

def test_handle_time_series_settings(mock_st):
    with patch('src.app_logic.st', mock_st):
        mock_st.session_state.data_df_original = pd.DataFrame({
            'device_id': ['a', 'b'],
            'timestamp': [pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-02')],
            'value1': [1, 2],
            'value2': [3, 4]
        })
        mock_st.session_state.time_series_specs = {}

        mock_st.sidebar.multiselect.side_effect = [['device_id'], ['value1', 'value2']]
        mock_st.sidebar.selectbox.return_value = 'timestamp'

        handle_time_series_settings()

        assert mock_st.session_state.time_series_specs['id_cols'] == ['device_id']
        assert mock_st.session_state.time_series_specs['timestamp_col'] == 'timestamp'
        assert mock_st.session_state.time_series_specs['value_cols'] == ['value1', 'value2']

def test_handle_single_time_series_selection(mock_st):
    with patch('src.app_logic.st', mock_st):
        mock_st.session_state.data_df_original = pd.DataFrame({
            'device_id': ['a', 'b'],
            'timestamp': [pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-02')],
            'value1': [1, 2],
        })
        mock_st.session_state.time_series_specs = {
            'id_cols': ['device_id'],
            'timestamp_col': 'timestamp',
            'value_cols': ['value1']
        }

        mock_st.sidebar.selectbox.side_effect = ['a', 'value1']
        mock_st.sidebar.button.return_value = True

        handle_single_time_series_selection()

        assert not mock_st.session_state.time_series_specs['processed_series'].empty
        mock_st.sidebar.success.assert_called_once()

def test_handle_population_analysis_settings(mock_st):
    with patch('src.app_logic.st', mock_st):
        mock_st.session_state.data_df_original = pd.DataFrame({
            'device_id': ['a', 'b'],
            'timestamp': [pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-02')],
            'value1': [1, 2],
        })
        mock_st.session_state.time_series_specs = {
            'timestamp_col': 'timestamp',
            'selected_value_col_for_analysis': 'value1'
        }
        mock_st.session_state.widget_fe_acf_lags = "1,5,10"
        mock_st.session_state.widget_fe_rolling_windows = "1,5,10,20"

        mock_st.sidebar.button.return_value = True

        with patch('src.app_logic.concurrent.futures.ProcessPoolExecutor') as mock_executor:
            handle_population_analysis_settings()
            mock_executor.assert_called_once()

def test_handle_feature_engineering_settings(mock_st):
    with patch('src.app_logic.st', mock_st):
        mock_st.session_state.fe_acf_lags_general = [1, 5, 10]
        mock_st.session_state.fe_rolling_windows_general = [1, 5, 10, 20]

        handle_feature_engineering_settings()

        mock_st.sidebar.text_input.assert_any_call(
            "ACF Lags (comma-separated integers)",
            value="1,5,10",
            key="acf_lags_input_widget"
        )
        mock_st.sidebar.text_input.assert_any_call(
            "Rolling Windows (comma-separated integers)",
            value="1,5,10,20",
            key="rolling_windows_input_widget"
        )

def test_handle_save_load_app_state_save(mock_st):
    with patch('src.app_logic.st', mock_st):
        with patch('src.app_logic.gather_settings_for_save') as mock_gather, \
             patch('src.app_logic.json.dumps') as mock_dumps:

            mock_st.sidebar.button.return_value = True
            mock_gather.return_value = {'setting': 'value'}
            mock_dumps.return_value = '{"setting": "value"}'

            handle_save_load_app_state()

            mock_gather.assert_called_once()
            mock_dumps.assert_called_once_with({'setting': 'value'}, indent=4)
            mock_st.sidebar.download_button.assert_called_once()

def test_handle_save_load_app_state_load(mock_st):
    with patch('src.app_logic.st', mock_st):
        mock_file = MagicMock()
        mock_st.sidebar.file_uploader.return_value = mock_file

        with patch('src.app_logic.json.load') as mock_load, \
             patch('src.app_logic.apply_loaded_settings_to_session_state') as mock_apply:

            mock_load.return_value = {'setting': 'value'}
            mock_apply.return_value = (True, "Success")

            handle_save_load_app_state()

            mock_load.assert_called_once_with(mock_file)
            mock_apply.assert_called_once_with({'setting': 'value'})
            mock_st.sidebar.success.assert_called_once()
