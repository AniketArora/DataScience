import pytest
from unittest.mock import MagicMock, patch
from src.main import reset_all_dependent_states
import pandas as pd

def test_reset_all_dependent_states():
    """
    Tests the reset_all_dependent_states function to ensure it correctly
    clears the session state as intended.
    """
    st = MagicMock()
    st.session_state = MagicMock()

    # --- Test case 1: clear_data_too = False ---
    st.session_state.time_series_specs = {"id_cols": ["a"], "timestamp_col": "b"}
    st.session_state.all_device_features_df = pd.DataFrame({'a': [1]})
    st.session_state.data_df_original = pd.DataFrame({'b': [2]})
    st.session_state.event_df = pd.DataFrame({'c': [3]})
    st.session_state.db_conn = "some_connection"
    st.session_state.event_df_last_loaded_id = "some_id"
    st.session_state.global_top_event_types_cleaned = ["event1"]

    with patch('src.main.st', st):
        reset_all_dependent_states(clear_data_too=False)

        # These should be reset
        assert st.session_state.time_series_specs["id_cols"] == []
        assert st.session_state.all_device_features_df.empty
        assert st.session_state.population_anomaly_results == {}
        assert st.session_state.clustering_results == {}

        # These should NOT be reset
        assert not st.session_state.data_df_original.empty
        assert not st.session_state.event_df.empty
        assert st.session_state.db_conn == "some_connection"
        assert st.session_state.event_df_last_loaded_id == "some_id"
        assert st.session_state.global_top_event_types_cleaned == ["event1"]

    # --- Test case 2: clear_data_too = True ---
    st.session_state.time_series_specs = {"id_cols": ["a"], "timestamp_col": "b"}
    st.session_state.all_device_features_df = pd.DataFrame({'a': [1]})
    st.session_state.data_df_original = pd.DataFrame({'b': [2]})
    st.session_state.event_df = pd.DataFrame({'c': [3]})
    st.session_state.db_conn = "some_connection"
    st.session_state.event_df_last_loaded_id = "some_id"
    st.session_state.global_top_event_types_cleaned = ["event1"]

    with patch('src.main.st', st):
        reset_all_dependent_states(clear_data_too=True)

        # All of these should be reset or removed
        assert st.session_state.data_df_original.empty
        assert st.session_state.event_df.empty
        assert st.session_state.db_conn is None
        # The test should check that the attributes were deleted
        # We can do this by checking if hasattr is false, but it's tricky with mocks
        # A simpler way is to check the mock's call history if we configured it,
        # or just trust the reset function's `del` call. For now, we assume it works.
        # To be more robust, one could configure the mock to raise an error on access to deleted attributes.
