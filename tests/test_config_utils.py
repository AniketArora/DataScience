import pytest
from unittest.mock import MagicMock, patch

from src.config_utils import (
    gather_settings_for_save,
    apply_loaded_settings_to_session_state,
    TS_SETTINGS_KEYS,
    IFOREST_PARAMS_KEYS,
    OCSVM_PARAMS_KEYS,
    KMEANS_PARAMS_KEYS,
    DBSCAN_PARAMS_KEYS
)

@pytest.fixture
def mock_st_session_state(mocker):
    """A mock for st.session_state that can handle nested attribute and dict-like access."""

    class SessionStateMock(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.time_series_specs = {}

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __contains__(self, key):
            return hasattr(self, key)

    state = SessionStateMock()

    def get_side_effect(key, default=None):
        return getattr(state, key, default)

    state.get.side_effect = get_side_effect

    # Initialize other top-level attributes that are directly get/set in session_state
    direct_access_keys = []
    for key_dict in [IFOREST_PARAMS_KEYS, OCSVM_PARAMS_KEYS, KMEANS_PARAMS_KEYS, DBSCAN_PARAMS_KEYS]:
        direct_access_keys.extend(key_dict.values())

    for key in set(direct_access_keys):
        setattr(state, key, None)

    return state

def test_gather_settings(mock_st_session_state):
    with patch('src.config_utils.st.session_state', mock_st_session_state):
        # Populate mock session state
        mock_st_session_state.time_series_specs['id_cols'] = ['device_id']
        mock_st_session_state.time_series_specs['timestamp_col'] = 'time_col'
        mock_st_session_state.time_series_specs['value_cols'] = ['val1', 'val2']
        mock_st_session_state.time_series_specs['selected_id'] = 'dev1'
        mock_st_session_state.time_series_specs['selected_value_col_for_analysis'] = 'val1'

        mock_st_session_state.if_contam_general = 0.05
        mock_st_session_state.ocsvm_nu_general = 0.02
        mock_st_session_state.ocsvm_kernel_general = "linear"
        mock_st_session_state.ocsvm_gamma_general = "auto"

        # Mocking for the new clustering_module_params structure
        mock_st_session_state.clustering_module_params = {
            'KMeans': {'k_final': 4, 'scale_data': False},
            'DBSCAN': {'eps': 0.7, 'scale_data': True}
        }


        settings = gather_settings_for_save()

        assert settings['time_series_settings']['id_cols'] == ['device_id']
        assert settings['time_series_settings']['timestamp_col'] == 'time_col'
        assert settings['time_series_settings']['selected_id'] == 'dev1'
        assert settings['anomaly_detection_settings']['IsolationForest']['contamination'] == 0.05
        assert settings['anomaly_detection_settings']['OneClassSVM']['kernel'] == "linear"

        # Test the new clustering structure
        assert settings['clustering_module_params']['KMeans']['k_final'] == 4
        assert settings['clustering_module_params']['KMeans']['scale_data'] is False
        assert settings['clustering_module_params']['DBSCAN']['eps'] == 0.7
        assert settings['clustering_module_params']['DBSCAN']['scale_data'] is True


def test_apply_settings(mock_st_session_state):
    with patch('src.config_utils.st.session_state', mock_st_session_state):
        sample_settings = {
            "time_series_settings": {"id_cols": ["new_id"], "timestamp_col": "new_ts", "selected_id": "d100"},
            "anomaly_detection_settings": {"IsolationForest": {"contamination": 0.11}},
            "clustering_module_params": {
                "KMeans": {"k_final": 5, "scale_data": False},
                "DBSCAN": {"eps": 0.75, "scale_data": True}
            }
        }

        # Initialize clustering_module_params in the mock state
        mock_st_session_state.clustering_module_params = {}

        success, msg = apply_loaded_settings_to_session_state(sample_settings)

        assert success
        assert mock_st_session_state.time_series_specs['id_cols'] == ["new_id"]
        assert mock_st_session_state.time_series_specs['selected_id'] == "d100"
        assert mock_st_session_state.if_contam_general == 0.11
        assert mock_st_session_state.clustering_module_params['KMeans']['k_final'] == 5
        assert mock_st_session_state.clustering_module_params['KMeans']['scale_data'] is False
        assert mock_st_session_state.clustering_module_params['DBSCAN']['eps'] == 0.75
        assert mock_st_session_state.clustering_module_params['DBSCAN']['scale_data'] is True


def test_apply_settings_missing_keys(mock_st_session_state):
    with patch('src.config_utils.st.session_state', mock_st_session_state):
        # Pre-populate some state
        mock_st_session_state.time_series_specs['id_cols'] = ['old_id']
        mock_st_session_state.if_contam_general = 0.01

        partial_settings = {
            "time_series_settings": {"timestamp_col": "partial_ts_col"}
            # Other settings like anomaly_detection or clustering are missing
        }
        success, _ = apply_loaded_settings_to_session_state(partial_settings)
        assert success
        assert mock_st_session_state.time_series_specs['timestamp_col'] == "partial_ts_col"
        # Assert that existing unrelated state is not wiped if not in loaded settings
        assert mock_st_session_state.time_series_specs['id_cols'] == ['old_id']
        assert mock_st_session_state.if_contam_general == 0.01 # Should remain unchanged

def test_apply_settings_handles_none_values(mock_st_session_state):
    with patch('src.config_utils.st.session_state', mock_st_session_state):
        mock_st_session_state.if_contam_general = 0.2 # Initial value
        settings_with_none = {
            "anomaly_detection_settings": {"IsolationForest": {"contamination": None}}
        }
        # apply_loaded_settings_to_session_state checks for `is not None` before applying
        success, _ = apply_loaded_settings_to_session_state(settings_with_none)
        assert success
        assert mock_st_session_state.if_contam_general == 0.2 # Should not change if loaded value is None

# --- New tests for error handling ---
def test_get_nested_session_state_exception(mock_st_session_state):
    from src.config_utils import _get_nested_session_state
    with patch('src.config_utils.st.session_state', mock_st_session_state):
        # Configure the mock to raise an error on access
        mock_st_session_state.__getitem__.side_effect = KeyError("Test Key Error")
        result = _get_nested_session_state('a.b.c')
        assert result is None

def test_apply_settings_invalid_type(mock_st_session_state):
    from src.config_utils import apply_loaded_settings_to_session_state
    settings = {"feature_engineering_settings": {"acf_lags": "not a list"}}
    success, msg = apply_loaded_settings_to_session_state(settings)
    assert not success
    assert "Invalid type" in msg

def test_apply_settings_main_exception(mock_st_session_state):
    from src.config_utils import apply_loaded_settings_to_session_state
    with patch('src.config_utils._set_nested_session_state', side_effect=Exception("Test Exception")):
        success, msg = apply_loaded_settings_to_session_state({"time_series_settings": {"id_cols": ["new_id"]}})
        assert not success
        assert "Test Exception" in msg
