import pytest
from unittest.mock import patch
from src.config_utils import (
    gather_settings_for_save,
    apply_loaded_settings_to_session_state,
    _get_nested_session_state,
    _set_nested_session_state,
)


# A custom class to mock st.session_state more accurately
class SessionStateMock:
    def __init__(self, initial_state=None):
        if initial_state is None:
            initial_state = {}
        self.__dict__["_state"] = initial_state

    def __getattr__(self, name):
        return self._state.get(name)

    def __setattr__(self, name, value):
        self._state[name] = value

    def __delattr__(self, name):
        if name in self._state:
            del self._state[name]

    def __getitem__(self, name):
        return self._state[name]

    def __setitem__(self, name, value):
        self._state[name] = value

    def __delitem__(self, name):
        del self._state[name]

    def __contains__(self, name):
        return name in self._state

    def get(self, name, default=None):
        return self._state.get(name, default)


@pytest.fixture
def mock_st_session_state():
    """Provides a robust mock for st.session_state."""
    initial_state = {
        "time_series_specs": {},
    }
    return SessionStateMock(initial_state)


def test_gather_settings_full(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        mock_st_session_state.time_series_specs = {
            "id_cols": ["device_id"],
            "timestamp_col": "time_col",
            "value_cols": ["val1", "val2"],
            "selected_id": "dev1",
            "selected_value_col_for_analysis": "val1",
        }
        mock_st_session_state.if_contam_general = 0.05
        mock_st_session_state.ocsvm_nu_general = 0.02
        mock_st_session_state.clustering_module_params = {
            "method": "K-Means",
            "n_clusters": 3,
        }
        mock_st_session_state.fe_acf_lags_general = [1, 2, 3]
        mock_st_session_state.fe_rolling_windows_general = [10, 20]

        settings = gather_settings_for_save()

        assert settings["time_series_settings"]["id_cols"] == ["device_id"]
        assert (
            settings["anomaly_detection_settings"]["IsolationForest"]["contamination"]
            == 0.05
        )
        assert settings["clustering_module_params"]["n_clusters"] == 3
        assert "clustering_settings" not in settings
        assert settings["feature_engineering_settings"]["acf_lags"] == [1, 2, 3]


def test_gather_settings_defaults(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        settings = gather_settings_for_save()
        assert settings["time_series_settings"]["id_cols"] == []
        assert settings["feature_engineering_settings"]["acf_lags"] == [1, 5, 10]


def test_apply_settings_full(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        sample_settings = {
            "time_series_settings": {"id_cols": ["new_id"], "timestamp_col": "new_ts"},
            "anomaly_detection_settings": {"IsolationForest": {"contamination": 0.11}},
            "clustering_module_params": {"method": "DBSCAN", "eps": 0.5},
            "feature_engineering_settings": {
                "acf_lags": [5, 10],
                "rolling_windows": [3, 6],
            },
        }

        success, msg = apply_loaded_settings_to_session_state(sample_settings)

        assert success
        assert mock_st_session_state.time_series_specs["id_cols"] == ["new_id"]
        assert mock_st_session_state.if_contam_general == 0.11
        assert mock_st_session_state.clustering_module_params["eps"] == 0.5
        assert mock_st_session_state.fe_acf_lags_general == [5, 10]


def test_apply_settings_handles_none_values(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        mock_st_session_state.if_contam_general = 0.2
        settings_with_none = {
            "anomaly_detection_settings": {"IsolationForest": {"contamination": None}}
        }
        success, _ = apply_loaded_settings_to_session_state(settings_with_none)
        assert success
        assert mock_st_session_state.if_contam_general == 0.2


def test_apply_settings_creates_clustering_params_dict(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        if "clustering_module_params" in mock_st_session_state:
            del mock_st_session_state.clustering_module_params

        settings = {"clustering_module_params": {"method": "K-Means"}}
        apply_loaded_settings_to_session_state(settings)

        assert "clustering_module_params" in mock_st_session_state
        assert mock_st_session_state.clustering_module_params["method"] == "K-Means"


def test_apply_settings_error_handling(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        settings_bad_type = {"feature_engineering_settings": {"acf_lags": "not-a-list"}}
        success, msg = apply_loaded_settings_to_session_state(settings_bad_type)
        assert not success
        assert "Invalid type for fe_acf_lags_general" in msg

        with patch(
            "src.config_utils._set_nested_session_state",
            side_effect=Exception("mock error"),
        ):
            settings = {"time_series_settings": {"id_cols": ["any"]}}
            success, msg = apply_loaded_settings_to_session_state(settings)
            assert not success
            assert "Error applying loaded settings: mock error" in msg


def test_nested_session_state_helpers(mock_st_session_state):
    with patch("src.config_utils.st.session_state", mock_st_session_state):
        assert _set_nested_session_state("a.b.c", 100)
        assert mock_st_session_state.a["b"]["c"] == 100
        assert _get_nested_session_state("a.b.c") == 100
        assert _get_nested_session_state("x.y.z", "default") == "default"

        mock_st_session_state.a["b"] = "not-a-dict"
        assert not _set_nested_session_state("a.b.c.d", 200)
        assert _get_nested_session_state("a.b.c.d") is None
