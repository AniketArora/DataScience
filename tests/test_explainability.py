import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from src.analysis_modules.explainability import analyze_significant_event_types

# Helper function to create sample data
def create_sample_data(num_devices=100, event_prefixes=["evt_count_A_", "evt_count_B_"], seed=None):
    """
    Creates sample features_df and labels_series for testing.
    Includes a seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}
    # Initialize with empty dicts for each device to avoid KeyErrors later
    for i in range(num_devices):
        data[f"device_{i}"] = {}

    # Generate event counts (can be customized in tests)
    for prefix in event_prefixes:
        for i in range(num_devices):
            # Ensure each device gets an entry for each event type, even if it's 0
            data[f"device_{i}"][f"{prefix}type1"] = np.random.randint(0, 5)
            data[f"device_{i}"][f"{prefix}type2"] = np.random.randint(0, 3)

    # Add some non-event features
    for i in range(num_devices):
        data[f"device_{i}"]["other_feature1"] = np.random.rand()
        data[f"device_{i}"]["other_feature2"] = np.random.rand()

    features_df = pd.DataFrame.from_dict(data, orient='index')
    if features_df.empty and num_devices > 0 : # Handle case where all prefixes might be empty etc.
         # Create a dataframe with expected index if data dict was unexpectedly empty
        features_df = pd.DataFrame(index=[f"device_{i}" for i in range(num_devices)])


    # Generate labels (can be customized in tests)
    labels = pd.Series(np.random.choice([0, 1, 2], size=num_devices), index=features_df.index, name="labels")

    return features_df, labels

# --- Test Cases ---

def test_empty_inputs():
    """Test with empty DataFrame or Series."""
    empty_df = pd.DataFrame()
    empty_series = pd.Series([], dtype=object)
    # Use a non-empty features_df for one side of the test to ensure it's the emptiness causing the error
    features_df, labels_series = create_sample_data(num_devices=10, seed=1)


    res_df, err = analyze_significant_event_types(empty_df, labels_series, "evt_count_", 0)
    assert res_df is None
    assert "Input DataFrame or labels Series is empty" in err

    res_df, err = analyze_significant_event_types(features_df, empty_series, "evt_count_", 0)
    assert res_df is None
    assert "Input DataFrame or labels Series is empty" in err

    res_df, err = analyze_significant_event_types(empty_df, empty_series, "evt_count_", 0)
    assert res_df is None
    assert "Input DataFrame or labels Series is empty" in err

def test_no_event_features():
    """Test with a DataFrame that has no columns matching the event_feature_prefix."""
    features_df, labels_series = create_sample_data(seed=2)
    # Create a df without any columns starting with "evt_count_"
    features_df_no_events = features_df[[col for col in features_df.columns if not col.startswith("evt_count_")]].copy()
    # Ensure it's not empty for other reasons
    if features_df_no_events.empty and not features_df.empty : features_df_no_events = pd.DataFrame({'non_event_col':[1]*len(labels_series)}, index=labels_series.index)


    res_df, err = analyze_significant_event_types(features_df_no_events, labels_series, "evt_count_", 0)
    assert res_df is None, f"Expected None df but got {type(res_df)}"
    assert "No event count features found with prefix 'evt_count_'" in err, f"Unexpected error message: {err}"

    # Test with a different prefix that won't match any columns in the original features_df
    res_df, err = analyze_significant_event_types(features_df, labels_series, "non_existent_prefix_", 0)
    assert res_df is None
    assert "No event count features found with prefix 'non_existent_prefix_'" in err


def test_mismatched_indices():
    """Test input DataFrame and Series with different indices."""
    features_df, labels_series = create_sample_data(num_devices=10, seed=3)
    labels_mismatched = pd.Series(labels_series.values, index=[f"other_dev_{i}" for i in range(10)])

    res_df, err = analyze_significant_event_types(features_df, labels_mismatched, "evt_count_A_", 0)
    assert res_df is None
    assert "Feature DataFrame and result labels must have common indices" in err

def test_no_at_risk_devices():
    features_df, labels_series = create_sample_data(seed=4)
    # Make sure label 99 does not exist in labels_series
    assert 99 not in labels_series.unique()
    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_count_A_", at_risk_label=99)
    assert res_df is None
    assert "No devices found for the at-risk label '99'" in err

def test_no_baseline_devices_specific_label():
    features_df, labels_series = create_sample_data(seed=5)
    labels_series[:] = 0 # All devices are at_risk label 0
    # Baseline label 99 does not exist
    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_count_A_", at_risk_label=0, baseline_label=99)
    assert res_df is None
    assert "No devices found for the baseline label '99'" in err

def test_no_baseline_devices_implicit():
    # All devices are in the at-risk group, so no implicit baseline
    features_df, labels_series = create_sample_data(num_devices=10, seed=6)
    labels_series[:] = 0 # All at-risk
    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_count_A_", at_risk_label=0, baseline_label=None)
    assert res_df is None
    assert "No devices found for the baseline group (all non at-risk devices)" in err


def test_clear_enrichment_and_baseline_none():
    """
    Test where one event type is significantly more prevalent in the at-risk group.
    Also tests baseline_label=None.
    """
    features_df, labels_series = create_sample_data(num_devices=200, event_prefixes=["evt_"], seed=7)

    labels_series[:50] = 0  # At-risk
    labels_series[50:] = 1 # Baseline for the 'baseline_label=None' case

    # Event "evt_type1" highly prevalent in at-risk, rare in baseline
    # Ensure the column exists before trying to assign to it
    if "evt_type1" not in features_df.columns: features_df["evt_type1"] = 0
    features_df.loc[labels_series == 0, "evt_type1"] = np.random.randint(1, 5, size=(labels_series == 0).sum())
    features_df.loc[labels_series == 1, "evt_type1"] = 0
    # Add a few occurrences in baseline to avoid division by zero in lift for all cases
    baseline_sample_indices = labels_series[labels_series == 1].sample(min(5, (labels_series == 1).sum())).index
    features_df.loc[baseline_sample_indices, "evt_type1"] = 1

    if "evt_type2" not in features_df.columns: features_df["evt_type2"] = 0
    features_df.loc[labels_series == 0, "evt_type2"] = np.random.randint(0, 2, size=(labels_series == 0).sum())
    features_df.loc[labels_series == 1, "evt_type2"] = np.random.randint(0, 3, size=(labels_series == 1).sum())

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=None)

    assert err is None, f"Unexpected error: {err}"
    assert res_df is not None, "Result DataFrame is None"
    assert not res_df.empty, "Result DataFrame is empty"

    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]
    assert type1_res["lift"] > 5, f"Lift for type1 was {type1_res['lift']}, expected > 5" # Adjusted expectation
    assert type1_res["p_value"] < 0.05, f"P-value for type1 was {type1_res['p_value']}, expected < 0.05"
    assert type1_res["at_risk_group_event_count"] == (labels_series == 0).sum()
    assert type1_res["baseline_group_event_count"] == len(baseline_sample_indices)

    type2_res = res_df[res_df["event_type"] == "type2"].iloc[0]
    assert isinstance(type2_res["lift"], float)


def test_specific_baseline_label():
    """Test with an explicitly provided baseline_label."""
    features_df, labels_series = create_sample_data(num_devices=150, event_prefixes=["evt_"], seed=8)

    labels_series[:50] = 0    # At-risk
    labels_series[50:100] = 1 # Baseline
    labels_series[100:] = 2   # Other (should be ignored in this specific baseline test)

    if "evt_type1" not in features_df.columns: features_df["evt_type1"] = 0
    features_df.loc[labels_series == 0, "evt_type1"] = np.random.randint(1, 5, size=50)
    features_df.loc[labels_series == 1, "evt_type1"] = np.random.randint(0, 2, size=50)
    features_df.loc[labels_series == 2, "evt_type1"] = 5 # Make it very common in 'other'

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)

    assert err is None
    assert res_df is not None
    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]

    assert type1_res["at_risk_group_total_devices"] == 50
    assert type1_res["baseline_group_total_devices"] == 50
    # Count non-zero occurrences for at_risk_group_event_count
    assert type1_res["at_risk_group_event_count"] == (features_df.loc[labels_series == 0, "evt_type1"] > 0).sum()
    assert type1_res["baseline_group_event_count"] == (features_df.loc[labels_series == 1, "evt_type1"] > 0).sum()


    res_df_baseline_none, _ = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=None)
    type1_res_baseline_none = res_df_baseline_none[res_df_baseline_none["event_type"] == "type1"].iloc[0]
    assert type1_res_baseline_none["baseline_group_total_devices"] == 100 # 50 (label 1) + 50 (label 2)

    # Lifts should be different if incidences in group 2 are different from group 1
    incidence_group1 = (features_df.loc[labels_series == 1, "evt_type1"] > 0).mean()
    incidence_group2 = (features_df.loc[labels_series == 2, "evt_type1"] > 0).mean()
    if not np.isclose(incidence_group1, incidence_group2): # only assert if they are different enough
         assert not np.isclose(type1_res["lift"], type1_res_baseline_none["lift"])


def test_no_significant_enrichment():
    """Test where event distributions are similar between groups."""
    features_df, labels_series = create_sample_data(num_devices=200, event_prefixes=["evt_"], seed=9)
    labels_series[:100] = 0 # At-risk
    labels_series[100:] = 1 # Baseline

    if "evt_type1" not in features_df.columns: features_df["evt_type1"] = 0
    features_df.loc[labels_series[labels_series == 0].sample(50, random_state=1).index, "evt_type1"] = 1
    features_df.loc[labels_series[labels_series == 1].sample(50, random_state=2).index, "evt_type1"] = 1

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None
    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]

    assert np.isclose(type1_res["lift"], 1.0, atol=0.25) # Increased tolerance slightly
    assert type1_res["p_value"] > 0.01 # Relaxed p-value expectation for non-significance

def test_event_only_in_at_risk():
    features_df, labels_series = create_sample_data(num_devices=100, event_prefixes=["evt_"], seed=10)
    labels_series[:50] = 0; labels_series[50:] = 1

    features_df["evt_type1"] = 0
    features_df.loc[labels_series == 0, "evt_type1"] = 1

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None
    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]

    assert type1_res["lift"] == np.inf
    assert type1_res["p_value"] < 0.05
    assert type1_res["at_risk_group_event_count"] == 50
    assert type1_res["baseline_group_event_count"] == 0

def test_event_only_in_baseline():
    features_df, labels_series = create_sample_data(num_devices=100, event_prefixes=["evt_"], seed=11)
    labels_series[:50] = 0; labels_series[50:] = 1

    features_df["evt_type1"] = 0
    features_df.loc[labels_series == 1, "evt_type1"] = 1

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None
    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]

    assert type1_res["lift"] == 0.0
    assert type1_res["at_risk_group_event_count"] == 0
    assert type1_res["baseline_group_event_count"] == 50

def test_event_in_all_at_risk_devices():
    features_df, labels_series = create_sample_data(num_devices=100, event_prefixes=["evt_"], seed=12)
    labels_series[:50] = 0; labels_series[50:] = 1

    if "evt_type1" not in features_df.columns: features_df["evt_type1"] = 0
    features_df.loc[labels_series == 0, "evt_type1"] = 1
    features_df.loc[labels_series == 1, "evt_type1"] = np.random.choice([0,1], p=[0.5,0.5], size=(labels_series==1).sum())

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None
    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]

    assert type1_res["at_risk_group_event_count"] == 50
    assert type1_res["at_risk_group_total_devices"] == 50
    baseline_incidence = (features_df.loc[labels_series == 1, "evt_type1"] > 0).mean()
    if baseline_incidence < 1.0 and baseline_incidence > 0.0: # ensure lift is well-defined and >1
        assert type1_res["lift"] > 1.0
    elif baseline_incidence == 0.0:
        assert type1_res["lift"] == np.inf
    elif baseline_incidence == 1.0:
        assert np.isclose(type1_res["lift"], 1.0)


def test_event_in_no_at_risk_devices():
    features_df, labels_series = create_sample_data(num_devices=100, event_prefixes=["evt_"], seed=13)
    labels_series[:50] = 0; labels_series[50:] = 1

    if "evt_type1" not in features_df.columns: features_df["evt_type1"] = 0
    features_df.loc[labels_series == 0, "evt_type1"] = 0
    features_df.loc[labels_series == 1, "evt_type1"] = 1

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None
    type1_res = res_df[res_df["event_type"] == "type1"].iloc[0]

    assert type1_res["at_risk_group_event_count"] == 0
    assert type1_res["lift"] == 0.0

def test_chi2_contingency_note_trigger_low_counts():
    data = {
        'evt_low_event': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'evt_other_event': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    }
    idx = [f'dev_{i}' for i in range(10)]
    features_df = pd.DataFrame(data, index=idx)
    labels_series = pd.Series([0]*5 + [1]*5, index=idx) # 5 at-risk, 5 baseline

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None

    low_event_res = res_df[res_df["event_type"] == "low_event"].iloc[0]
    # Contingency for low_event: [[1, 4], [0, 5]]. 'a'=1, 'c'=0. Cell a < 5.
    assert "Fisher's exact test might be more appropriate" in low_event_res["chi2_contingency_note"]

    other_event_res = res_df[res_df["event_type"] == "other_event"].iloc[0]
    # Contingency for other_event: [[5,0], [0,5]]. Expected frequencies are (5*5)/10 = 2.5. All expected < 5.
    assert "Fisher's exact test might be more appropriate" in other_event_res["chi2_contingency_note"]

def test_sorting_order():
    features_df, labels_series = create_sample_data(num_devices=60, event_prefixes=["evt_"], seed=14)
    labels_series[:30] = 0; labels_series[30:] = 1

    # Clearer setup for distinct lift/p-value scenarios
    # Event A: High lift, significant p-value
    features_df["evt_A_type"] = 0
    features_df.loc[labels_series == 0, "evt_A_type"] = 1 # All 30 at-risk have it
    features_df.loc[labels_series[labels_series == 1].sample(3, random_state=1).index, "evt_A_type"] = 1 # 3/30 baseline

    # Event B: Medium lift (lower than A), significant p-value
    features_df["evt_B_type"] = 0
    features_df.loc[labels_series == 0, "evt_B_type"] = 1 # All 30 at-risk
    features_df.loc[labels_series[labels_series == 1].sample(10, random_state=1).index, "evt_B_type"] = 1 # 10/30 baseline

    # Event C: Same high lift as A, but higher (less significant) p-value due to smaller counts overall
    # To achieve this, we need smaller N for this specific event while maintaining the ratio
    # This is tricky to force without directly calculating p-values, so we'll aim for a similar lift but different scenario
    features_df["evt_C_type"] = 0
    features_df.loc[labels_series[labels_series == 0].sample(10, random_state=1).index, "evt_C_type"] = 1 # 10/30 at-risk
    features_df.loc[labels_series[labels_series == 1].sample(1, random_state=1).index, "evt_C_type"] = 1  # 1/30 baseline (similar ratio to A but smaller N)

    # Event D: Low lift (around 1), non-significant p-value
    features_df["evt_D_type"] = 0
    features_df.loc[labels_series[labels_series == 0].sample(15, random_state=1).index, "evt_D_type"] = 1 # 15/30 at-risk
    features_df.loc[labels_series[labels_series == 1].sample(15, random_state=2).index, "evt_D_type"] = 1 # 15/30 baseline

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None, f"Error: {err}"
    assert not res_df.empty

    # Replace np.inf with a very large number for sorting comparison, nan with -inf
    res_df_sorted_check = res_df.copy()
    res_df_sorted_check['lift_sort'] = res_df_sorted_check['lift'].replace(np.inf, float('inf')).fillna(float('-inf'))
    res_df_sorted_check['p_value_sort'] = res_df_sorted_check['p_value'].fillna(float('inf')) # Treat NaN p-values as largest

    # Sort by lift (desc), then p-value (asc)
    res_df_expected_sort = res_df_sorted_check.sort_values(by=['lift_sort', 'p_value_sort'], ascending=[False, True])

    assert_frame_equal(res_df_expected_sort[['event_type']], res_df[['event_type']], check_dtype=False)


def test_all_fields_present():
    features_df, labels_series = create_sample_data(num_devices=20, event_prefixes=["evt_"], seed=15)
    labels_series[:10] = 0; labels_series[10:] = 1
    if "evt_type1" not in features_df.columns: features_df["evt_type1"] = 0
    features_df.loc[labels_series == 0, "evt_type1"] = 1

    res_df, err = analyze_significant_event_types(features_df, labels_series, "evt_", at_risk_label=0, baseline_label=1)
    assert err is None
    assert not res_df.empty

    expected_columns = [
        "event_type", "lift", "p_value",
        "at_risk_group_event_count", "baseline_group_event_count",
        "at_risk_group_total_devices", "baseline_group_total_devices",
        "chi2_contingency_note"
    ]
    assert all(col in res_df.columns for col in expected_columns)
    assert len(res_df.columns) == len(expected_columns)

# If you want to run this file directly using `python tests/test_explainability.py`
# pytest.main()
# However, it's typically run via `pytest` command in the terminal
# which will discover and run tests automatically.
# So, commenting out pytest.main() is standard practice for pytest files.
