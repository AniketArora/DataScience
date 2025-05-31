import pandas as pd
import streamlit as st
from statsmodels.tsa.stattools import adfuller

def get_series_summary_stats(series: pd.Series):
    """Calculates basic descriptive statistics for a time series."""
    if not isinstance(series, pd.Series):
        return pd.DataFrame() # Or raise error

    summary = series.describe().to_frame().T
    summary['median'] = series.median() # describe() might not include median for Series
    summary['mode'] = series.mode().iloc[0] if not series.mode().empty else 'N/A' # Handle multiple modes or no mode
    # Reorder to make it a bit more standard
    cols_order = ['count', 'mean', 'std', 'min', '25%', '50%', 'median', '75%', 'max', 'mode']
    # Filter out any columns not present (e.g., if describe() changes for some Series types)
    summary = summary[[col for col in cols_order if col in summary.columns]]
    return summary

def get_missing_values_summary(series: pd.Series):
    """Calculates and summarizes missing values in a time series."""
    if not isinstance(series, pd.Series):
        return pd.DataFrame()

    missing_count = series.isnull().sum()
    missing_percentage = (missing_count / len(series)) * 100 if len(series) > 0 else 0

    summary_df = pd.DataFrame({
        'Metric': ['Total Count', 'Missing Count', 'Missing Percentage (%)'],
        'Value': [len(series), missing_count, f"{missing_percentage:.2f}%"]
    })
    return summary_df

def perform_stationarity_test(series: pd.Series, significance_level=0.05):
    """
    Performs the Augmented Dickey-Fuller test for stationarity.
    Returns a dictionary with test results and interpretation.
    """
    if not isinstance(series, pd.Series):
        return {
            "error": "Input is not a pandas Series."
        }
    if series.empty:
        return {
            "error": "Input series is empty."
        }

    # ADF test requires no NaN values
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        return {
            "error": "Series is empty after dropping NaN values. Cannot perform ADF test."
        }

    try:
        result = adfuller(series_cleaned)
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]

        interpretation = f"The p-value is {p_value:.4f}. "
        if p_value <= significance_level:
            interpretation += f"The series is likely stationary (reject H0 at {significance_level*100}% significance)."
        else:
            interpretation += f"The series is likely non-stationary (fail to reject H0 at {significance_level*100}% significance)."

        return {
            "ADF Statistic": adf_statistic,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Interpretation": interpretation,
            "Is Stationary (at chosen alpha)": p_value <= significance_level
        }
    except Exception as e:
        return {
            "error": f"ADF test failed: {e}"
        }

if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    data = [i + (i*0.1) + (i%5) for i in range(100)] # Sample data with some pattern
    non_stationary_series = pd.Series(data)

    # Add some NaNs
    non_stationary_series.iloc[[5, 15, 25]] = None

    stationary_series = pd.Series([1,2,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1,3,1,2,1,3])

    print("--- Summary Statistics ---")
    print(get_series_summary_stats(non_stationary_series))

    print("\n--- Missing Values ---")
    print(get_missing_values_summary(non_stationary_series))

    print("\n--- Stationarity Test (Non-Stationary Example) ---")
    stationarity_results_ns = perform_stationarity_test(non_stationary_series)
    for key, val in stationarity_results_ns.items():
        print(f"{key}: {val}")

    print("\n--- Stationarity Test (Stationary Example) ---")
    stationarity_results_s = perform_stationarity_test(stationary_series)
    for key, val in stationarity_results_s.items():
        print(f"{key}: {val}")

    print("\n--- Stationarity Test (Empty Series) ---")
    empty_series = pd.Series([], dtype=float)
    print(perform_stationarity_test(empty_series))

    print("\n--- Stationarity Test (All NaN Series) ---")
    all_nan_series = pd.Series([None, None, None], dtype=float)
    print(perform_stationarity_test(all_nan_series))
