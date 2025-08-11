import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)


@st.cache_data
def decompose_time_series(series: pd.Series, model="additive", period=None):
    """
    Performs time series decomposition into trend, seasonal, and residual components.

    Args:
        series (pd.Series): The time series to decompose. Must have a DatetimeIndex.
        model (str): Type of decomposition model ('additive' or 'multiplicative').
        period (int, optional): The period of the seasonality. If None, it will be
                                inferred if the series index is a DatetimeIndex with
                                a frequency (e.g., daily data might infer period=7 for weekly).
                                However, explicit setting is often more reliable.
                                A minimum of 2 periods of data is often needed.

    Returns:
        statsmodels.tsa.seasonal.DecomposeResult or None: The decomposition result object,
                                                          or None if decomposition fails.
        str or None: An error message if decomposition fails.
    """
    if not isinstance(series, pd.Series):
        msg = "Input is not a pandas Series."
        logger.warning(msg)
        return None, msg

    if not isinstance(series.index, pd.DatetimeIndex):
        msg = "Series index must be a DatetimeIndex."
        logger.warning(msg)
        return None, msg

    if series.empty:
        msg = "Input series is empty."
        logger.warning(msg)
        return None, msg

    # Ensure there are enough observations for the period
    # seasonal_decompose requires at least 2 full periods if period is specified,
    # or a reasonable amount of data if period is inferred.
    min_obs = 2
    if period:
        min_obs = 2 * period

    if len(series.dropna()) < min_obs:
        msg = f"Not enough data points (need at least {min_obs} non-NaN) for the given period to decompose effectively."
        logger.warning(msg)
        return None, msg

    try:
        if series.index.freq is None and period is None:
            msg = "Series frequency is not set. Please specify a 'period' for decomposition."
            logger.warning(msg)
            return None, msg

        result = seasonal_decompose(series.dropna(), model=model, period=period)
        return result, None
    except Exception as e:
        logger.error(
            "Time series decomposition failed for series %s: %s",
            series.name if series.name else "Unnamed",
            e,
            exc_info=True,
        )
        return None, f"Time series decomposition failed: {e}"


if __name__ == "__main__":
    # Example Usage
    # Create a sample time series with trend, seasonality, and noise
    idx = pd.date_range(start="2020-01-01", periods=100, freq="D")
    data = [
        i + 10 * (i % 7) + ((i / 30) ** 2) + 5 * pd.np.random.randn()
        for i in range(100)
    ]  # Trend, weekly seasonality, noise
    sample_series = pd.Series(data, index=idx, name="SampleTS")
    sample_series.iloc[[10, 30]] = pd.NA  # Add some missing values

    print("--- Additive Decomposition (Period=7) ---")
    additive_result, error = decompose_time_series(
        sample_series, model="additive", period=7
    )
    if error:
        print(f"Error: {error}")
    if additive_result:
        # In a real app, you'd pass additive_result.plot() to st.pyplot() or plot components individually
        fig = additive_result.plot()
        # To show plot in a non-Streamlit environment for testing:
        # plt.show()
        print("Trend head:", additive_result.trend.head())
        print("Seasonal head:", additive_result.seasonal.head())
        print("Residual head:", additive_result.resid.head())

    print("\n--- Multiplicative Decomposition (Period=7) ---")
    # For multiplicative, series values should be positive
    positive_sample_series = sample_series + abs(sample_series.min()) + 1
    multiplicative_result, error = decompose_time_series(
        positive_sample_series, model="multiplicative", period=7
    )
    if error:
        print(f"Error: {error}")
    if multiplicative_result:
        # fig = multiplicative_result.plot()
        # plt.show()
        print("Trend head:", multiplicative_result.trend.head())
        print("Seasonal head:", multiplicative_result.seasonal.head())
        print("Residual head:", multiplicative_result.resid.head())

    print("\n--- Decomposition Error (No Period, No Freq) ---")
    series_no_freq = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        index=pd.to_datetime(
            [
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2021-01-10",
                "2021-01-11",
                "2021-01-12",
                "2021-01-13",
                "2021-01-14",
                "2021-01-15",
                "2021-01-16",
                "2021-01-17",
                "2021-01-18",
            ]
        ),
    )
    # Ensure index has no frequency set for this test
    series_no_freq.index.freq = None
    res_no_freq, error_no_freq = decompose_time_series(series_no_freq, model="additive")
    if error_no_freq:
        print(f"Error: {error_no_freq}")

    print("\n--- Decomposition Error (Not Enough Data) ---")
    short_series = sample_series.head(10)
    res_short, error_short = decompose_time_series(
        short_series, model="additive", period=7
    )
    if error_short:
        print(f"Error: {error_short}")
