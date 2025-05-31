# Python Data Analysis Project

This project aims to provide a flexible and user-friendly tool for advanced data analysis, with an initial strong focus on **time series data**. It helps users explore complex datasets, uncover patterns, detect anomalies, and gain insights, particularly for scenarios like analyzing data from numerous IoT devices.

## Core Focus & Features

The primary goal is to enable deep dives into datasets, especially those containing time series records from multiple sources (e.g., sensors, logs).

**Current & Planned Features:**

*   **Database Connectivity:** Connect to PostgreSQL and Elasticsearch to load data.
*   **Interactive Frontend:** Utilize Streamlit for a responsive and intuitive user interface.
*   **Time Series Specialization:**
    *   **Data Handling:** UI elements to specify entity/device IDs, timestamp columns, and metric columns for analysis.
    *   **Profiling:** Detailed statistical summaries per time series, missing value analysis, and stationarity tests (e.g., Augmented Dickey-Fuller).
    *   **Decomposition:** Break down time series into trend, seasonality, and residuals to understand underlying patterns.
    *   **Anomaly Detection:** Implement statistical methods (Z-score, IQR) to identify unusual data points in time series. Visual highlighting of anomalies.
*   **General Data Analysis (Existing):**
    *   Descriptive statistics for general datasets.
    *   Data filtering capabilities.
    *   Basic visualizations (histograms, scatter plots).
*   **Extensible Architecture:** Designed with modular analysis components to facilitate future expansion.

## Future Enhancements (Vision)

*   **Advanced Anomaly Detection:** Explore more sophisticated techniques (e.g., clustering-based, machine learning models).
*   **Comparative Analysis:** Group and compare behavior across different segments of devices/entities.
*   **Event Correlation:** Analyze relationships between discrete events and time series patterns.
*   **Automated Insight Suggestion:** Develop capabilities to proactively highlight interesting patterns or potential hypotheses.
*   **Scalability:** Continuously improve performance for handling very large datasets.

## Project Structure

*   `src/main.py`: Main Streamlit application.
*   `src/database.py`: Modules for database connections.
*   `src/utils.py`: Utility functions.
*   `src/analysis_modules/`: Directory containing specialized analysis functions (e.g., `profiling.py`, `decomposition.py`, `anomalies.py`).
*   `data/`: Directory for sample datasets.
*   `tests/`: Directory for unit tests.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
*   `requirements.txt`: Project dependencies.

## Getting Started (Placeholder)

(Instructions on how to set up and run the project will be added here once the initial development is complete.)