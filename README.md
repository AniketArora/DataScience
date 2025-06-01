# Python Data Analysis Project

This project provides a Streamlit-based application for advanced data analysis. Given the nature of available data (specifically, the absence of timed failure labels), the current focus is on **unsupervised learning techniques to identify unusual device behaviors and patterns from time series data.** These findings can then be correlated with a user-provided list of devices known to have failed, aiming to uncover leading indicators or characteristics of problematic devices.

## Core Focus & Features

The primary goal is to perform deep exploratory analysis on time series data from multiple sources (e.g., IoT sensors) to detect anomalies and segment operational behaviors.

**Key Capabilities:**

*   **Database Connectivity & Data Handling:** Connect to PostgreSQL and Elasticsearch to load data. UI elements to specify entity/device IDs, timestamp columns, and metric columns for analysis.
*   **Interactive Frontend:** Utilize Streamlit for a responsive and intuitive user interface for data exploration, parameter input, and visualization of results.
*   **Time Series Specialization (Foundational):**
    *   **Individual Series Profiling:** Detailed statistical summaries, missing value analysis, and stationarity tests (e.g., Augmented Dickey-Fuller) for selected time series.
    *   **Individual Series Decomposition:** Break down individual time series into trend, seasonality, and residuals.
    *   **Basic Anomaly Detection (Individual Series):** Implement statistical methods (Z-score, IQR) to identify unusual data points within a single time series. Visual highlighting of these anomalies.
*   **Advanced Unsupervised Analysis (New Focus):**
    *   **Time Series Feature Engineering:** Generates a rich "behavioral fingerprint" for each device/entity from its time series data. This can include rolling statistics (mean, std, min, max), trend metrics (slope, changes), volatility measures, autocorrelation features, complexity and entropy measures, and more.
    *   **Population-Level Anomaly Detection (Device Fingerprints):** Applies unsupervised machine learning models (e.g., Isolation Forest, One-Class SVM, Autoencoders) to the device feature fingerprints. This identifies devices exhibiting rare or significantly different behavior compared to the entire population.
    *   **Device Behavior Clustering:** Groups devices based on their feature fingerprints using clustering algorithms (e.g., K-Means, DBSCAN, Hierarchical Clustering). This helps discover distinct operational patterns or segments within the device population. Some clusters may represent behaviors that correlate with higher (or lower) failure likelihood.
    *   **Explainability & Insight Generation:**
        *   Identifies key features that distinguish anomalous devices or define behavioral clusters (e.g., using feature importance from models or statistical comparisons between clusters).
        *   Provides initial natural language summaries of these findings to aid interpretation.
    *   **Validation with Known Failures:** Allows users to input a list of known failed device identifiers. The system then assesses how effectively the unsupervised methods (anomaly detection, clustering) highlight these devices (e.g., are known failed devices overrepresented in certain clusters or among top anomalies?). This helps validate if detected "unusual" behaviors are meaningful risk indicators.
*   **General Data Analysis (Existing):**
    *   Descriptive statistics for general datasets.
    *   Data filtering capabilities.
*   **Extensible Architecture:** Designed with modular analysis components to facilitate future expansion.

## Future Enhancements (Vision - Refined)

*   **Event Data Integration:** Incorporate discrete event data (e.g., maintenance logs, error codes) into feature engineering and correlate event occurrences with time series patterns and device anomalies.
*   **Advanced Unsupervised Models:** Explore and implement more sophisticated unsupervised learning models for anomaly detection (e.g., VAEs, GMMs) and clustering.
*   **Dynamic Modeling:** Develop capabilities for models to adapt to evolving data patterns and concept drift over time.
*   **Root Cause Analysis Support:** Enhance explainability features to provide deeper insights into potential root causes of anomalous behavior, possibly by linking to specific feature deviations or event sequences.
*   **Comparative Analysis & Fleet Health Monitoring:** Develop dashboards for comparing behavior across different segments of devices/entities and monitoring overall fleet health based on anomaly rates and cluster distributions.
*   **Scalability:** Continuously improve performance for handling very large datasets and a large number of devices/entities.

## Project Structure (Updated)

*   `src/main.py`: Main Streamlit application.
*   `src/database.py`: Modules for database connections.
*   `src/utils.py`: Utility functions.
*   `src/analysis_modules/`: Directory containing specialized analysis functions:
    *   `profiling.py` (Individual series profiling)
    *   `decomposition.py` (Individual series decomposition)
    *   `anomalies.py` (Individual series anomaly detection, to be expanded/refocused for unsupervised population methods)
    *   `feature_engineering.py` (New - For creating device behavioral fingerprints)
    *   `clustering.py` (New - For device behavior clustering)
    *   `explainability.py` (New - For interpreting unsupervised model results)
*   `data/`: Directory for sample datasets.
*   `tests/`: Directory for unit tests.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
*   `requirements.txt`: Project dependencies.

## Getting Started (Placeholder)

(Instructions on how to set up and run the project will be added here once the initial development is complete.)