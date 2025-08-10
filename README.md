# Python Data Analysis Project

This project aims to provide a flexible and user-friendly tool for analyzing datasets from various sources. It features a Streamlit-based frontend for interactive data exploration and visualization.

## Features

*   **Connect to Multiple Databases:** Easily connect to databases like PostgreSQL and Elasticsearch.
*   **Interactive Frontend:** Utilize Streamlit for a responsive and intuitive user interface.
*   **Data Loading and Display:** Load data from connected sources and display it in a tabular format.
*   **Basic Data Analysis:** Perform initial data analysis tasks such as:
    *   Descriptive statistics (mean, median, standard deviation, etc.)
    *   Data filtering
*   **Visualization:** Generate simple plots like histograms and scatter plots.
*   **Extensible:** Designed to be easily extendable with new data sources, analysis modules, and visualizations.

## Planned Structure

*   `src/main.py`: Main Streamlit application.
*   `src/database.py`: Modules for database connections.
*   `src/utils.py`: Utility functions.
*   `data/`: Directory for sample datasets.
*   `tests/`: Directory for unit tests.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
*   `pyproject.toml`: Project dependencies and configuration.

## Getting Started (Placeholder)

(Instructions on how to set up and run the project will be added here once the initial development is complete.)

## How to Launch

To launch the project, first set up the environment and install the dependencies using `uv`:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

Then, run the following command in your terminal:

```bash
streamlit run src/main.py
```