# AGENTS.md: Instructions for AI Agents

This document provides instructions and guidelines for AI agents working on this codebase. Adhering to these guidelines is crucial for maintaining code quality, consistency, and stability.

## 1. Project Overview

This is a Python-based data analysis tool with a web frontend built using the Streamlit framework.

### Key Technologies:
- **Backend:** Python
- **Frontend:** Streamlit
- **Data Handling:** Pandas
- **Data Analysis:** Scikit-learn, Statsmodels, Matplotlib
- **Database:** PostgreSQL, Elasticsearch
- **Dependency Management:** `uv`
- **Testing:** `pytest` (for unit tests) and Selenium (for UI tests)

### Architecture:
- `src/main.py`: The main entry point of the Streamlit application. It handles the UI and orchestrates the analysis modules.
- `src/analysis_modules/`: Contains individual modules for different types of data analysis (e.g., `profiling.py`, `clustering.py`). This is the primary location for adding new analysis features.
- `src/database.py`: Handles database connections.
- `src/utils.py`: Contains utility functions used across the application.
- `tests/`: Contains all tests. It is organized to mirror the `src` directory structure.
- `pyproject.toml`: Defines project metadata and dependencies.

## 2. Environment Setup

To work on this project, you must set up a virtual environment and install the required dependencies.

1.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv sync
    ```

## 3. Running the Application

To run the Streamlit application locally, use the following command:

```bash
streamlit run src/main.py
```

## 4. Testing

A comprehensive test suite is in place to ensure code quality. All tests must pass before submitting any changes.

### Running All Tests

The test suite includes both unit and UI tests. To run all tests, execute the following commands:

1.  **Set the PYTHONPATH:** The application uses absolute imports from the project root. You must set the `PYTHONPATH` to the root of the repository.
    ```bash
    export PYTHONPATH=$(pwd)
    ```

2.  **Run pytest:** The CI environment uses `uv run pytest`. You should do the same.
    ```bash
    uv run pytest
    ```

### UI Tests (Selenium)

The UI tests use Selenium and require a web driver. The CI workflow installs Google Chrome. If you are running locally and encounter issues, ensure you have a compatible version of Google Chrome and ChromeDriver installed.

## 5. Code Conventions

- **Style Guide:** All Python code must follow the [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/).
- **Docstrings:** Use Google-style docstrings for all modules, classes, and functions.
- **Modularity:** The project is designed to be modular. New functionality, especially analysis features, should be added as new functions or classes in the `src/analysis_modules/` directory. Avoid adding complex logic directly into `src/main.py`.

## 6. Adding a New Analysis Module

To add a new analysis feature:

1.  **Create a new function** in the appropriate file within `src/analysis_modules/`. If the functionality is new, you might need to create a new file (e.g., `src/analysis_modules/new_feature.py`).
2.  **Import the new function** in `src/main.py`.
3.  **Integrate the function** into the Streamlit UI within `src/main.py`, likely within a new tab or section.
4.  **Add corresponding unit tests** in the `tests/` directory. For a new module `src/analysis_modules/new_feature.py`, the tests should be in `tests/test_new_feature.py`.

## 7. Frontend Verification

Since this is a Streamlit application, any changes to the UI must be verified.

1.  **Run the UI tests** located in `tests/test_ui.py`.
2.  **Manually inspect** the application by running it locally to ensure that UI elements are rendered correctly and the application is behaving as expected.
3.  For any significant UI change, you may be required to update the Selenium tests.

## 8. Managing Dependencies

Project dependencies are managed in `pyproject.toml`.

-   To **add a new dependency**, add it to the `dependencies` list in `pyproject.toml` and then run `uv sync`.
-   Do **not** use `pip install` directly for new dependencies, as this will not update the project's dependency list.

## 9. Do's and Don'ts

### Do:
- **Write tests** for all new code.
- **Run all tests** before submitting changes.
- **Follow the existing modular structure.**
- **Keep `src/main.py` focused on UI and orchestration.**
- **Update `pyproject.toml`** when adding new dependencies.

### Don't:
- **Do not commit** code that breaks existing tests.
- **Do not add business logic** directly to `src/main.py`. Place it in the appropriate module in `src/analysis_modules/`.
- **Do not ignore UI verification.** Changes to the frontend must be tested.
- **Do not modify build artifacts** directly. Always change the source files.
