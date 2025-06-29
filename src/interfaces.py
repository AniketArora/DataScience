from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, Tuple, Optional # For type hinting

class AnalysisModuleInterface(ABC):
    """
    Abstract Base Class for an analysis module.

    This interface defines the common structure and methods that all analysis
    modules should implement to be consistently integrated into the application.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the display name of the analysis module.
        This name is used in UI elements like tabs or selectboxes.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Returns a brief description of what the analysis module does.
        This can be used for tooltips or informational messages.
        """
        pass

    @abstractmethod
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary defining the parameters the module requires.

        The structure for each parameter definition should be:
        {
            "param_name": {
                "type": "int" | "float" | "str" | "selectbox" | "checkbox" | "multiselect",
                "default": Any,
                "label": str,
                "help": Optional[str],
                # Type-specific attributes:
                "min_value": Optional[Union[int, float]], (for "int", "float")
                "max_value": Optional[Union[int, float]], (for "int", "float")
                "step": Optional[Union[int, float]], (for "int", "float")
                "format": Optional[str], (for "float", e.g., "%.2f")
                "options": Optional[list], (for "selectbox", "multiselect")
                # ... other streamlit widget params as needed
            }
        }
        For example:
        {
            "n_clusters": {"type": "int", "default": 3, "label": "Number of Clusters", "min_value": 1, "help": "Select K for K-Means"},
            "eps": {"type": "float", "default": 0.5, "label": "Epsilon (DBSCAN)", "min_value": 0.01, "format": "%.2f", "help": "DBSCAN eps parameter"},
            "method": {"type": "selectbox", "default": "Method A", "label": "Analysis Method", "options": ["Method A", "Method B"]},
            "use_scaling": {"type": "checkbox", "default": True, "label": "Scale Data", "help": "Apply standard scaling before analysis."}
        }
        """
        pass

    @abstractmethod
    def render_parameters_ui(self, st_object: Any, current_values: Dict[str, Any], module_key: str) -> Dict[str, Any]:
        """
        Renders Streamlit input widgets for the module's parameters.

        Args:
            st_object (Any): The Streamlit object to render widgets onto
                             (e.g., st.sidebar, st.container(), st.expander()).
            current_values (Dict[str, Any]): A dictionary of current parameter values,
                                             typically from st.session_state or a local dict.
                                             Used to set the initial state of widgets.
            module_key (str): A unique key prefix for this module instance. This is crucial
                              for ensuring that Streamlit widget keys are unique, especially
                              if multiple instances of this module or other modules are
                              rendered on the same page. Widget keys should be formed like
                              f"{module_key}_{param_name}".

        Returns:
            Dict[str, Any]: A dictionary containing the parameter values selected by the user
                            through the rendered UI elements.
        """
        pass

    @abstractmethod
    def run_analysis(self, data: pd.DataFrame, params: Dict[str, Any], session_state: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Performs the core analysis logic of the module.

        Args:
            data (pd.DataFrame): The input data for the analysis. This is typically
                                 a Pandas DataFrame.
            params (Dict[str, Any]): A dictionary of parameter values, usually obtained
                                     from `render_parameters_ui`.
            session_state (Dict[str, Any]): Provides access to relevant parts of Streamlit's
                                            session state if needed by the analysis module
                                            (e.g., for accessing other computed data or global settings).
                                            It's recommended to pass only necessary parts for better scoping,
                                            but the whole st.session_state can be passed if simpler.

        Returns:
            Tuple[Any, Optional[str]]: A tuple containing:
                - results_object (Any): The output of the analysis. This can be any data
                                        structure (e.g., DataFrame, dict, custom object)
                                        that holds the results.
                - error_message (Optional[str]): A string containing an error message if
                                                 the analysis failed, otherwise None.
        """
        pass

    @abstractmethod
    def render_results(self, st_object: Any, results: Any, session_state: Dict[str, Any]) -> None:
        """
        Renders the analysis results using Streamlit components.

        Args:
            st_object (Any): The Streamlit object to render results onto
                             (e.g., st.container(), st.tabs(), st.expander()).
            results (Any): The results_object returned by `run_analysis`.
            session_state (Dict[str, Any]): Provides access to relevant parts of Streamlit's
                                            session state if needed for rendering
                                            (e.g., for contextual information or comparison data).
        """
        pass

# Example of how a concrete class might use this (for illustration, not part of the file)
# class MySpecificAnalysis(AnalysisModuleInterface):
#     def get_name(self) -> str:
#         return "My Specific Analysis"

#     def get_description(self) -> str:
#         return "This module performs a specific type of analysis X."

#     def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
#         return {
#             "threshold": {"type": "float", "default": 0.5, "label": "Threshold", "min_value": 0.0, "max_value": 1.0},
#             "method": {"type": "selectbox", "default": "A", "label": "Method", "options": ["A", "B", "C"]}
#         }

#     def render_parameters_ui(self, st_object: Any, current_values: Dict[str, Any], module_key: str) -> Dict[str, Any]:
#         updated_values = {}
#         defs = self.get_parameter_definitions()

#         updated_values["threshold"] = st_object.slider(
#             defs["threshold"]["label"],
#             min_value=defs["threshold"]["min_value"],
#             max_value=defs["threshold"]["max_value"],
#             value=current_values.get("threshold", defs["threshold"]["default"]),
#             key=f"{module_key}_threshold"
#         )
#         updated_values["method"] = st_object.selectbox(
#             defs["method"]["label"],
#             options=defs["method"]["options"],
#             index=defs["method"]["options"].index(current_values.get("method", defs["method"]["default"])),
#             key=f"{module_key}_method"
#         )
#         return updated_values

#     def run_analysis(self, data: pd.DataFrame, params: Dict[str, Any], session_state: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
#         # Actual analysis logic here
#         results_df = data[data["some_column"] > params["threshold"]]
#         if params["method"] == "B":
#             results_df = results_df.head(10)
#         return results_df, None

#     def render_results(self, st_object: Any, results: Any, session_state: Dict[str, Any]) -> None:
#         if isinstance(results, pd.DataFrame):
#             st_object.write(f"Analysis completed. Found {len(results)} matching records.")
#             st_object.dataframe(results)
#         else:
#             st_object.error("Invalid results format.")
