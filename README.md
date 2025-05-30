# PsyAnalysisToolbox

The **PsyAnalysisToolbox** is a versatile collection of tools designed to facilitate the analysis of psychophysiological data. It provides a suite of Python-based modules for processing and analyzing various biosignals, along with a framework that can be extended to incorporate analyses from other platforms.
<!-- NOTE: This high-level description is good and should remain stable. -->

## Overview

This toolbox is designed to streamline common and advanced analysis pipelines in psychophysiology research. It provides a modular framework for processing and analyzing data from various biosignal modalities. The system emphasizes a configurable approach, allowing researchers to tailor analyses to specific experimental designs and research questions. Core capabilities revolve around signal processing, feature extraction, statistical analysis, and group-level aggregation of psychophysiological data.

The `AnalysisService` acts as a central delegator to these specialized analyzers, simplifying the execution of complex analysis chains. Additionally, the `GroupAnalyzer` module supports group-level statistical analyses by aggregating and processing data from multiple participants.

The toolbox is designed with extensibility in mind, allowing for the future integration of new analysis modules, including scripts written in other languages such as MATLAB, to create a comprehensive research environment.
<!-- NOTE: The "Core Features" section has been integrated into this more general "Overview" to reduce specific update needs. The capabilities mentioned above (signal processing, feature extraction, stats, group analysis) are broad enough to cover many additions. -->

## Modules Overview

The toolbox is architected around a core set of Python-based services and specialized analysis modules, primarily located within the `analysis/` directory. This structure facilitates a modular approach to psychophysiological data analysis. Key architectural components currently include: <!-- NOTE: This general description of the architecture is excellent for avoiding frequent updates. -->

*   **`analysis.analysis_service.AnalysisService`**: A facade/service layer that delegates analysis tasks to specialized analyzers. This is often the primary entry point for performing analyses.
*   **`analysis.group_analyzer.GroupAnalyzer`**: Orchestrates group-level analyses by aggregating data from multiple participants and performing statistical tests, often utilizing the `AnalysisService`.
*   **Specialized Analyzer Modules (within `analysis/`)**: Individual Python modules that implement specific algorithms or processing steps for various biosignals (e.g., HRV, EEG spectral analysis, fNIRS GLM, connectivity, EDA features) and statistical operations. The toolbox can be expanded by adding new modules here.

The design anticipates future extensions, which might involve new Python modules, integration with external tools, or scripts written in other programming languages. The emphasis is on maintaining clear interfaces and data exchange mechanisms.
<!-- NOTE: This forward-looking statement is good for universality. -->

*(Supporting modules for tasks like reporting, plotting, configuration management, and general utilities are also part of the ecosystem but are not detailed here for brevity.)*

## Basic Usage

The primary way to use the Python modules in this toolbox is by importing the `AnalysisService` (for orchestrated analyses) or individual analyzer classes (for specific tasks) into your scripts or Jupyter notebooks. Configuration for specific analyses (e.g., frequency bands, electrode pairs, statistical parameters) is typically managed through external configuration files (e.g., YAML) loaded by the toolbox.

```python
# Example: Using the AnalysisService (conceptual)
from analysis.analysis_service import AnalysisService
from utils.config_loader import load_main_config # Assuming a config loader
from utils.logger_setup import setup_logger # Assuming a logger setup

logger = setup_logger()
main_config = load_main_config('path/to/your/config.yaml')

analysis_service = AnalysisService(logger, main_config)

# --- Example: HRV Analysis ---
# Assuming nni_data is an array of NN intervals
# rmssd = analysis_service.calculate_rmssd_from_nni_array(nni_data)
# logger.info(f"Calculated RMSSD: {rmssd}")

# --- Example: PSD/FAI Analysis ---
# Assuming raw_eeg, events, event_id_map, and configs are prepared
# psd_results, fai_results = analysis_service.calculate_psd_and_fai(
#     raw_eeg_processed, events_array, event_id_map,
#     fai_alpha_band_config, eeg_bands_config_for_beta,
#     fai_electrode_pairs_config, analysis_epoch_tmax_config
# )

# --- Example: Group Analysis ---
# from analysis.group_analyzer import GroupAnalyzer
# Assuming all_participant_artifacts is a list of processed data from individuals
# group_analyzer = GroupAnalyzer(logger, output_base_dir='path/to/group_results', ...)
# group_analyzer.run_group_analysis(all_participant_artifacts)
```

Refer to the specific methods within each analyzer class and the `AnalysisService` for detailed parameters and functionality. Example scripts or Jupyter notebooks in the `scripts/` or `notebooks/` directory (if available) would provide more concrete usage examples.

## Directory Structure (Conceptual)

A well-organized directory structure is crucial, especially for a growing toolbox:

```
PsyAnalysisToolbox/
├── analysis/                  # Core Python analysis modules (HRV, PSD, GLM, etc.)
│   ├── __init__.py
│   └── ... (analyzer_service.py, hrv_analyzer.py, etc.)
├── configs/                   # Configuration files (e.g., YAML, JSON)
├── data/                      # Example datasets, raw data (if small), or processed data
│   ├── sample_participant_01/
│   └── ...
├── matlab_scripts/            # Placeholder for MATLAB analysis scripts
│   ├── processing/
│   └── stats/
├── notebooks/                 # Jupyter notebooks for examples, exploration, tutorials
├── reporting/                 # Modules for generating reports, plots (e.g., plotting_service.py)
├── results/                   # Output directory for analysis results (often .gitignore'd)
│   ├── individual/
│   └── group/
├── scripts/                   # Main executable Python scripts, batch processing scripts
├── tests/                     # Unit and integration tests for Python modules
├── utils/                     # Utility functions (e.g., data loaders, helpers, logger setup)
├── .env.example               # Example environment file
├── .gitignore
├── LICENSE.md
├── README.md
└── requirements.txt           # Python dependencies
```

## Extending the Toolbox

The PsyAnalysisToolbox is designed to be extensible:

### Adding New Python Analyzers

1.  Create a new Python file in the `analysis/` directory (e.g., `new_feature_analyzer.py`).
2.  Define a class (e.g., `NewFeatureAnalyzer`) with an `__init__(self, logger, ...)` method and methods for its specific analysis tasks.
3.  Optionally, integrate it into `AnalysisService` by adding an instance of your new analyzer and delegating methods to it.
4.  Add unit tests for your new module in the `tests/` directory.

### Integrating MATLAB (or other language) Scripts

1.  **Organize Scripts**: Place MATLAB scripts in a dedicated directory (e.g., `matlab_scripts/`).
2.  **Data Exchange**:
    *   Use common, easily parsable file formats for data input/output (e.g., `.csv`, `.tsv`, `.mat` files, JSON).
    *   Python scripts can prepare input data and save it in a format MATLAB can read.
    *   MATLAB scripts can save their results in a format Python can read.
3.  **Calling MATLAB from Python**:
    *   **MATLAB Engine API for Python**: If MATLAB is installed, you can use this API to call MATLAB functions and scripts directly from Python, passing data between workspaces.
        ```python
        # Conceptual example
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        # result = eng.your_matlab_function(matlab_input_data, nargout=1)
        # eng.quit()
        ```
    *   **System Calls**: Python's `subprocess` module can be used to run compiled MATLAB executables or MATLAB scripts via the command line if they are designed to accept command-line arguments and handle file I/O.
4.  **Workflow Integration**: Python scripts (e.g., in `scripts/` or within the `AnalysisService`) can orchestrate workflows that include steps processed by MATLAB.

## Running Tests

**[TODO: Describe how to run tests. If using pytest, for example:]**

To run the automated tests for the Python modules:
```bash
pytest tests/
```

Ensure that any test data or specific configurations required for tests are set up correctly.

## Built With

*   **Python**: The primary programming language for the core toolbox.
*   **MNE-Python**: For EEG, MEG, and fNIRS data processing and analysis.
*   **MNE-NIRS**: For fNIRS specific functionalities.
*   **NeuroKit2**: For biosignal processing (ECG, EDA, etc.).
*   **Pingouin**: For statistical analyses.
*   **NumPy**: For numerical operations.
*   **Pandas**: For data manipulation and analysis.
*   **SciPy**: For scientific and technical computing.
*   **(Potentially) MATLAB**: For specialized analyses not yet implemented in Python or for leveraging existing MATLAB codebases.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these general guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them with clear, descriptive messages.
4.  Ensure your code adheres to any existing style guidelines (e.g., run a linter like Flake8).
5.  Write unit tests for new functionality.
6.  Push your changes to your fork (`git push origin feature/your-feature-name`).
7.  Open a Pull Request to the main repository.

Please also consider creating an issue to discuss significant changes before starting work.
*(You might want to create a `CONTRIBUTING.md` file with more detailed guidelines.)*

## Versioning

We use SemVer (Semantic Versioning) for versioning. For the versions available, see the tags on this repository.

## Authors

*   **[Your Name / Your Lab's Name]** - *Initial work & Lead Developer* - **[yourusername]**

*(See also the list of contributors who participated in this project.)*

## License

This project is licensed under the **[Choose an OSI approved license, e.g., MIT, Apache 2.0, GPLv3]** License - see the `LICENSE.md` file for details.
*(You'll need to create a `LICENSE.md` file and choose a license.)*

## Acknowledgments

*   Mention any specific datasets, algorithms, or individuals whose work was foundational.
*   Hat tip to any open-source projects that were particularly inspirational or heavily used.
*   Any funding sources or institutional support.