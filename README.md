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
*   **`analysis.group_analyzer.GroupAnalyzer`**: Functions as a configurable pipeline engine for comprehensive group-level analysis. It processes individual participant artifacts, performs data aggregation (e.g., concatenating dataframes, collecting specific metrics across participants), and executes statistical analyses. The entire workflow is defined by a user-provided list of tasks (`analysis_tasks_config`). The `GroupAnalyzer` achieves universality by using a general configuration dictionary (`general_configs`) for any study-specific parameters (like lists of experimental conditions or frequency bands) needed by particular tasks, and it employs a suite of `available_analyzers` for the statistical operations.
*   **Specialized Analyzer Modules (within `analysis/`)**: Individual Python modules that implement specific algorithms or processing steps for various biosignals (e.g., HRV, EEG spectral analysis, fNIRS GLM, connectivity, EDA features) and statistical operations. The toolbox can be expanded by adding new modules here.

The design anticipates future extensions, which might involve new Python modules, integration with external tools, or scripts written in other programming languages. The emphasis is on maintaining clear interfaces and data exchange mechanisms.
<!-- NOTE: This forward-looking statement is good for universality. -->

*(Supporting modules for tasks like reporting, plotting, configuration management, and general utilities are also part of the ecosystem but are not detailed here for brevity.)*

## Built With

*   **Python**: The primary programming language for the core toolbox.
*   **MNE-Python**: For EEG, MEG, and fNIRS data processing and analysis.
*   **MNE-NIRS**: For fNIRS specific functionalities.
*   **NeuroKit2**: For biosignal processing (ECG, EDA, etc.).
*   **Pingouin**: For statistical analyses.
*   **NumPy**: For numerical operations.
*   **Pandas**: For data manipulation and analysis.
*   **SciPy**: For scientific and technical computing.

## Authors

*   **Cagatay Gutt** - *Initial work & Lead Developer* - **[yourusername]**

*(See also the list of contributors who participated in this project.)*