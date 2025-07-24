from PsyAnalysisToolbox.Python.utils.parallel_runner import DAGTask
from PsyAnalysisToolbox.Python.services.questionnaire_service import process_and_score_questionnaire
from PsyAnalysisToolbox.Python.services.reporting_service import generate_reports
from PsyAnalysisToolbox.Python.services.preprocessing_helpers import _preprocess_eeg_stream, _preprocess_ecg_stream, _preprocess_eda_stream, _preprocess_fnirs_stream

def build_participant_dag(participant_artifacts, config, components, logger):
    """
    Build the DAG for a participant's pipeline. Returns a list of DAGTask objects.
    Each step should use the appropriate modular service (preprocessing, analysis, questionnaire, reporting, etc.).
    Each step checks for required inputs and logs errors if dependencies are missing.
    """
    def preprocess_eeg():
        if not participant_artifacts.has('xdf_streams') or 'eeg' not in participant_artifacts.get('xdf_streams', {}):
            logger.error("EEG preprocessing skipped: 'eeg' not found in xdf_streams.")
            return None
        return _preprocess_eeg_stream(
            config,
            components,
            logger,
            participant_artifacts.get('xdf_streams'),
            participant_artifacts.get('participant_id'),
            participant_artifacts.get('output_dir')
        )
    def preprocess_fnirs():
        if not participant_artifacts.has('xdf_streams') or 'fnirs_cw_amplitude' not in participant_artifacts.get('xdf_streams', {}):
            logger.error("fNIRS preprocessing skipped: 'fnirs_cw_amplitude' not found in xdf_streams.")
            return None
        return _preprocess_fnirs_stream(
            config,
            components,
            logger,
            participant_artifacts.get('xdf_streams'),
            participant_artifacts.get('participant_id'),
            participant_artifacts.get('output_dir')
        )
    def preprocess_ecg():
        if not participant_artifacts.has('xdf_streams') or 'ecg_df' not in participant_artifacts.get('xdf_streams', {}):
            logger.error("ECG preprocessing skipped: 'ecg_df' not found in xdf_streams.")
            return None
        return _preprocess_ecg_stream(
            config,
            components,
            logger,
            participant_artifacts.get('xdf_streams'),
            participant_artifacts.get('participant_id'),
            participant_artifacts.get('output_dir')
        )
    def preprocess_eda():
        if not participant_artifacts.has('xdf_streams') or 'eda_df' not in participant_artifacts.get('xdf_streams', {}):
            logger.error("EDA preprocessing skipped: 'eda_df' not found in xdf_streams.")
            return None
        return _preprocess_eda_stream(
            config,
            components,
            logger,
            participant_artifacts.get('xdf_streams'),
            participant_artifacts.get('participant_id'),
            participant_artifacts.get('output_dir')
        )
    def process_questionnaires():
        if not participant_artifacts.has('questionnaire_raw_df'):
            logger.error("Questionnaire processing skipped: 'questionnaire_raw_df' not found in artifacts.")
            return None
        return process_and_score_questionnaire(participant_artifacts, config, components, logger)
    def run_analyses():
        required = ['eeg_preprocessed','fnirs_preprocessed','ecg_preprocessed','eda_preprocessed','questionnaire_processed']
        missing = [k for k in required if not participant_artifacts.has(k)]
        if missing:
            logger.error(f"Analysis step skipped: missing required preprocessed data: {missing}")
            return None
        return components['analysis_service'].run(participant_artifacts, config, logger)
    def generate_reports_task():
        if not participant_artifacts.has('analyses_done'):
            logger.error("Reporting skipped: 'analyses_done' not found in artifacts.")
            return None
        return generate_reports(participant_artifacts, config, components, logger)

    dag_tasks = [
        DAGTask('preprocess_eeg', preprocess_eeg, deps=[], outputs=['eeg_preprocessed']),
        DAGTask('preprocess_fnirs', preprocess_fnirs, deps=[], outputs=['fnirs_preprocessed']),
        DAGTask('preprocess_ecg', preprocess_ecg, deps=[], outputs=['ecg_preprocessed']),
        DAGTask('preprocess_eda', preprocess_eda, deps=[], outputs=['eda_preprocessed']),
        DAGTask('process_questionnaires', process_questionnaires, deps=[], outputs=['questionnaire_processed']),
        DAGTask('run_analyses', run_analyses, deps=['eeg_preprocessed','fnirs_preprocessed','ecg_preprocessed','eda_preprocessed','questionnaire_processed'], outputs=['analyses_done']),
        DAGTask('generate_reports', generate_reports_task, deps=['analyses_done'], outputs=['results_reported'])
    ]
    return dag_tasks 