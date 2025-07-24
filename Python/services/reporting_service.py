# moved from reporters/reporting_service.py
# (content unchanged, just update imports elsewhere) 

def generate_reports(participant_artifacts, config, components, logger):
    """
    Generates all reports and plots for a participant.
    """
    xml_reporter = components.get('xml_reporter')
    plot_reporter = components.get('plot_reporter')
    pid = participant_artifacts.participant_id if hasattr(participant_artifacts, 'participant_id') else participant_artifacts.get('participant_id')
    output_dir = participant_artifacts.output_dir if hasattr(participant_artifacts, 'output_dir') else participant_artifacts.get('output_dir')
    q_results_df = participant_artifacts.get('questionnaire_results_df')
    fai_results_df = participant_artifacts.get('fai_results_df')
    plv_results_df = participant_artifacts.get('plv_results_df')
    # Example: Save questionnaire results
    if q_results_df is not None:
        xml_reporter.save_dataframe(data_df=q_results_df, output_dir=output_dir, filename=f"{pid}_questionnaire_results.xml")
        logger.info("Saved questionnaire results XML.")
    # Example: Save FAI and PLV results
    if fai_results_df is not None:
        xml_reporter.save_dataframe(data_df=fai_results_df, output_dir=output_dir, filename=f"{pid}_fai_results.xml")
        logger.info("Saved FAI results XML.")
    if plv_results_df is not None:
        xml_reporter.save_dataframe(data_df=plv_results_df, output_dir=output_dir, filename=f"{pid}_plv_results.xml")
        logger.info("Saved PLV results XML.")
    # Example: Generate plots (implement as needed)
    if plot_reporter is not None:
        plot_reporter.generate_all_plots(participant_artifacts, config, output_dir)
        logger.info("Generated all plots.")
    return True 