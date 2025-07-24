# moved from processors/questionnaire_service.py
# (content unchanged, just update imports elsewhere) 

def process_and_score_questionnaire(participant_artifacts, config, components, logger):
    """
    Handles questionnaire preprocessing, mapping, and scoring for a participant.
    """
    raw_df = participant_artifacts.get('questionnaire_raw_df')
    if raw_df is None or raw_df.empty:
        logger.info("No raw questionnaire data found. Skipping questionnaire processing.")
        return None
    preprocessor = components['questionnaire_preprocessor']
    scale_processor = components['questionnaire_scale_processor']
    preproc_config = dict(config.items('QuestionnairePreprocessing')) if config.has_section('QuestionnairePreprocessing') else {}
    long_df = preprocessor.extract_items(raw_df, preproc_config)
    if long_df is None or long_df.empty:
        logger.warning("Questionnaire preprocessing produced no data.")
        return None
    scale_defs = {s.replace('ScaleDef_', ''): {'items': [i.strip() for i in config.get(s, 'items').split(',')],
                                               'scoring_method': config.get(s, 'scoring_method', fallback='sum')}
                 for s in config.sections() if s.startswith('ScaleDef_')}
    id_col = config.get('QuestionnairePreprocessing', 'output_participant_id_col_name')
    scored_df = scale_processor.score_scales(
        data_df=long_df,
        scale_definitions=scale_defs,
        participant_id_col=id_col)
    participant_artifacts.set('questionnaire_results_df', scored_df)
    logger.info("Questionnaire processing and scoring complete.")
    return scored_df 