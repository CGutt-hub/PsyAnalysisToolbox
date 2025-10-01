process quest_analyzer {
    input: path input_parquet val answer_key_pattern val tick_pattern
    output: path quest_out
    script: """python questionnaire_analyzer.py $input_parquet $answer_key_pattern $tick_pattern $quest_out"""
}
