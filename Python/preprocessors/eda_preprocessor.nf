process eda_preprocessor {
    input: 
        tuple val(participant_id), path(input_parquet)
    output: 
        tuple val(participant_id), path(eda_out)
    script: 
        """
        python eda_preprocessor.py $input_parquet $participant_id $eda_out
        """
}
