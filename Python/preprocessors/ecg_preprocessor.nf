process ecg_preprocessor {
    input: 
        tuple val(participant_id), path(input_parquet)
    output: 
        tuple val(participant_id), path(ecg_out)
    script: 
        """
        python ecg_preprocessor.py $input_parquet $participant_id $ecg_out
        """
}
