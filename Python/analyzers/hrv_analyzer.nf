process hrv_analyzer {
    input: 
        path input_parquet 
        val sfreq 
        val participant_id
    output: 
        path hrv_out
    script: 
        """
        python hrv_analyzer.py $input_parquet $sfreq $participant_id $hrv_out
        """
}
