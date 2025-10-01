process scr_analyzer {
    input: 
        path input_parquet 
        val participant_id
    output: 
        path scr_out
    script: 
        """
        python scr_analyzer.py $input_parquet $participant_id $scr_out
        """
}
