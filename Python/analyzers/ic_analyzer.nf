process ic_analyzer {
    input: 
        path input_parquet 
        val participant_id
    output: 
        path ic_out
    script: 
        """
        python ic_analyzer.py $input_parquet $participant_id $ic_out
        """
}
