process correl_analyzer {
    input: 
        path input_parquet 
        val participant_id
    output: 
        path correl_out
    script: 
        """
        python correl_analyzer.py $input_parquet $participant_id $correl_out
        """
}
