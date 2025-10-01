process fai_analyzer {
    input: 
        path input_parquet 
        val fai_band_name 
        val electrode_pairs 
        val participant_id
    output: 
        path fai_out
    script: 
        """
        python fai_analyzer.py $input_parquet $fai_band_name $electrode_pairs $participant_id $fai_out
        """
}
