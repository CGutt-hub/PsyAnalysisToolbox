process psd_analyzer {
    input: 
        path input_fif 
        path bands_parquet 
        val participant_id 
        path channels_parquet
    output: 
        path psd_out
    script: 
        """
        python psd_analyzer.py $input_fif $bands_parquet $participant_id $channels_parquet $psd_out
        """
}
