process plv_analyzer {
    input: 
        val signal_parquet_list 
        val bands_config 
        val channels_list 
        val participant_id
    output: 
        path plv_out
    script: 
        """
        python plv_analyzer.py $signal_parquet_list $bands_config $channels_list $participant_id $plv_out
        """
}
