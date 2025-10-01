process event_preprocessor {
    input: 
        path events_parquet 
        val sfreq
    output: 
        path event_out
    script: 
        """
        python event_preprocessor.py $events_parquet $sfreq $event_out
        """
}
