process glm_analyzer {
    input: 
        path input_parquet 
        path design_matrix_parquet 
        val sfreq 
        val ch_types_map 
        val contrasts_config 
        val participant_id
    output: 
        path glm_out
    script: 
        """
        python glm_analyzer.py $input_parquet $design_matrix_parquet $sfreq $ch_types_map $contrasts_config $participant_id $glm_out
        """
}
