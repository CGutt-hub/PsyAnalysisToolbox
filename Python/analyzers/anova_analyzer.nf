process anova_analyzer {
    input: 
        path input_parquet 
        val dv 
        val between 
        val participant_id 
        val apply_fdr
    output: 
        path anova_out
    script: 
        """
        python anova_analyzer.py $input_parquet $dv $between $participant_id $apply_fdr $anova_out
        """
}
