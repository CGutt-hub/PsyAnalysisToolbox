process erp_analyzer {
    input: 
        path input_fif 
        val participant_id
    output: 
        path erp_out
    script: 
        """
        python erp_analyzer.py $input_fif $participant_id $erp_out
        """
}
