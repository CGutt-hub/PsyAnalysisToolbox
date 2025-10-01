process fnirs_preprocessor {
    input: 
        tuple val(participant_id), path(input_file)
    output: 
        tuple val(participant_id), path(fnirs_out)
    script: 
        """
        python fnirs_preprocessor.py $input_file $participant_id $fnirs_out
        """
}
