process eeg_preprocessor {
    input: 
        tuple val(participant_id), path(input_file)
    output: 
        tuple val(participant_id), path(eeg_out)
    script: 
        """
        python eeg_preprocessor.py $input_file $participant_id $eeg_out
        """
}
