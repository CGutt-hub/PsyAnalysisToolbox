process epoching_processor {
    input: 
        tuple val(participant_id), path(signal_file), path(trigger_file)
    output: 
        tuple val(participant_id), path(epoch_out)
    script: 
        """
        python epoching_processor.py $signal_file $trigger_file $participant_id $epoch_out
        """
}
