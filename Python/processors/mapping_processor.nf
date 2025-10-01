process mapping_processor {
    input: 
        tuple val(participant_id), path(input_a), path(input_b), val(key_a), val(key_b)
    output: 
        tuple val(participant_id), path(mapped_out)
    script: 
        """
        python mapping_processor.py $input_a $input_b $key_a $key_b $participant_id $mapped_out
        """
}
