process xdf_reader {
    input: 
        tuple path(xdf_file), val(participant_id)
    output: 
        tuple val(participant_id), path(xdf_out)
    script: 
        """
        python xdf_reader.py $xdf_file $participant_id $xdf_out
        """
}
