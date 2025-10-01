process txt_reader {
    input: 
        tuple path(txt_file), val(participant_id)
    output: 
        tuple val(participant_id), path(txt_out)
    script: 
        """
        python txt_reader.py $txt_file $participant_id $txt_out
        """
}
