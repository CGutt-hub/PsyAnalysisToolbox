// CSV Reader as a Nextflow module
process csv_reader {
    input: 
        tuple path(csv_file), val(participant_id)
    output: 
        tuple val(participant_id), path(csv_out)
    script: 
        """
        python csv_reader.py $csv_file $participant_id $csv_out
        """
}
