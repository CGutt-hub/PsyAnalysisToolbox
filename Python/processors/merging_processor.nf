// Merging Processor
process merging_processor {
    input: 
        tuple val(participant_id), path(modality_files), val(merge_keys)
    output: 
        tuple val(participant_id), path(merged_out)
    script: 
        """
        python merging_processor.py $modality_files $merge_keys $participant_id $merged_out
        """
}