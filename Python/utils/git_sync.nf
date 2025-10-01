process git_sync {
    input:
        tuple val(participant_id), path(plot_file)
    script: 
        """
        python git_sync.py $plot_file $participant_id
        """
}
