process plotter {
    input: 
        tuple val(participant_id), path(data_file), val(plot_type)
    output: 
        path("${participant_id}_${data_file.baseName}.pdf")
    script: 
        """
        python plotter.py $plot_type $data_file ${participant_id}_${data_file.baseName}.pdf
        """
}
