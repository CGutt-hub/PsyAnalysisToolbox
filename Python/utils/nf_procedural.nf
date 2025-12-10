// Efficient participant discovery and output folder creation for any analysis pipeline
process new_participant {
    input:
        val input_dir
        val output_dir
        val participant_pattern
    output:
        tuple val(participant_id), path(output_folder)
    script:
    """
    for d in \$(ls -d ${input_dir}/${participant_pattern}); do
        participant_id=\$(basename "\$d")
        out_path=\"${output_dir}/\$participant_id\"
        if [ ! -d \"\$out_path\" ] || [ \"\$(ls -A \"\$out_path\" 2>/dev/null | wc -l)\" -eq 0 ]; then
            mkdir -p \"\$out_path\"
            echo \"\$participant_id\" > participant_id.txt
            echo \"\$out_path\" > output_folder.txt
        fi
    done
    """
    publishDir output_dir, mode: 'copy', pattern: '*.txt'
}
// Generic Nextflow procedural infrastructure for Python-based pipelines
// Handles output directory creation, log management, and python_dispatcher process

// Python dispatcher for all module calls
process python_dispatcher {
    debug params.debug_mode ?: false
    errorStrategy 'finish'
    input:
        val python_exe
        val script_name
        val input_file
        val script_args
    output:
        // Emit both .fif and .parquet files from any subfolder, for all output types
        path "**/*.parquet", emit: 'parquet', optional: true
        path "**/*.fif", emit: 'fif', optional: true
    script:
        """
        ${python_exe} -u ${script_name} ${input_file} ${script_args}
        """
}

// Per-participant log management
process copy_participant_logs {
    publishDir "${output_dir}/logs", mode: 'copy'
    input:
    val participant_id
    path output_dir
    val trigger  // To ensure this runs after other processes
    output:
    path "*.{log,txt}"
    script:
    """
    # Create participant-specific log summary
    echo "Pipeline execution logs for participant: ${participant_id}" > ${participant_id}_pipeline.log
    echo "Execution date: \$(date)" >> ${participant_id}_pipeline.log
    echo "Participant ID: ${participant_id}" >> ${participant_id}_pipeline.log
    echo "Input directory: ${params.input_dir}/${participant_id}" >> ${participant_id}_pipeline.log
    echo "Output directory: ${output_dir}" >> ${participant_id}_pipeline.log
    echo "Python executable: ${params.python_exe}" >> ${participant_id}_pipeline.log
    echo "================================" >> ${participant_id}_pipeline.log
    # Copy relevant Nextflow logs that mention this participant
    if [ -f "${workflow.launchDir}/.nextflow.log" ]; then
        echo "Nextflow execution logs:" >> ${participant_id}_pipeline.log
        grep -i "${participant_id}" "${workflow.launchDir}/.nextflow.log" >> ${participant_id}_pipeline.log || echo "No participant-specific logs found in main log" >> ${participant_id}_pipeline.log
    fi
    # Copy work directory logs if available
    if [ -d "${workflow.workDir}" ]; then
        echo "Work directory logs:" >> ${participant_id}_pipeline.log
        find ${workflow.workDir} -type f -name "*.log" -exec grep -l "${participant_id}" {} \\; | head -20 | while read logfile; do
            echo "--- \$(basename \$logfile) ---" >> ${participant_id}_pipeline.log
            tail -50 "\$logfile" >> ${participant_id}_pipeline.log
            echo "" >> ${participant_id}_pipeline.log
        done
    fi
    # Create a processing summary
    echo "Processing completed for ${participant_id}" > ${participant_id}_summary.txt
    echo "Check ${participant_id}_pipeline.log for detailed execution information" >> ${participant_id}_summary.txt
    """
}
