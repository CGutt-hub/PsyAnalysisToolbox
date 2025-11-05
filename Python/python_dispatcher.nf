process python_dispatcher {
    debug params.debug_mode ?: false
    errorStrategy 'finish'
    input:
        val python_exe
        val script_name
        val input_file
        val script_args
    output:
        // Some scripts (e.g., git_sync) don't produce parquet files; make the output optional
        path "*.parquet", emit: 'parquet', optional: true
    script:
        """
        ${python_exe} -u ${script_name} ${input_file} ${script_args}
        """
}