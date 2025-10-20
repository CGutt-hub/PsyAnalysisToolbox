process python_dispatcher {
    debug params.debug_mode ?: false
    errorStrategy 'finish'
    input:
        val python_exe
        val script_name
        val input_file
        val script_args
    output:
        path "*.parquet", emit: 'parquet', optional: true, type: 'file'
    script:
        """
        ${python_exe} -u ${script_name} ${input_file} ${script_args}
        """
}