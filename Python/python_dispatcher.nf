process python_dispatcher {
    input:
        val python_exe
        val script_name
        val script_args
        val output_name
    output:
        path output_name
    script:
        """
        ${python_exe} ${script_name} ${script_args}
        """
}