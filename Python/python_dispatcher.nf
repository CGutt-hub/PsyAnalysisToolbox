process python_dispatcher {
    debug params.debug_mode ?: false
    errorStrategy 'finish'
    input:
        val python_exe
        val script_name
        val input_file
        val script_args
    output:
        path "*_*.parquet", type: 'file'
    script:
        def input_stem = input_file.toString().tokenize('/')[-1].replaceFirst(/\.[^.]+$/, '')
        def script_base = script_name.toString().tokenize('/')[-1].replaceFirst(/\.py$/, '').split('_')[0]
        def output_prefix = "${input_stem}_${script_base}"
        """
        ${python_exe} -u ${script_name} ${input_file} ${script_args}
        """
}

process plot_dispatcher {
    debug params.debug_mode ?: false
    errorStrategy 'finish'
    input:
        val python_exe
        val script_name
        val input_file
        val script_args
    output:
        path "*.pdf", type: 'file'
    script:
        def input_stem = input_file.toString().tokenize('/')[-1].replaceFirst(/\.[^.]+$/, '')
        def script_base = script_name.toString().tokenize('/')[-1].replaceFirst(/\.py$/, '').split('_')[0]
        def output_prefix = "${input_stem}_${script_base}"
        """
        ${python_exe} -u ${script_name} ${input_file} ${script_args} ${output_prefix}
        """
}