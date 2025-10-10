process python_dispatcher {
    echo true  
    input:
        val python_exe
        val script_name
        val input_file
        val script_args
    output:
        path "${input_file.toString().tokenize('/')[-1].replaceFirst(/\.[^.]+$/, '')}_${script_name.toString().tokenize('/')[-1].replaceFirst(/\.py$/, '').split('_')[0]}*"
    script:
        def input_stem = input_file.toString().tokenize('/')[-1].replaceFirst(/\.[^.]+$/, '')
        def script_base = script_name.toString().tokenize('/')[-1].replaceFirst(/\.py$/, '').split('_')[0]
        def output_prefix = "${input_stem}_${script_base}"
        """
        ${python_exe} -u ${script_name} ${input_file} '${script_args}'
        """
}