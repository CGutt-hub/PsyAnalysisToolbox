process groupLevel_preprocessor {
    input: 
        path input_dir
    output: 
        path group_out
    script: 
        """
        python groupLevel_preprocessor.py $input_dir $group_out
        """
}
