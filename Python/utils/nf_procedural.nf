// Generic Nextflow procedural infrastructure for Python-based pipelines
// Handles output directory creation, log management, and python_dispatcher process
// Continuous workflow wrapper: discovers participants and creates output folders
// Runs continuously, watching for new participants
workflow workflow_wrapper {
    take:
        input_dir
        output_dir
        participant_pattern
    
    main:
        // Convert glob pattern to regex
        def regex_pattern = participant_pattern.replaceAll(/\*/, '.*').replaceAll(/\?/, '.')
        def input_path = new File("${workflow.launchDir}/${input_dir}")
        def output_path = new File("${workflow.launchDir}/${output_dir}")
        
        // Initial discovery of existing participants
        def output_dirs = output_path.exists() ? output_path.list() as Set : [] as Set
        def new_participants = input_path.list().findAll { it.matches(regex_pattern) }.findAll { !(it in output_dirs) }
        
        // Watch for new participant directories continuously
        watched_participants = Channel
            .watchPath("${workflow.launchDir}/${input_dir}/*", 'create,modify')
            .map { path -> path.getName() }
            .filter { it.matches(regex_pattern) }
            .unique()
        
        // Combine initial discovery with watched directories
        all_participants = Channel.fromList(new_participants).concat(watched_participants)
            .filter { pid ->
                def safe_id = pid.replaceAll('\r', '').trim().replaceAll('[^A-Za-z0-9._-]', '_')
                def out_folder = new File("${workflow.launchDir}/${output_dir}/${safe_id}")
                !out_folder.exists() // Only process if output folder doesn't exist yet
            }
        
        // Create folders and emit tuple [participant_id, output_folder]
        // This ensures they're always synchronized
        participant_context = all_participants.map { pid ->
            def safe_id = pid.replaceAll('\r', '').trim().replaceAll('[^A-Za-z0-9._-]', '_')
            def folder_path = new File("${workflow.launchDir}/${output_dir}/${safe_id}")
            folder_path.mkdirs()
            def folder = "${output_dir}/${safe_id}"
            [pid, folder] // Return tuple [participant_id, output_folder]
        }
    
    emit:
        participant_context  // Emits [participant_id, output_folder] tuples
}

// Process to finalize a participant after all processing completes
// Consolidates logs and syncs to git - triggered automatically by dependency graph
process finalize_participant {
    maxForks 1  // Serialize git operations to avoid lock conflicts
    
    input:
        val participant_id
        val output_folder
        val trigger  // Final output - ensures this runs after all processing
    
    script:
    """
    #!/bin/bash
    set -e
    
    participant="${participant_id}"
    output_dir="${workflow.launchDir}/${output_folder}"
    log_file="\${output_dir}/module_logs.txt"
    
    echo "=== Finalizing ${participant_id} ==="
    
    # Find and consolidate logs
    find "${workflow.workDir}" -type f -name ".command.log" -path "*\${participant}*" -print0 | 
        sort -z | 
        xargs -0 cat > "\${log_file}" 2>/dev/null || true
    
    echo "Logs consolidated: \${log_file}"
    
    # Git sync
    cd "${workflow.launchDir}"
    git add .
    git commit -m "Update results for ${participant_id}: \$(date +%Y-%m-%d_%H-%M-%S)" || true
    git push || true
    
    echo "Finalization complete for ${participant_id}"
    """
}

// Generic IOInterface: exe script input params
// Language-agnostic CLI wrapper for any executable (Python, Java, Rust, etc.)
process IOInterface {
    input:
        val env_exe             // Executable path (python, java, rust binary, etc.)
        val script              // Script path relative to workflow.launchDir
        path input              // Input file(s) - automatically staged by Nextflow
        val params              // Additional arguments

    output:
        path "*.{fif,parquet}"

    script:
    // Shell-escape single quotes by replacing ' with '\''
    def escapeArg = { arg -> arg.toString().replace("'", "'\\''") }
    
    // Format inputs: convert to quoted shell arguments
    def inputArgs = input instanceof Collection 
        ? input.collect { "'${escapeArg(it)}'" }.join(' ')
        : "'${escapeArg(input)}'"
    
    // Format additional args: smart split that preserves bracketed/braced structures
    def extraArgs = ""
    if (params && params.toString().trim() != "") {
        def paramStr = params.toString().trim()
        
        // Parse arguments respecting brackets and braces
        def args = []
        def currentArg = new StringBuilder()
        def depth = 0
        def inQuote = false
        
        for (int i = 0; i < paramStr.length(); i++) {
            def c = paramStr.charAt(i)
            
            if (c == '"' as char || c == "'" as char) {
                inQuote = !inQuote
                currentArg.append(c)
            } else if (!inQuote) {
                if (c == '[' as char || c == '{' as char) {
                    depth++
                    currentArg.append(c)
                } else if (c == ']' as char || c == '}' as char) {
                    depth--
                    currentArg.append(c)
                } else if (c == ' ' as char && depth == 0) {
                    if (currentArg.length() > 0) {
                        args.add(currentArg.toString())
                        currentArg = new StringBuilder()
                    }
                } else {
                    currentArg.append(c)
                }
            } else {
                currentArg.append(c)
            }
        }
        
        if (currentArg.length() > 0) {
            args.add(currentArg.toString())
        }
        
        extraArgs = args.collect { "'${escapeArg(it)}'" }.join(' ')
    }
    
    """
    ${env_exe} "${workflow.launchDir}/${script}" ${inputArgs} ${extraArgs}
    """
}

