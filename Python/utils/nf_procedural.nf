// Generic Nextflow procedural infrastructure for Python-based pipelines
// Handles output directory creation, log management, and IOInterface process
// Includes branch watchdog for automatic finalization when all branches complete

import java.util.concurrent.locks.ReentrantLock

// Global lock for git operations - prevents race conditions when multiple participants finish simultaneously
@groovy.transform.Field
def git_lock = new ReentrantLock()

// Workflow wrapper: discovers participants, creates output folders, and starts watchdog
// Supports continuous mode via watchPath for new participants
// Watchdog monitors trace file and triggers finalization + git sync when all branches terminate
workflow workflow_wrapper {
    take:
        input_dir
        output_dir
        participant_pattern
        terminal_processes    // Comma-separated list of terminal process names (branch endpoints)
    
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
        
        // Create folders, start watchdog, and emit tuple [participant_id, output_folder]
        participant_context = all_participants.map { pid ->
            def safe_id = pid.replaceAll('\r', '').trim().replaceAll('[^A-Za-z0-9._-]', '_')
            def folder_path = new File("${workflow.launchDir}/${output_dir}/${safe_id}")
            folder_path.mkdirs()
            
            // Create pipeline.log
            def log_file = new File(folder_path, "pipeline.log")
            log_file.text = "=== Pipeline started: ${new Date().format('yyyy-MM-dd HH:mm:ss')} for ${safe_id} ===\n"
            
            // Start watchdog as background process
            def trace_file = "${workflow.launchDir}/pipeline_trace.txt"
            def launch_dir = workflow.launchDir
            def terminals = terminal_processes
            
            Thread.start {
                watchdog_monitor(safe_id, folder_path.absolutePath, trace_file, launch_dir, terminals)
            }
            
            def folder = "${output_dir}/${safe_id}"
            [pid, folder] // Return tuple [participant_id, output_folder]
        }
    
    emit:
        participant_context  // Emits [participant_id, output_folder] tuples
}

// Watchdog: monitors trace file for participant completion, then finalizes
def watchdog_monitor(String participant_id, String output_folder, String trace_file, 
                     String launch_dir, String terminal_processes) {
    def terminals = terminal_processes.split(',').collect { it.trim() } as Set
    def poll_interval = 5000  // 5 seconds
    def stable_threshold = 3  // Polls with no change before declaring done (when failures present)
    def stable_count = 0
    def last_task_count = 0
    
    println "[watchdog] Started for ${participant_id}, watching for terminals: ${terminals}"
    
    while (true) {
        try {
            def trace = new File(trace_file)
            if (!trace.exists()) {
                Thread.sleep(poll_interval)
                continue
            }
            
            // Parse trace file for this participant's tasks
            def lines = trace.readLines()
            if (lines.size() < 2) {
                Thread.sleep(poll_interval)
                continue
            }
            
            def header = lines[0].split('\t')
            def statusIdx = header.findIndexOf { it == 'status' }
            def nameIdx = header.findIndexOf { it == 'name' }
            def processIdx = header.findIndexOf { it == 'process' }
            
            def participant_tasks = lines[1..-1].findAll { line ->
                line.contains(participant_id)
            }.collect { line ->
                def cols = line.split('\t')
                [
                    process: processIdx >= 0 && cols.size() > processIdx ? cols[processIdx].split(':')[-1].trim() : '',
                    name: nameIdx >= 0 && cols.size() > nameIdx ? cols[nameIdx] : '',
                    status: statusIdx >= 0 && cols.size() > statusIdx ? cols[statusIdx].trim() : ''
                ]
            }
            
            if (participant_tasks.isEmpty()) {
                Thread.sleep(poll_interval)
                continue
            }
            
            // Analyze branch completion
            def completed_terminals = participant_tasks.findAll { 
                it.status == 'COMPLETED' && it.process in terminals 
            }.collect { it.process } as Set
            
            def failed_processes = participant_tasks.findAll { 
                it.status in ['FAILED', 'ABORTED'] 
            }.collect { it.process } as Set
            
            // Track stability (no new tasks)
            def current_count = participant_tasks.size()
            if (current_count == last_task_count) {
                stable_count++
            } else {
                stable_count = 0
                last_task_count = current_count
            }
            
            // Check if done: all terminals completed, OR failures + stable (no more tasks coming)
            def all_done = completed_terminals.size() == terminals.size()
            def failed_and_stable = !failed_processes.isEmpty() && stable_count >= stable_threshold
            
            if (all_done || failed_and_stable) {
                // Finalize
                watchdog_finalize(participant_id, output_folder, launch_dir, 
                                  completed_terminals, failed_processes, terminals.size())
                break
            }
            
            Thread.sleep(poll_interval)
            
        } catch (Exception e) {
            println "[watchdog] Error for ${participant_id}: ${e.message}"
            Thread.sleep(poll_interval)
        }
    }
}

// Finalize: write completion log and git sync
def watchdog_finalize(String participant_id, String output_folder, String launch_dir,
                      Set completed, Set failed, int total_terminals) {
    def timestamp = new Date().format('yyyy-MM-dd HH:mm:ss')
    def log_file = new File(output_folder, 'pipeline.log')
    
    // Append completion to log
    log_file.append("\n=== Pipeline completed: ${timestamp} for ${participant_id} ===\n")
    log_file.append("=== Branches: ${completed.size()}/${total_terminals} succeeded ===\n")
    if (!failed.isEmpty()) {
        log_file.append("=== Failed processes: ${failed.sort().join(', ')} ===\n")
    }
    
    // Git sync - acquire lock to prevent concurrent git operations from multiple watchdogs
    git_lock.lock()
    try {
        def git_check = ["git", "-C", launch_dir, "status", "--porcelain"].execute()
        git_check.waitFor()
        def changes = git_check.text.trim()
        
        if (changes) {
            ["git", "-C", launch_dir, "add", "."].execute().waitFor()
            def commit_msg = "Results: ${participant_id} (${completed.size()}/${total_terminals}) ${new Date().format('yyyy-MM-dd_HH-mm-ss')}"
            ["git", "-C", launch_dir, "commit", "-m", commit_msg].execute().waitFor()
            ["git", "-C", launch_dir, "push"].execute().waitFor()
            log_file.append("${timestamp} [git_sync] Synced: ${commit_msg}\n")
        } else {
            log_file.append("${timestamp} [git_sync] No changes to commit\n")
        }
    } catch (Exception e) {
        log_file.append("${timestamp} [git_sync] Error: ${e.message}\n")
    } finally {
        git_lock.unlock()
    }
    
    println "[watchdog] Finalized ${participant_id}: ${completed.size()}/${total_terminals} branches succeeded"
}

// Generic IOInterface: exe script input params
// Language-agnostic CLI wrapper for any executable (Python, Java, Rust, etc.)
// Watchdog monitors trace file for failures, so no need for .failed markers
process IOInterface {
    input:
        val env_exe             // Executable path (python, java, rust binary, etc.)
        val script              // Script path relative to workflow.launchDir
        path input              // Input file(s) - automatically staged by Nextflow
        val extraParams         // Additional arguments (renamed to avoid shadowing Nextflow params)

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
    if (extraParams && extraParams.toString().trim() != "") {
        def paramStr = extraParams.toString().trim()
        
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
    
    // Extract script name for logging
    def scriptName = script.toString().tokenize('/').last().replace('.py', '')
    
    """
    #!/bin/bash
    
    # Extract participant ID from input filename (pattern like EV_002_*)
    INPUT_FILE=\$(basename "${inputArgs}" | sed "s/'//g")
    PARTICIPANT_ID=\$(echo "\$INPUT_FILE" | grep -oE '^[A-Za-z]+_[0-9]+' | head -1)
    
    # Run processing with logging
    if [ -n "\$PARTICIPANT_ID" ]; then
        LOG_FILE="${workflow.launchDir}/${params.output_dir}/\${PARTICIPANT_ID}/pipeline.log"
        TEMP_OUT=\$(mktemp)
        ${env_exe} -u "${workflow.launchDir}/${script}" ${inputArgs} ${extraArgs} 2>&1 | tee "\$TEMP_OUT"
        EXIT_CODE=\${PIPESTATUS[0]}
        
        # Add timestamp to each line before appending to log
        while IFS= read -r line; do
            echo "\$(date '+%Y-%m-%d %H:%M:%S') \$line" >> "\$LOG_FILE"
        done < "\$TEMP_OUT"
        rm -f "\$TEMP_OUT"
        
        if [ \$EXIT_CODE -ne 0 ]; then
            echo "" >> "\$LOG_FILE"
            echo "\$(date '+%Y-%m-%d %H:%M:%S') [ERROR] ${scriptName} exit code \$EXIT_CODE" >> "\$LOG_FILE"
        fi
        exit \$EXIT_CODE
    else
        ${env_exe} -u "${workflow.launchDir}/${script}" ${inputArgs} ${extraArgs}
    fi
    """
}