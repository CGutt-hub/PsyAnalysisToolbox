// Generic Nextflow procedural infrastructure for Python-based pipelines
// Handles output directory creation, log management, and trace-based watchdog finalization
// Watchdog monitors Nextflow trace file to detect when all terminal processes complete

import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.ConcurrentHashMap

// Global lock for git operations - prevents race conditions when multiple participants finish simultaneously
@groovy.transform.Field
def git_lock = new ReentrantLock()

// Track finalized participants to avoid duplicate finalization
@groovy.transform.Field
def finalized_participants = ConcurrentHashMap.newKeySet()

// Workflow wrapper: discovers participants and creates output folders
// Starts a trace-watching thread that finalizes participants when all their terminals complete
workflow workflow_wrapper {
    take:
        input_dir
        output_dir
        participant_pattern
        terminal_processes    // Comma-separated list of terminal process names (e.g., "panas_plotter,bisbas_plotter,...")
    
    main:
        // Parse terminal process names
        def terminal_list = terminal_processes.split(',').collect { it.trim() }
        def num_terminals = terminal_list.size()
        
        // Start trace-watching thread for finalization
        def trace_file = new File("${workflow.launchDir}/pipeline_trace.txt")
        def results_path = "${workflow.launchDir}/${output_dir}"
        
        Thread.start {
            // Track ALL processes per participant: [pid: [process: status]]
            def participant_processes = new ConcurrentHashMap<String, ConcurrentHashMap<String, String>>()
            def last_activity = new ConcurrentHashMap<String, Long>()  // Last trace activity per participant
            def last_line_count = 0
            def INACTIVITY_MS = 15000  // 15 seconds of no new trace entries after all done
            
            // Terminal statuses (process is done, allows finalization)
            def TERMINAL_STATUSES = ['COMPLETED', 'FAILED', 'ABORTED'] as Set
            // Blocking statuses (process not done, prevents finalization)
            def BLOCKING_STATUSES = ['SUBMITTED', 'RUNNING', 'PENDING', 'CACHED'] as Set
            
            while (true) {
                try {
                    Thread.sleep(5000)  // Check every 5 seconds
                    
                    if (!trace_file.exists()) continue
                    
                    def lines = trace_file.readLines()
                    def now = System.currentTimeMillis()
                    
                    // Check for participants that are done (no running processes + inactivity)
                    last_activity.each { pid, lastTime ->
                        if (finalized_participants.contains(pid)) return  // Already finalized
                        
                        def procs = participant_processes.getOrDefault(pid, [:])
                        def blocking = procs.findAll { it.value in BLOCKING_STATUSES }
                        def completed = procs.findAll { it.value == 'COMPLETED' }
                        def failed = procs.findAll { it.value == 'FAILED' }
                        def aborted = procs.findAll { it.value == 'ABORTED' }
                        
                        // Only finalize if: has processes AND none blocking AND inactive
                        if (procs.size() > 0 && blocking.size() == 0 && (now - lastTime > INACTIVITY_MS)) {
                            if (finalized_participants.add(pid)) {
                                def pid_log = new File("${results_path}/${pid}/${pid}_pipeline.log")
                                if (pid_log.exists()) {
                                    pid_log.append("[${new Date().format('yyyy-MM-dd HH:mm:ss')}] Pipeline complete: ${completed.size()} succeeded, ${failed.size()} failed, ${aborted.size()} skipped\n")
                                }
                                finalize_participant(pid, results_path, procs, procs.size(), git_lock)
                            }
                        }
                    }
                    
                    if (lines.size() <= last_line_count) continue
                    
                    // Parse header to find column indices
                    def header = lines[0].split('\t')
                    def nameIdx = header.findIndexOf { it == 'name' }
                    def statusIdx = header.findIndexOf { it == 'status' }
                    def processIdx = header.findIndexOf { it == 'process' }
                    
                    if (nameIdx < 0 || statusIdx < 0 || processIdx < 0) continue
                    
                    // Process new lines only
                    for (int i = last_line_count ?: 1; i < lines.size(); i++) {
                        def cols = lines[i].split('\t')
                        if (cols.size() <= Math.max(nameIdx, Math.max(statusIdx, processIdx))) continue
                        
                        def taskName = cols[nameIdx]
                        def status = cols[statusIdx]
                        def processName = cols[processIdx]
                        
                        // Extract participant ID from task name (e.g., "panas_plotter (EV_003_panas.parquet)")
                        def pidMatch = taskName =~ /\(([A-Za-z]+_[0-9]+)/
                        if (!pidMatch) continue
                        def pid = pidMatch[0][1]
                        
                        // Track ALL statuses for any process
                        participant_processes.computeIfAbsent(pid, { new ConcurrentHashMap<String, String>() })
                        participant_processes[pid][processName] = status
                        last_activity[pid] = now  // Reset inactivity timer on any trace entry
                        
                        // Write to participant's log file (only for terminal COMPLETED/FAILED)
                        if (processName in terminal_list && status in ['COMPLETED', 'FAILED']) {
                            def pid_log = new File("${results_path}/${pid}/${pid}_pipeline.log")
                            if (pid_log.exists()) {
                                def branch = processName.replace('_plotter', '').replace('_rel', '_rel')
                                pid_log.append("[${new Date().format('yyyy-MM-dd HH:mm:ss')}] ${branch} -> ${status}\n")
                            }
                        }
                    }
                    
                    last_line_count = lines.size()
                    
                } catch (Exception e) {
                    // Ignore errors (file might be locked by Nextflow)
                }
            }
        }
        
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
        participant_context = all_participants.map { pid ->
            def safe_id = pid.replaceAll('\r', '').trim().replaceAll('[^A-Za-z0-9._-]', '_')
            def folder_path = new File("${workflow.launchDir}/${output_dir}/${safe_id}")
            folder_path.mkdirs()
            
            // Create {participant_id}_pipeline.log
            def log_file = new File(folder_path, "${safe_id}_pipeline.log")
            log_file.text = "=== Pipeline started: ${new Date().format('yyyy-MM-dd HH:mm:ss')} for ${safe_id} ===\n"
            
            def folder = "${output_dir}/${safe_id}"
            [pid, folder] // Return tuple [participant_id, output_folder]
        }
    
    emit:
        participant_context  // Emits [participant_id, output_folder] tuples
}

// Finalize a single participant - called by watchdog when all terminals complete
def finalize_participant(String pid, String results_path, Map branch_status, int total_terminals, ReentrantLock lock) {
    def timestamp = new java.text.SimpleDateFormat('yyyy-MM-dd HH:mm:ss').format(new Date())
    def log_file = new File("${results_path}/${pid}/${pid}_pipeline.log")
    
    def completed = branch_status.findAll { it.value == 'COMPLETED' }.keySet().toList()
    def failed = branch_status.findAll { it.value == 'FAILED' }.keySet().toList()
    def status = failed.isEmpty() ? 'SUCCESS' : 'PARTIAL'
    
    // Write completion summary to participant's log
    log_file.append("\n=== Pipeline ${status}: ${timestamp} ===\n")
    log_file.append("=== Branches completed: ${completed.size()}/${total_terminals} ===\n")
    log_file.append("=== Completed: ${completed.sort().join(', ')} ===\n")
    if (failed) {
        log_file.append("=== FAILED: ${failed.sort().join(', ')} ===\n")
    }
    
    // Git sync with lock - find git repo root by going up from results_path
    lock.lock()
    try {
        // Find the git repository root (traverse up from results_path)
        def git_root = new File(results_path)
        while (git_root != null && !new File(git_root, ".git").exists()) {
            git_root = git_root.getParentFile()
        }
        
        if (git_root == null) {
            log_file.append("[${timestamp}] Git error: No git repository found above ${results_path}\n")
            return
        }
        
        def git_root_path = git_root.getAbsolutePath()
        log_file.append("[${timestamp}] Git repo root: ${git_root_path}\n")
        
        def git_status = ["git", "-C", git_root_path, "status", "--porcelain"].execute()
        git_status.waitFor()
        
        if (git_status.text.trim()) {
            ["git", "-C", git_root_path, "add", "."].execute().waitFor()
            def msg = "${status}: ${pid} (${completed.size()}/${total_terminals})"
            if (failed) msg += " - failed: ${failed.join(',')}"
            ["git", "-C", git_root_path, "commit", "-m", msg].execute().waitFor()
            def push_proc = ["git", "-C", git_root_path, "push"].execute()
            push_proc.waitFor()
            log_file.append("[${timestamp}] Git sync complete (exit: ${push_proc.exitValue()})\n")
        }
    } catch (Exception e) {
        log_file.append("[${timestamp}] Git error: ${e.message}\n")
    } finally {
        lock.unlock()
    }
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
        LOG_FILE="${workflow.launchDir}/${params.output_dir}/\${PARTICIPANT_ID}/\${PARTICIPANT_ID}_pipeline.log"
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