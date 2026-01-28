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

// Track participant processes from trace file (shared across watchdog iterations)
@groovy.transform.Field
def participant_processes = new ConcurrentHashMap<String, ConcurrentHashMap<String, String>>()

// Workflow wrapper: discovers participants and creates output folders
// Starts a trace-watching thread that finalizes participants when all processes complete
workflow workflow_wrapper {
    take:
        input_dir
        output_dir
        participant_pattern
    
    main:
        // Trace-watching thread: finalize participants when all their processes complete
        def trace_file = new File("${workflow.launchDir}/pipeline_trace.txt")
        def results_path = "${workflow.launchDir}/${output_dir}"
        
        Thread.start {
            def seen = [] as Set
            while (true) {
                Thread.sleep(5000)
                if (!trace_file.exists()) continue
                try {
                    def procs = [:].withDefault { [:] }  // pid -> [process: status]
                    trace_file.eachLine { line ->
                        // Trace format: task_id \t hash \t native_id \t process \t tag \t name \t status ...
                        def cols = line.split('\t')
                        if (cols.size() < 7 || !cols[0].isNumber()) return
                        def proc = cols[3], tag = cols[4], status = cols[6]
                        def pid = (tag =~ /([A-Za-z]+_\d+)/)?[0]?[1]
                        if (pid) procs[pid][proc] = status
                    }
                    procs.each { pid, jobs ->
                        if (pid in seen || pid in finalized_participants) return
                        if (jobs.size() < 5 || jobs.any { it.value in ['SUBMITTED','RUNNING','PENDING'] }) return
                        seen << pid
                        finalized_participants.add(pid)
                        def ok = jobs.count { it.value in ['COMPLETED','CACHED'] }
                        def fail = jobs.count { it.value == 'FAILED' }
                        new File("${results_path}/${pid}/${pid}_pipeline.log") << "[${new Date().format('yyyy-MM-dd HH:mm:ss')}] Finalized: ${ok} ok, ${fail} failed\n"
                        finalize_participant(pid, results_path, jobs, jobs.size(), git_lock)
                    }
                } catch (e) {}
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
    
    // Debug: Log finalization attempt to participant's log
    if (log_file.exists()) {
        log_file.append("[${timestamp}] [finalize] Starting finalization for ${pid}\n")
    }
    
    def completed = branch_status.findAll { it.value == 'COMPLETED' }.keySet().toList()
    def failed = branch_status.findAll { it.value == 'FAILED' }.keySet().toList()
    def cached = branch_status.findAll { it.value == 'CACHED' }.keySet().toList()
    def status = failed.isEmpty() ? 'SUCCESS' : 'PARTIAL'
    
    // Write completion summary to participant's log
    if (log_file.exists()) {
        log_file.append("\n=== Pipeline ${status}: ${timestamp} ===\n")
        log_file.append("=== Branches completed: ${completed.size() + cached.size()}/${total_terminals} ===\n")
        log_file.append("=== Completed: ${completed.sort().join(', ')} ===\n")
        if (cached) {
            log_file.append("=== Cached: ${cached.sort().join(', ')} ===\n")
        }
        if (failed) {
            log_file.append("=== FAILED: ${failed.sort().join(', ')} ===\n")
        }
        log_file.append("=== Pipeline finished: ${timestamp} for ${pid} ===\n\n")
    }
    
    // Helper to run git command and capture output to log file
    def runGit = { List<String> cmd, File logFile ->
        def proc = cmd.execute()
        def stdout = new StringBuilder()
        def stderr = new StringBuilder()
        proc.consumeProcessOutput(stdout, stderr)
        proc.waitFor()
        def ts = new java.text.SimpleDateFormat('yyyy-MM-dd HH:mm:ss').format(new Date())
        if (stdout.toString().trim()) {
            logFile.append("[${ts}] [finalize] git stdout: ${stdout.toString().trim()}\n")
        }
        if (stderr.toString().trim()) {
            logFile.append("[${ts}] [finalize] git stderr: ${stderr.toString().trim()}\n")
        }
        return [exitCode: proc.exitValue(), stdout: stdout.toString(), stderr: stderr.toString()]
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
            if (log_file.exists()) {
                log_file.append("[${timestamp}] [finalize] Git error: No git repository found above ${results_path}\n")
            }
            return
        }
        
        def git_root_path = git_root.getAbsolutePath()
        if (log_file.exists()) {
            log_file.append("[${timestamp}] [finalize] Git repo root: ${git_root_path}\n")
        }
        
        // Check git status
        def statusResult = runGit(["git", "-C", git_root_path, "status", "--porcelain"], log_file)
        log_file.append("[${timestamp}] [finalize] Git status check (exit: ${statusResult.exitCode})\n")
        
        if (statusResult.stdout.trim()) {
            log_file.append("[${timestamp}] [finalize] Changes detected, starting sync...\n")
            
            // Pull first to avoid push conflicts (rebase to keep history clean)
            log_file.append("[${timestamp}] [finalize] Pulling latest changes...\n")
            def pullResult = runGit(["git", "-C", git_root_path, "pull", "--rebase", "--autostash"], log_file)
            log_file.append("[${timestamp}] [finalize] Git pull (exit: ${pullResult.exitCode})\n")
            
            if (pullResult.exitCode != 0) {
                log_file.append("[${timestamp}] [finalize] Warning: Pull failed, attempting to continue...\n")
                // Try to abort rebase if stuck
                runGit(["git", "-C", git_root_path, "rebase", "--abort"], log_file)
            }
            
            // Stage all changes
            def addResult = runGit(["git", "-C", git_root_path, "add", "."], log_file)
            log_file.append("[${timestamp}] [finalize] Git add (exit: ${addResult.exitCode})\n")
            
            // Commit
            def msg = "${status}: ${pid} (${completed.size() + cached.size()}/${total_terminals})"
            if (failed) msg += " - failed: ${failed.join(',')}"
            def commitResult = runGit(["git", "-C", git_root_path, "commit", "-m", msg], log_file)
            log_file.append("[${timestamp}] [finalize] Git commit (exit: ${commitResult.exitCode})\n")
            
            if (commitResult.exitCode == 0 || commitResult.stderr.contains("nothing to commit")) {
                // Push with retry
                def pushResult = runGit(["git", "-C", git_root_path, "push"], log_file)
                log_file.append("[${timestamp}] [finalize] Git push (exit: ${pushResult.exitCode})\n")
                
                if (pushResult.exitCode != 0) {
                    // Retry: pull and push again
                    log_file.append("[${timestamp}] [finalize] Push failed, retrying with pull...\n")
                    runGit(["git", "-C", git_root_path, "pull", "--rebase", "--autostash"], log_file)
                    pushResult = runGit(["git", "-C", git_root_path, "push"], log_file)
                    log_file.append("[${timestamp}] [finalize] Git push retry (exit: ${pushResult.exitCode})\n")
                }
                // Note: Final sync status not logged here to avoid unsynced state in log file
            } else {
                log_file.append("[${timestamp}] [finalize] Git commit failed, skipping push\n")
            }
        } else {
            log_file.append("[${timestamp}] [finalize] No changes to sync\n")
        }
    } catch (Exception e) {
        if (log_file.exists()) {
            log_file.append("[${timestamp}] [finalize] Git error: ${e.message}\n")
            log_file.append("[${timestamp}] [finalize] Stack trace: ${e.getStackTrace().take(5).join('\\n')}\n")
        }
    } finally {
        lock.unlock()
    }
}

// Generic IOInterface: exe script input params
// Language-agnostic CLI wrapper for any executable (Python, Java, Rust, etc.)
// Watchdog monitors trace file for failures, so no need for .failed markers
process IOInterface {
    tag "${input}"              // Tag with input filename for trace file (enables watchdog tracking)
    
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