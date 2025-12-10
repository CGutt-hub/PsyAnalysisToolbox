import os, sys, fnmatch
get_subdir = lambda f: os.path.join(os.path.dirname(f), os.path.splitext(os.path.basename(f))[0])
find = lambda d, p: [os.path.join(d, f) for f in os.listdir(d) if fnmatch.fnmatch(f, p)] if os.path.isdir(d) else []
run = lambda s, pat: (lambda m: (print(*m, sep='\n'), 0)[1] if m else (print(""), 1)[1])(find(get_subdir(s), pat))

if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(run(a[0], a[1]) if len(a) == 2 else (print("Usage: python find_stream_file.py <signal_file> <pattern>"), 1)[1])
