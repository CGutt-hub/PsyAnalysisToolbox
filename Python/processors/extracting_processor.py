import polars as pl, sys, os

# usage lambda
usage = lambda: (print('[PROC] Usage: python extracting_processor.py <input_parquet> <selector1> [selector2 ...]') or sys.exit(1))

get_output_filename = lambda base: f"{base}_extr.parquet"



# Use lambdas for concise logic, named functions for multi-step logic
def resolver(s, dcols):
    if s is None or s == '':
        return []
    if ':' in s:
        parts = s.split(':')
        start = 0 if parts[0].strip() == '' else (int(parts[0].strip()) - 1 if not parts[0].strip().startswith('-') else len(dcols) + int(parts[0].strip()))
        end = len(dcols) - 1 if (len(parts) <= 1 or parts[1].strip() == '') else (int(parts[1].strip()) - 1 if not parts[1].strip().startswith('-') else len(dcols) + int(parts[1].strip()))
        return dcols[start:end+1] if start <= end else []
    if s.lstrip('-').isdigit():
        idx = int(s)
        if 1 <= idx <= len(dcols):
            return [dcols[idx - 1]]
        if idx < 0 and abs(idx) <= len(dcols):
            return [dcols[len(dcols) + idx]]
    name = s.lower()
    return ([c for c in dcols if c.lower() == name] or
            [c for c in dcols if name in c.lower()] or
            [c for c in dcols if c.lower().startswith(name)] or [])

write_outputs = lambda df, selectors, base, out_folder: (
    os.makedirs(out_folder, exist_ok=True) or sum([
        (lambda sel_cols, idx, sel: (
            (lambda out_df: (
                out_df.write_parquet(os.path.join(out_folder, f"{base}_extr{idx+1}.parquet")) or
                (print(f"[PROC] Wrote {base}_extr{idx+1}.parquet columns={out_df.columns}") or 1)
            ))(
                df.select(['time'] + sel_cols) if ('time' in df.columns and sel_cols) else (
                    df.select(sel_cols) if sel_cols else pl.DataFrame({'time': df['time'].to_list(), 'empty': [0]*df.height})
                )
            )
        ))(resolver(sel, (df.columns[1:] if df.columns and df.columns[0].lower() == 'time' else df.columns)), idx, sel)
        for idx, sel in enumerate(selectors)
    ], 0)
)

write_signal = lambda input_parquet, base, out_folder, writes_count: (
    pl.DataFrame({'signal':[1], 'source':[os.path.basename(input_parquet)], 'streams':[writes_count]})
        .write_parquet(os.path.join(out_folder, get_output_filename(base))) or
    print(f"[PROC] Extraction finished. Wrote signal {os.path.join(out_folder, get_output_filename(base))} with {writes_count} streams")
)

run = lambda input_parquet, selectors: (
    print(f"[PROC] Extracting started for: {input_parquet} selectors={selectors}") or
    (lambda df:
        (lambda base, out_folder:
            (lambda writes_count:
                write_signal(input_parquet, base, out_folder, writes_count)
            )(write_outputs(df, selectors, base, out_folder))
        )(os.path.splitext(os.path.basename(input_parquet))[0], os.path.join(os.path.dirname(input_parquet), f"{os.path.splitext(os.path.basename(input_parquet))[0]}_extr"))
    )(pl.read_parquet(input_parquet))
)

if __name__ == '__main__':
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_parquet = args[1]
            selectors = args[2:] if len(args) > 2 else []
            run(input_parquet, selectors)
    except Exception as e:
        print(f"[PROC] Error: {e}")
        sys.exit(1)
