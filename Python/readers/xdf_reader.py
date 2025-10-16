import pyxdf, polars as pl, sys, os, tempfile
if __name__ == "__main__":
    usage = lambda: print("[READER] Usage: python xdf_reader.py <input_xdf> <output_dir>") or sys.exit(1)
    get_output_filename = lambda base, idx: f"{base}_xdf{idx+1}.parquet"
    run = lambda input_xdf, output_dir: (
        print(f"[READER] Started for: {input_xdf}") or
        (lambda streams:
            print(f"[READER] Loading XDF file: {input_xdf}") or
            (len(streams) > 0 and (
                print(f"[READER] Found {len(streams)} streams.") or
                [
                    (lambda idx, stream:
                        print(f"[READER] Processing stream {idx+1}/{len(streams)}: '{stream['info']['name'][0] if 'name' in stream['info'] and len(stream['info']['name']) > 0 else 'Unknown'}'") or
                        (lambda df:
                            (lambda temp_file:
                                df.write_parquet(temp_file) or
                                os.rename(temp_file, get_output_filename(os.path.splitext(os.path.basename(input_xdf))[0], idx)) or
                                print(f"[READER] Parquet file saved: {get_output_filename(os.path.splitext(os.path.basename(input_xdf))[0], idx)} with shape {df.shape}")
                            )(tempfile.mktemp(suffix='.parquet', prefix=f"xdf_temp_{idx}_", dir="."))
                        )(pl.DataFrame(stream['time_series']) if len(stream['time_series']) > 0 else pl.DataFrame([]))
                    )(idx, stream)
                    for idx, stream in enumerate(streams)
                ] or
                print(f"[READER] Reading finished. Files created in current directory")
            ) or print(f"[READER] No streams found in XDF file: {input_xdf}"))
        )(pyxdf.load_xdf(input_xdf)[0])
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_xdf = args[1]
            output_dir = args[2]
            run(input_xdf, output_dir)
    except Exception as e:
        print(f"[READER] Error: {e}")
        sys.exit(1)
