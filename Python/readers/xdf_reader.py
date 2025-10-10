import pyxdf, polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python xdf_reader.py <input_xdf> <output_dir>") or sys.exit(1)
    get_output_filename = lambda base, idx: f"{base}_xdf_stream{idx+1}.parquet"
    run = lambda input_xdf, output_dir: (
        print(f"[Nextflow] XDF Reader started for: {input_xdf}") or
        (lambda folder:
            print(f"[Nextflow] Loading XDF file: {input_xdf}") or
            (lambda streams:
                print(f"[Nextflow] Found {len(streams)} streams in XDF.") or
                (len(streams) > 0 and (
                    os.makedirs(folder, exist_ok=True) or
                    [
                        (lambda idx, stream:
                            print(f"[Nextflow] Processing stream {idx+1}/{len(streams)}: '{stream['info']['name'][0] if 'name' in stream['info'] and len(stream['info']['name']) > 0 else 'Unknown'}'") or
                            (lambda df:
                                df.write_parquet(os.path.join(folder, get_output_filename(os.path.splitext(os.path.basename(input_xdf))[0], idx))) or
                                print(f"[Nextflow] Parquet file saved: {os.path.join(folder, get_output_filename(os.path.splitext(os.path.basename(input_xdf))[0], idx))} with shape {df.shape}")
                            )(pl.DataFrame(stream['time_series']) if len(stream['time_series']) > 0 else pl.DataFrame([]))
                        )(idx, stream)
                        for idx, stream in enumerate(streams)
                    ] or
                    print(f"[Nextflow] XDF reading finished. Output dir: {folder}")
                ) or print(f"[Nextflow] No streams found in XDF file: {input_xdf}"))
            )(pyxdf.load_xdf(input_xdf)[0])
        )(f"{os.path.splitext(os.path.basename(input_xdf))[0]}_xdf")
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
        print(f"[Nextflow] XDF Reader errored. Error: {e}")
        sys.exit(1)
