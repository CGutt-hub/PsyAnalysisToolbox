import pyxdf, polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python xdf_reader.py <input_xdf> <output_dir>") or sys.exit(1)
    get_output_filename = lambda input_file, idx: f"{os.path.splitext(os.path.basename(input_file))[0]}_stream{idx+1}_xdf.parquet"
    args = sys.argv
    (lambda a:
        usage() if len(a) < 3 else
        (lambda input_xdf, output_dir:
            print(f"[Nextflow] XDF Reader started for: {input_xdf}") or
            (lambda streams, folder:
                print(f"[Nextflow] Found {len(streams)} streams in XDF.") or
                os.makedirs(folder, exist_ok=True) or
                [
                    (lambda idx, stream:
                        print(f"[Nextflow] Writing stream '{stream['info']['name'][0]}' shape: {pl.DataFrame(stream['time_series']).shape if len(stream['time_series']) > 0 else (0,0)} to {get_output_filename(input_xdf, idx)}") or
                        (lambda df:
                            df.write_parquet(get_output_filename(input_xdf, idx)) or
                            print(f"[Nextflow] Parquet file saved: {get_output_filename(input_xdf, idx)}")
                        )(pl.DataFrame(stream['time_series']) if len(stream['time_series']) > 0 else pl.DataFrame([]))
                    )(idx, stream)
                    for idx, stream in enumerate(streams)
                ] or
                print(f"[Nextflow] XDF reading finished. Output dir: {folder}")
            )(pyxdf.load_xdf(input_xdf)[0], os.path.join(output_dir, "streams"))
        )(a[1], a[2])
    )(args)
