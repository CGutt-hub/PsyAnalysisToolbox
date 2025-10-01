import pyxdf, polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python xdf_reader.py <input_xdf> <output_dir>") or sys.exit(1)
    run = lambda input_xdf, output_dir: (
        print(f"[Nextflow] XDF Reader started for: {input_xdf}"),
        (lambda streams: (
            print(f"[Nextflow] Found {len(streams)} streams in XDF."),
            [
                (lambda stream: (
                    (lambda name, df: (
                        print(f"[Nextflow] Writing stream '{name}' shape: {df.shape} to {os.path.join(output_dir, name + '.parquet')}") ,
                        df.write_parquet(os.path.join(output_dir, name + '.parquet')),
                        print(f"[Nextflow] Parquet file saved: {os.path.join(output_dir, name + '.parquet')}")
                    ))(stream['info']['name'][0], pl.DataFrame(stream['time_series']) if len(stream['time_series']) > 0 else pl.DataFrame([]))
                ))(stream)
                for stream in streams
            ],
            print(f"[Nextflow] XDF reading finished. Output dir: {output_dir}")
        ))(pyxdf.load_xdf(input_xdf)[0])
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] XDF Reader errored. Error: {e}")
        sys.exit(1)
