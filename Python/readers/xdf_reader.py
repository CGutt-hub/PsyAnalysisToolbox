import pyxdf, polars as pl, sys, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python xdf_reader.py <input_xdf> <output_dir>") or sys.exit(1)
    # Lambda: main XDF reading logic, maximally nested
    run = lambda input_xdf, output_dir: (
        # Lambda: print start message
        print(f"[Nextflow] XDF reading started for: {input_xdf}") or (
            # Lambda: read XDF and process streams
            (lambda streams:
                # Lambda: print number of streams
                print(f"[Nextflow] Found {len(streams)} streams in XDF.") or [
                    # Lambda: process each stream
                    (lambda stream:
                        # Lambda: extract name and DataFrame from stream
                        (lambda name, df:
                            # Lambda: print stream info and write Parquet
                            print(f"[Nextflow] Writing stream '{name}' shape: {df.shape} to {os.path.join(output_dir, name + '.parquet')}") or (
                                # Lambda: write DataFrame to Parquet
                                (lambda _: df.write_parquet(os.path.join(output_dir, name + '.parquet')))(df)
                            )
                        )(stream['info']['name'][0], pl.DataFrame(stream['time_series']) if len(stream['time_series']) > 0 else pl.DataFrame([]))
                    )(stream)
                    for stream in streams
                ] or (
                    # Lambda: print finished message
                    print(f"[Nextflow] XDF reading finished. Output dir: {output_dir}")
                )
            )(pyxdf.load_xdf(input_xdf)[0])
        )
    )
    try:
        args = sys.argv
        # Lambda: check argument count and run main logic
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] XDF reading errored. Error: {e}")
        sys.exit(1)
