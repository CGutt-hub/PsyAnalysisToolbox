import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python txt_reader.py <input_txt> <output_parquet>") or sys.exit(1)
    # Lambda: main TXT reading logic, maximally nested
    run = lambda input_txt, output_parquet: (
        # Lambda: print start message
        print(f"[Nextflow] TXT reading started for: {input_txt}") or (
            # Lambda: try reading TXT with different delimiters
            (lambda df:
                # Lambda: check if DataFrame is valid
                (lambda d:
                    # Lambda: print DataFrame shape
                    print(f"[Nextflow] Loaded TXT DataFrame shape: {d.shape}") or (
                        # Lambda: write DataFrame to Parquet
                        (lambda _: d.write_parquet(output_parquet))(d) or (
                            # Lambda: print finished message
                            print(f"[Nextflow] TXT reading finished. Output: {output_parquet}")
                        )
                    )
                )(df) if df is not None else (
                    # Lambda: print error and exit if parsing failed
                    print(f"[Nextflow] TXT reading errored. Could not parse TXT: {input_txt}") or sys.exit(1)
                )
            )(
                # Lambda: try each separator and return first successful DataFrame
                (lambda input_txt:
                    (lambda seps:
                        next((
                            # Lambda: try reading with separator
                            (lambda sep:
                                (lambda d:
                                    d if d is not None else None
                                )(pl.read_csv(input_txt, separator=sep))
                            )(sep)
                            for sep in seps
                            if (lambda s: True)(sep)
                        ), None)
                    )(['\t', ',', ';', '|', ' '])
                )(input_txt)
            )
        )
    )
    try:
        args = sys.argv
        # Lambda: check argument count and run main logic
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] TXT reading errored. Error: {e}")
        sys.exit(1)