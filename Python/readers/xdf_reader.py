import pyxdf, polars as pl, sys, os
if __name__ == "__main__":
    # 1) usage lambda
    usage = lambda: print("[READER] Usage: python xdf_reader.py <input_xdf> <output_dir>") or sys.exit(1)

    # 2) signal output filename lambda (top-level signalling parquet)
    get_signal_filename = lambda base: f"{base}_xdf.parquet"

    # 3) run: inline nested lambda that processes streams and writes per-stream parquets
    run = lambda input_xdf, output_dir: (
        print(f"[READER] Started for: {input_xdf}") or
        (lambda streams: (
            print(f"[READER] Loading XDF file: {input_xdf}") or
            (len(streams) > 0 and (
                print(f"[READER] Found {len(streams)} streams.") or
                (lambda base: (
                    os.makedirs(os.path.join(output_dir, f"{base}_xdf"), exist_ok=True) or
                    ([
                        (lambda idx, stream: (
                            (lambda info: (
                                (lambda name: (
                                    print(f"[READER] Processing stream {idx+1}/{len(streams)}: '{name}'") or
                                    (lambda df: (
                                        df.write_parquet(os.path.join(output_dir, f"{base}_xdf", f"{base}_xdf{idx+1}.parquet")) or
                                        print(f"[READER] Parquet file saved: {base}_xdf{idx+1}.parquet with shape {df.shape}")
                                    ))(pl.DataFrame(stream.get('time_series', [])))
                                ))(
                                    (
                                        (lambda info:
                                            (lambda info0:
                                                (
                                                    (next(iter(info0.get('name') or []), 'Unknown') if isinstance(info0, dict) else 'Unknown')
                                                    if isinstance(info0, dict) and info0.get('name') is not None
                                                    else (
                                                        info0.get('name')
                                                        if isinstance(info0, dict) and isinstance(info0.get('name', None), str) and info0.get('name')
                                                        else 'Unknown'
                                                    )
                                                )
                                            )(info[0] if isinstance(info, list) and info else (info if isinstance(info, dict) else {}))
                                        )(stream.get('info', {}))
                                    )
                                )
                            ))(stream.get('info', {}))
                        ))(idx, stream)
                        for idx, stream in enumerate(streams)
                    ]) and (
                        os.makedirs(output_dir, exist_ok=True) or
                        pl.DataFrame({'signal':[1], 'source':[os.path.basename(input_xdf)], 'streams':[len(streams)]}).write_parquet(os.path.join(output_dir, get_signal_filename(base))) or
                        print(f"[READER] Reading finished. Files created in {output_dir}")
                    )
                ))(os.path.splitext(os.path.basename(input_xdf))[0])
            )) or (
                # no streams: still write a top-level empty signal file
                os.makedirs(output_dir, exist_ok=True) or
                pl.DataFrame({'signal':[1], 'source':[os.path.basename(input_xdf)], 'streams':[0]}).write_parquet(os.path.join(output_dir, get_signal_filename(os.path.splitext(os.path.basename(input_xdf))[0]))) or
                print(f"[READER] No streams found in XDF file: {input_xdf}")
            )
        ))(pyxdf.load_xdf(input_xdf)[0])
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

