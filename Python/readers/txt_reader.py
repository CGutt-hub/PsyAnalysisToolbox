import polars as pl
import sys
import os


def usage():
    print("[READER] Usage: python txt_reader.py <input_txt> <encoding>")
    sys.exit(1)


def get_output_filename(input_file):
    return f"{os.path.splitext(os.path.basename(input_file))[0]}_txt.parquet"


def run(input_txt, encoding):
    """Convert a text file to a parquet where each row is one line of the file.

    This keeps the reader simple and predictable for downstream preprocessors:
    the first (and only) column contains the text lines in order.
    """
    try:
        print(f"[READER] Started for: {input_txt}")
        with open(input_txt, 'r', encoding=encoding) as fh:
            # preserve empty lines explicitly
            lines = fh.read().split('\n')
        print(f"[READER] TXT file loaded: {input_txt}, lines: {len(lines)}")

        # Create a single-column DataFrame; column name 'text' is conventional but
        # downstream code often uses the first column regardless of name.
        df = pl.DataFrame({'text': lines})
        print(f"[READER] DataFrame created with shape: {df.shape}")
        out = get_output_filename(input_txt)
        df.write_parquet(out)
        print(f"[READER] Parquet file saved: {out}")
    except Exception as e:
        print(f"[READER] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        usage()
    else:
        input_txt = args[1]
        encoding = args[2].strip()
        run(input_txt, encoding)
