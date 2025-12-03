import polars as pl, sys, os

read_txt = lambda ip, enc: (
	print(f"[READER] Processing: {ip}"),
	pl.DataFrame({'lines': open(ip, 'r', encoding=enc).read().split('\n')}).write_parquet(f"{os.path.splitext(os.path.basename(ip))[0]}_txt.parquet"),
	print(f"[READER] Output: {os.path.splitext(os.path.basename(ip))[0]}_txt.parquet"),
	f"{os.path.splitext(os.path.basename(ip))[0]}_txt.parquet"
)[-1]

if __name__ == '__main__': (lambda a: read_txt(a[1], a[2]) if len(a) >= 3 else (print("[READER] Usage: python txt_reader.py <input.txt> <encoding>"), sys.exit(1)))(sys.argv)
