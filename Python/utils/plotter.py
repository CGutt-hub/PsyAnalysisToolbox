import polars as pl, matplotlib.pyplot as plt, sys, os
from matplotlib.backends.backend_pdf import PdfPages
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python plotter.py <plot_type> <input_parquet> <output_dir>") or sys.exit(1)
    # Lambda: main plotting logic, maximally nested
    run = lambda plot_type, input_parquet, output_dir: (
        # Lambda: print start message
        print(f"[Nextflow] Plotting started for: {input_parquet} as {plot_type}") or (
            # Lambda: read parquet file
            (lambda df:
                # Lambda: create figure and plot
                (lambda fig:
                    # Lambda: select plot type and plot
                    (lambda ax:
                        # Lambda: plot data in greyscale
                        (lambda _: (
                            ax.set_facecolor('white'),
                            plt.style.use('grayscale'),
                            (
                                ax.scatter(df[df.columns[0]], df[df.columns[1]], color='black') if plot_type == 'scatter' else
                                ax.bar(df[df.columns[0]], df[df.columns[1]], color='grey') if plot_type == 'bar' else
                                ax.plot(df[df.columns[0]], df[df.columns[1]], color='black')
                            ),
                            ax.set_title(f"{plot_type.title()} Plot"),
                            ax.set_xlabel(df.columns[0]),
                            ax.set_ylabel(df.columns[1]),
                            plt.tight_layout(),
                            # Lambda: save as PDF/A
                            (lambda pdf_path: (
                                print(f"[Nextflow] Saving plot to {pdf_path}"),
                                PdfPages(pdf_path).savefig(fig),
                                print(f"[Nextflow] Plot PDF/A saved: {pdf_path}")
                            ))(os.path.join(output_dir, f"plot_{plot_type}.pdf")),
                            # Lambda: save parquet as attachment
                            (lambda att_path: (
                                print(f"[Nextflow] Saving parquet attachment to {att_path}"),
                                df.write_parquet(att_path),
                                print(f"[Nextflow] Parquet attachment saved: {att_path}")
                            ))(os.path.join(output_dir, f"plot_{plot_type}_data.parquet"))
                        ))(None)
                    )(fig.add_subplot(111))
                )(plt.figure())
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        # Lambda: check argument count and run main logic
        (lambda a: usage() if len(a) < 4 else run(a[1], a[2], a[3]))(args)
    except Exception as e:
        print(f"[Nextflow] Plotting errored. Error: {e}")
        sys.exit(1)
