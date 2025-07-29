"""
Plot Reporter Module
-------------------
Handles generation and saving of plots for reporting.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt

class PlotReporter:
    def __init__(self, logger: logging.Logger, output_dir_base: str, reporting_figure_format_config: str, reporting_dpi_config: int):
        self.logger = logger
        self.output_dir_base = output_dir_base
        self.reporting_figure_format_config = reporting_figure_format_config
        self.reporting_dpi_config = reporting_dpi_config
        self.logger.info("PlotReporter initialized.")

    def generate_plot(self, participant_id_or_group: str, plot_config: Dict[str, Any], data_payload: pd.DataFrame) -> None:
        # Example: Save a simple plot (user should customize as needed)
        import os
        os.makedirs(self.output_dir_base, exist_ok=True)
        fig, ax = plt.subplots()
        data_payload.plot(ax=ax)
        plot_path = os.path.join(self.output_dir_base, f"{participant_id_or_group}.{self.reporting_figure_format_config}")
        plt.savefig(plot_path, dpi=self.reporting_dpi_config)
        plt.close(fig)
        self.logger.info(f"PlotReporter: Saved plot to {plot_path}.")