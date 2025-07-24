import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas for type checking
import numpy as np
import pandas as pd # Import pandas for type checking
import os
from matplotlib.figure import Figure # Import Figure directly
from typing import Union, Any, Optional, List, Dict 
import copy  # For deepcopy
import inspect
# import mne # For potential future topomaps

class PlotReporter:
    # Class-level defaults
    DEFAULT_FIG_FORMAT = "png"
    DEFAULT_DPI = 100
    DEFAULT_FIGSIZE = (10, 6)
    DEFAULT_PALETTE = None # Let seaborn choose by default
    DEFAULT_XTICKS_ROTATION = 45
    DEFAULT_XTICKS_HA = 'right'

    def __init__(self, logger, output_dir_base,
                 reporting_figure_format_config: Optional[str], 
                 reporting_dpi_config: Optional[Union[int, float]],
                 default_plot_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the PlotReporter.

        Args:
            logger (logging.Logger): Logger instance.
            output_dir_base (str): Base directory where all plots will be saved.
            reporting_figure_format_config (Optional[str]): Desired format for saved figures (e.g., "png", "svg").
            reporting_dpi_config (Optional[Union[int, float]]): Desired DPI for saved figures. # type: ignore
            default_plot_params (Optional[Dict[str, Any]]): Optional dictionary of default plot parameters
                                                           that apply to all plots unless overridden in plot_config.
        """
        self.logger = logger
        self.output_dir_base = output_dir_base 
        
        # Consolidate configuration handling
        self.config = {
            "reporting_figure_format": reporting_figure_format_config,
            "reporting_dpi": reporting_dpi_config,
            # Use deepcopy for default_plot_params to avoid modifying the original dict if it's mutable
            "default_plot_params": copy.deepcopy(default_plot_params) if default_plot_params is not None else {}
        }

        # Validate and set instance attributes from config
        fig_format_from_config = self.config.get("reporting_figure_format")
        if fig_format_from_config and isinstance(fig_format_from_config, str) and fig_format_from_config.strip():
            self.reporting_figure_format = fig_format_from_config
        else:
            if fig_format_from_config is not None: # It was provided but empty or not a string
                 self.logger.warning(f"PlotReporter: 'reporting_figure_format' ('{fig_format_from_config}') is invalid. Defaulting to '{self.DEFAULT_FIG_FORMAT}'.")
            else: # It was missing (None)
                 self.logger.warning(f"PlotReporter: 'reporting_figure_format' is missing. Defaulting to '{self.DEFAULT_FIG_FORMAT}'.")
            self.reporting_figure_format = self.DEFAULT_FIG_FORMAT

        dpi_from_config = self.config.get("reporting_dpi")
        if isinstance(dpi_from_config, (int, float)) and dpi_from_config > 0:
            self.reporting_dpi = dpi_from_config
        else:
            if dpi_from_config is not None: # It was provided but invalid
                self.logger.warning(f"PlotReporter: 'reporting_dpi' ({dpi_from_config}) is invalid. Defaulting to {self.DEFAULT_DPI}.")
            else: # It was missing (None)
                self.logger.warning(f"PlotReporter: 'reporting_dpi' is missing. Defaulting to {self.DEFAULT_DPI}.")
            self.reporting_dpi = self.DEFAULT_DPI

        # Initialize plot generators
        self.plot_generators = {
            "scatterplot": self._generate_scatterplot, # Renamed from correlation
            "barplot": self._generate_barplot, # Changed from anova_bar
            "faceted_barplot": self._generate_faceted_barplot, # Generalized from plv_summary
            "horizontal_barplot": self._generate_horizontal_barplot, # Generalized from fnirs_contrast_bar
            "lineplot": self._generate_lineplot,
            "histogram": self._generate_histogram,
            "boxplot": self._generate_boxplot,
            "violinplot": self._generate_violinplot,
            "heatmap": self._generate_heatmap,
            "kdeplot": self._generate_kdeplot,
            # Add more fundamental plot types here as needed
        }

        self.logger.info(f"PlotReporter initialized. Plots will be saved under: {self.output_dir_base}. Format: {self.reporting_figure_format}, DPI: {self.reporting_dpi}. Default plot params configured: {bool(self.config['default_plot_params'])}")

    def _extract_data(self, 
                        data_payload: Union[pd.DataFrame, Dict[str, Any], None], 
                        key_or_column: Optional[str], 
                        is_required: bool = True, 
                        default: Any = None) -> Any:
        """
        Extracts data from the data_payload (DataFrame or Dict).
        Args:
            data_payload: The source of data.
            key_or_column: The key (for dict) or column name (for DataFrame) to extract.
                           If None, returns default.
            is_required: If True and data is not found, logs an error.
            default: Value to return if key_or_column is None or not found and not is_required.
        Returns:
            The extracted data or default value.
        """
        if key_or_column is None:
            return default

        if data_payload is None:
            if is_required:
                self.logger.error(f"Data extraction failed: data_payload is None, but '{key_or_column}' was required.")
            return default

        data_val = None
        found = False

        if isinstance(data_payload, pd.DataFrame):
            if key_or_column in data_payload.columns:
                data_val = data_payload[key_or_column]
                found = True
        elif isinstance(data_payload, dict):
            if key_or_column in data_payload:
                data_val = data_payload[key_or_column]
                found = True # type: ignore
        else:
            self.logger.warning(f"Data extraction: data_payload is not a DataFrame or Dict (type: {type(data_payload)}). Cannot extract '{key_or_column}'.")

        if not found:
            if is_required:
                self.logger.error(f"Data extraction failed: Required key/column '{key_or_column}' not found in data_payload.")
            else:
                self.logger.debug(f"Data extraction: Optional key/column '{key_or_column}' not found. Using default.")
            return default
        
        return data_val

    def _save_plot(self, fig: Figure, participant_id_or_group: str, plot_name: str, plot_category_subdir: str, overwrite: bool = True):
        """Helper function to save matplotlib figures."""
        # Create a directory for the participant/group, then the plot category subdir
        plot_dir = os.path.join(self.output_dir_base, str(participant_id_or_group), plot_category_subdir)
        
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{plot_name}.{self.reporting_figure_format}") # Use instance variable

        if not overwrite and os.path.exists(plot_path):
            self.logger.info(f"Plot '{plot_path}' already exists and overwrite is False. Skipping save.")
            plt.close(fig)
            return None

        try:
            fig.savefig(plot_path, dpi=self.reporting_dpi) # Use instance variable # type: ignore
            plt.close(fig) # type: ignore
            self.logger.info(f"Plot saved: {plot_path}")
            return plot_path # type: ignore
        except Exception as e:
            self.logger.error(f"Failed to save plot {plot_name} for {participant_id_or_group}: {e}")
            plt.close(fig) 
            return None

    def _draw_significance_brackets(self, ax: Any, brackets_config: List[Dict[str, Any]]):
        """
        Draws significance brackets and text on a plot axis based on a configuration list.

        Args:
            ax (plt.Axes): The matplotlib axes object to draw on.
            brackets_config (List[Dict[str, Any]]): A list of configurations for each bracket.
                Each dict should contain:
                - 'x1', 'x2' (float): The x-coordinates (categorical indices) for the comparison.
                - 'y_start' (float): The y-axis value where the bracket should start.
                - 'text' (str): The significance text (e.g., '*', '**', 'n.s.').
                - 'height' (float, optional): The height of the bracket's vertical lines.
                - 'fontsize', 'color' (optional): Formatting for the text.
        """
        if not brackets_config: return

        for config in brackets_config:
            x1, x2, y_start = config.get('x1'), config.get('x2'), config.get('y_start')
            if None in [x1, x2, y_start]:
                self.logger.warning("Skipping significance bracket due to missing 'x1', 'x2', or 'y_start' in config.")
                continue
            try:
                # Ensure x1, x2, y_start are floats for plotting
                x1_f, x2_f, y_start_f = float(x1), float(x2), float(y_start) # type: ignore
                height = config.get('height', y_start_f * 0.05)
                text = config.get('text', '*')
                color = config.get('color', 'k')
                fontsize = config.get('fontsize', 10)
                ax.plot([x1_f, x1_f, x2_f, x2_f], [y_start_f, y_start_f + height, y_start_f + height, y_start_f], lw=1.5, c=color)
                ax.text((x1_f + x2_f) * 0.5, y_start_f + height, text, ha='center', va='bottom', color=color, fontsize=fontsize)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Skipping significance bracket due to conversion error: {e}")
                continue

    def generate_plot(self, 
                      participant_id_or_group: str, 
                      plot_config: Dict[str, Any], 
                      data_payload: Union[pd.DataFrame, Dict[str, Any], None],
                      overwrite: bool = True):
        """
        Generates and saves a plot based on the provided configuration and data.
        """
        plot_type = plot_config.get("plot_type")
        if not plot_type:
            self.logger.error("generate_plot: 'plot_type' missing in plot_config.")
            return

        generator_func = self.plot_generators.get(plot_type)
        if not generator_func:
            self.logger.error(f"generate_plot: Unknown plot_type '{plot_type}'. No generator function found.")
            return

        data_mapping = plot_config.get("data_mapping", {})
        # Merge default plot params with plot-specific params
        instance_default_plot_params = self.config.get("default_plot_params", {})
        plot_specific_params = plot_config.get("plot_params", {})
        # Deepcopy instance_default_plot_params to prevent modification if it contains mutable objects
        plot_params = {**copy.deepcopy(instance_default_plot_params), **plot_specific_params}
        plot_category_subdir = plot_config.get("plot_category_subdir", "general")
        
        # Prepare arguments for the generator function by extracting data
        call_args = {}
        for arg_name, data_key_config in data_mapping.items():
            
            # Check if the argument name is expected by the generator function
            # This is a basic check; a more thorough check would inspect the function signature
            # For now, we extract everything mapped and let the generator handle unused args via **plot_params
            # or check for None if required.
            
            # Check if the data_mapping key is valid (string or dict)
            if not isinstance(data_key_config, (str, dict)):
                 self.logger.warning(f"generate_plot: Invalid data_mapping entry for argument '{arg_name}'. Expected string or dict, got {type(data_key_config)}. Skipping.")
                 call_args[arg_name] = None # Ensure it's None if mapping is invalid
                 continue # Skip extraction for this invalid entry

            # --- Corrected Logic ---
            # If the mapping is a dictionary, it's a request to extract a complex object from the payload.
            if isinstance(data_key_config, dict):
                key_to_extract = data_key_config.get("key")
                is_req = data_key_config.get("required", True)
                default_val = data_key_config.get("default")
                call_args[arg_name] = self._extract_data(data_payload, key_to_extract, is_required=is_req, default=default_val)
            else:
                # If the mapping is a simple value (like a string for a column name), pass it directly.
                # This is the correct behavior for seaborn plotters that expect column names.
                call_args[arg_name] = data_key_config


        # If a generator expects 'plot_data_df' and it wasn't mapped,
        # but the main data_payload is a DataFrame, pass it as a fallback.
        generator_params = inspect.signature(generator_func).parameters
        if 'plot_data_df' in generator_params and 'plot_data_df' not in call_args:
            if isinstance(data_payload, pd.DataFrame):
                self.logger.info(f"Plot '{plot_type}': 'plot_data_df' not mapped. Passing the main data_payload DataFrame as a fallback.")
                call_args['plot_data_df'] = data_payload

        # Check for critically missing required data before proceeding
        missing_required_args = []
        for arg_name, data_key_config in data_mapping.items():
            # Only check for missing data if it was supposed to be extracted (i.e., a dict mapping)
            if isinstance(data_key_config, dict):
                is_req = data_key_config.get("required", True)
                if is_req and call_args.get(arg_name) is None:
                    key_or_col = data_key_config.get("key")
                    missing_required_args.append(f"'{arg_name}' (expected from key/column: '{key_or_col}')")
            elif call_args.get(arg_name) is None: # For simple mappings, just check if it's None
                key_or_col = data_key_config
                missing_required_args.append(f"'{arg_name}' (expected from key/column: '{key_or_col}')")
        
        if missing_required_args:
            self.logger.error(
                f"Cannot generate plot '{plot_type}' for '{participant_id_or_group}' due to missing required data: " # type: ignore
                f"{', '.join(missing_required_args)}. Check data_payload and data_mapping."
            )
            # Generate a placeholder plot indicating missing data
            placeholder_title = f"{plot_type.replace('_', ' ').title()} for {participant_id_or_group} (Missing Data)"
            placeholder_fig, ax_placeholder = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax_placeholder.set_title(placeholder_title)
            ax_placeholder.text(0.5, 0.5, f"Plot Generation Error:\nMissing Required Data\n({', '.join(missing_required_args)})", ha='center', va='center', fontsize=9, wrap=True) # type: ignore
            placeholder_fig.tight_layout()
            self._save_plot(placeholder_fig, participant_id_or_group, f"{plot_type}_{participant_id_or_group}_missing_data".replace(" ", "_").lower(), plot_category_subdir, overwrite=True) # Always overwrite placeholders
            return

        # Format title and plot name
        template_context = {"participant_id_or_group": participant_id_or_group, **plot_params, **{k: str(v)[:30] for k,v in call_args.items()}} # Add data keys for templating
        default_title_template = f"{plot_type.replace('_', ' ').title()} for {{participant_id_or_group}}" # Use double braces for literal braces in f-string
        
        # Use pop() to retrieve the title template AND remove it from plot_params to avoid the TypeError
        title_template_from_config = plot_params.pop("title", None)
        # Ensure title_template is a string before formatting
        actual_title_template = title_template_from_config if isinstance(title_template_from_config, str) else default_title_template
        final_title = actual_title_template.format(**template_context)
        
        default_plot_name_template = f"{plot_type}_{{participant_id_or_group}}" # participant_id_or_group is in template_context
        
        plot_name_template_from_config = plot_config.get("plot_name")
        # Ensure plot_name_template is a string before formatting
        actual_plot_name_template = plot_name_template_from_config if isinstance(plot_name_template_from_config, str) else default_plot_name_template
        raw_plot_name = actual_plot_name_template.format(**template_context) # This will be a string
        fig: Optional[Figure] = None # Initialize fig here to ensure it's always bound

        final_plot_name = raw_plot_name.replace(" ", "_").lower()

        try:
            self.logger.info(f"Generating plot '{final_plot_name}' of type '{plot_type}' for '{participant_id_or_group}'.")
            fig = generator_func(participant_id_or_group=participant_id_or_group, title=final_title, **call_args, **plot_params)
            if fig and isinstance(fig, Figure): # Ensure a matplotlib Figure is returned
                self._save_plot(fig, participant_id_or_group, final_plot_name, plot_category_subdir, overwrite=overwrite)
            else:
                self.logger.warning(f"Plot generator for '{plot_type}' returned no figure for '{participant_id_or_group}'. Plot not saved.")
        except Exception as e:
            self.logger.error(f"Error generating plot '{final_plot_name}' for '{participant_id_or_group}': {e}", exc_info=True)
            # Ensure figure is closed if an error occurs after figure creation but before saving
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure):
                 plt.close(fig)

    # --- Refactored Plot Generator Methods ---

    def _generate_faceted_barplot(self, participant_id_or_group: str, title: str, 
                                   plot_data_df: Optional[pd.DataFrame],
                                   x_col: str,
                                   y_col: str,
                                   **plot_params):
        """
        Generates a faceted bar plot using seaborn.catplot(kind='bar').
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            title (str): The fully formatted title for the plot.
            plot_data_df (Optional[pd.DataFrame]): DataFrame containing the data for plotting.
            x_col (str): Name of the column for the x-axis categories.
            y_col (str): Name of the column for the y-axis values (bar heights).
            **plot_params: Additional parameters from plot_config, such as:
                           - hue_col (str, optional): Column for color grouping.
                           - facet_col (str, optional): Column for faceting into columns.
                           - facet_row (str, optional): Column for faceting into rows.
                           - col_wrap (int, optional): Wrap facet_col into this many columns.
                           - errorbar (str or tuple, optional): Seaborn errorbar argument (e.g., 'se', ('ci', 95)).
                           - palette (str or list, optional): Color palette.
                           - capsize (float, optional): Cap size for error bars.
                           - sharey, sharex (bool, optional): Whether facets share axes.
                           - height, aspect (float, optional): Size of each facet.
                           - x_axis_label, y_axis_label (str, optional): Custom axis labels.
                           - facet_col_title_template (str, optional): Template for facet column titles (e.g., "{col_name}").
                           - facet_row_title_template (str, optional): Template for facet row titles (e.g., "{row_name}").
                           - xticks_rotation (int, optional), xticks_ha (str, optional): For x-tick label formatting.
                           - suptitle_y (float, optional): Y position for suptitle.
        """
        fig: Optional[Figure] = None # type: ignore
        g: Optional[sns.FacetGrid] = None # Initialize g here, will be None if plotting conditions aren't met

        if plot_data_df is not None and not plot_data_df.empty and \
           x_col in plot_data_df.columns and y_col in plot_data_df.columns:
            try:
                # Extract catplot specific parameters from plot_params
                catplot_args = {
                    'data': plot_data_df,
                    'x': x_col,
                    'y': y_col,
                    'hue': plot_params.get('hue_col'),
                    'col': plot_params.get('facet_col'),
                    'row': plot_params.get('facet_row'),
                    'kind': 'bar', # This function is specifically for bar plots
                    'errorbar': plot_params.get('errorbar', 'se'),
                    'capsize': plot_params.get('capsize', 0.1),
                    'palette': plot_params.get('palette', self.DEFAULT_PALETTE),
                    'sharey': plot_params.get('sharey', False),
                    'sharex': plot_params.get('sharex', True), # Default to True for catplot
                    'height': plot_params.get('height', 4),
                    'aspect': plot_params.get('aspect', 1.2),
                    'col_wrap': plot_params.get('col_wrap')
                }
                # Remove None keys as catplot might not handle them well for all args
                catplot_args = {k: v for k, v in catplot_args.items() if v is not None}

                g = sns.catplot(**catplot_args) # type: ignore
                
                g.set_axis_labels(plot_params.get('x_axis_label', x_col.replace('_',' ').title()), 
                                  plot_params.get('y_axis_label', y_col.replace('_',' ').title())) # type: ignore
                g.set_titles(col_template=plot_params.get('facet_col_title_template', "{col_name}"), # Use {col_name} for columns
                             row_template=plot_params.get('facet_row_title_template', "{row_name}")) # Use {row_name} for rows # type: ignore
                fig = g.figure # Use .figure instead of .fig
                fig.suptitle(title, y=plot_params.get('suptitle_y', 1.03))

                # Handle custom annotations at the figure level
                annotations = plot_params.get("annotations", [])
                has_annotations = isinstance(annotations, list) and len(annotations) > 0
                if has_annotations:
                    for ann_config in annotations:
                        if isinstance(ann_config, dict) and "text" in ann_config:
                            # For FacetGrid, text is added to the figure, not a specific axis
                            fig.text(ann_config.get('x', 0.98), ann_config.get('y', 0.5),
                                     ann_config["text"],
                                     transform=fig.transFigure, # Use figure-level coordinates
                                     fontsize=ann_config.get('fontsize', 9),
                                     verticalalignment=ann_config.get('va', 'center'),
                                     horizontalalignment=ann_config.get('ha', 'left'),
                                     bbox=ann_config.get('bbox'))

                if fig and hasattr(fig, 'axes'): # Ensure a figure object was created and has axes
                    for ax_g in fig.axes: # Rotate x-axis labels for all subplots
                        ax_g.tick_params(axis='x', 
                                         rotation=plot_params.get('xticks_rotation', self.DEFAULT_XTICKS_ROTATION))
                    # Adjust layout to prevent suptitle overlap if it exists, rect expects a tuple
                    fig.tight_layout(rect=(0, 0, 0.85, 0.96) if has_annotations else (0, 0, 1, 0.96)) 

            except Exception as e:
                self.logger.warning(f"Could not generate faceted bar plot for {participant_id_or_group} due to: {e}. Plotting placeholder.", exc_info=True)
                # Try to close the FacetGrid's figure if it exists, or the main fig variable
                if g is not None and hasattr(g, 'figure') and isinstance(g.figure, Figure): # g is now guaranteed to be bound
                    plt.close(g.figure)
                elif fig and isinstance(fig, plt.Figure): # Fallback if fig was assigned directly # type: ignore
                    plt.close(fig)
                fig = None # Ensure fig is None so placeholder is created
                # Placeholder will be created below if fig is still None
        
        if fig is None: # If data was invalid from the start or other pre-plotting issue
            fig, ax_placeholder = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax_placeholder.set_title(title) # Set title on the placeholder
            ax_placeholder.text(0.5, 0.5, "Faceted Bar Plot (No Data or Invalid Columns)", ha='center', va='center')
        return fig

    def _generate_scatterplot(self, 
                         participant_id_or_group: str,
                         title: str,
                         x_data: Union[pd.Series, np.ndarray, List, None], 
                         y_data: Union[pd.Series, np.ndarray, List, None], 
                         **plot_params):
        """
        Generates a generic scatter plot, optionally with a regression line and annotations.
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            title (str): The fully formatted title for the plot.
            x_data (Union[pd.Series, np.ndarray, List, None]): Data for the x-axis.
            y_data (Union[pd.Series, np.ndarray, List, None]): Data for the y-axis.
            **plot_params: Additional parameters such as:
                           - show_reg_line (bool, optional): Whether to show a regression line. Defaults to True.
                           - x_axis_label (str, optional): Label for the x-axis. Defaults to "X-axis".
                           - y_axis_label (str, optional): Label for the y-axis. Defaults to "Y-axis".
                           - annotations (list of dict, optional): For custom text on the plot (e.g., stats).
                             Each dict: {'text': 'str', 'xy': (float, float), 'xycoords': 'str', ...}
                           - scatter_kws (dict, optional): Keyword arguments for sns.scatterplot.
                           - reg_kws (dict, optional): Keyword arguments for sns.regplot.
        """
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE)) # type: ignore
        try:
            # Use plot_params for seaborn arguments
            show_reg_line = plot_params.get('show_reg_line', True)
            scatter_kws = plot_params.get('scatter_kws', {'s': 50, 'alpha': 0.7})
            reg_kws = plot_params.get('reg_kws', {'color': 'red', 'line_kws': {'linewidth': 2}})

            if x_data is not None and y_data is not None and \
               hasattr(x_data, '__len__') and hasattr(y_data, '__len__') and \
               len(x_data) == len(y_data) and len(x_data) > 0:
                sns.scatterplot(x=x_data, y=y_data, ax=ax, **scatter_kws) # type: ignore
                if show_reg_line and len(x_data) >= 2 : # Need at least 2 points for a regression line
                     sns.regplot(x=x_data, y=y_data, ax=ax, scatter=False, **reg_kws) # type: ignore
            else:
                ax.text(0.5, 0.5, "Scatter Plot (Insufficient or Mismatched Data)", ha='center', va='center')

            ax.set_xlabel(plot_params.get('x_axis_label', "X-axis"), fontsize=12)
            ax.set_ylabel(plot_params.get('y_axis_label', "Y-axis"), fontsize=12)
            ax.set_title(title, fontsize=14)

            # Handle custom annotations (e.g., stats text)
            annotations = plot_params.get("annotations", [])
            if isinstance(annotations, list):
                for ann_config in annotations:
                    if isinstance(ann_config, dict) and "text" in ann_config:
                        ax.text(ann_config.get('x', 0.05), ann_config.get('y', 0.95), 
                                                    ann_config["text"], # type: ignore
                                transform=ann_config.get('transform', ax.transAxes), 
                                fontsize=ann_config.get('fontsize', 10),
                                verticalalignment=ann_config.get('va', 'top'),
                                horizontalalignment=ann_config.get('ha', 'left'), 
                                bbox=ann_config.get('bbox')) # Get bbox from ann_config if provided

            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate scatterplot for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True) # type: ignore
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Scatter Plot (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_barplot(self, 
                           participant_id_or_group: str,
                           title: str,
                           plot_data_df: Optional[pd.DataFrame], 
                           x_col: str, 
                           y_col: str, 
                           **plot_params):
        """
        Generates a generic bar plot.
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            title (str): The fully formatted title for the plot.
            plot_data_df (Optional[pd.DataFrame]): DataFrame containing the data for plotting.
                                                 Expected to have x_col and y_col.
            x_col (str): Name of the column for the x-axis categories.
            y_col (str): Name of the column for the y-axis values (bar heights).
            **plot_params: Additional parameters such as:
                           - hue_col (str, optional): Column for color grouping.
                           - errorbar (str or tuple, optional): Seaborn errorbar argument (e.g., 'se', ('ci', 95)).
                           - palette (str or list, optional): Color palette for seaborn.
                           - annotations (list of dict, optional): For custom text on the plot.
                             Each dict: {'text': 'str', 'xy': (float, float), 'xycoords': 'str', ...}
        """ # Added default xticks rotation and ha
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try: # type: ignore
            if plot_data_df is not None and not plot_data_df.empty and \
               x_col in plot_data_df.columns and y_col in plot_data_df.columns:
                
                # Extract relevant plot_params for seaborn.barplot
                hue_col = plot_params.get('hue_col')
                errorbar_val = plot_params.get('errorbar', 'se') 
                palette = plot_params.get('palette', self.DEFAULT_PALETTE)
                capsize = plot_params.get('capsize', 0.1)
                
                # Ensure hue_col exists if specified
                if hue_col and hue_col not in plot_data_df.columns:
                    self.logger.warning(f"PlotReporter (_generate_barplot): hue_col '{hue_col}' not found in plot_data_df. Ignoring hue.")
                    hue_col = None

                sns.barplot(data=plot_data_df, x=x_col, y=y_col, # type: ignore
                            hue=hue_col, 
                            errorbar=errorbar_val, 
                            ax=ax, capsize=capsize, palette=palette)
                
                ax.set_ylabel(plot_params.get('y_axis_label', f"Mean {y_col}"), fontsize=12)
                ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_', ' ').capitalize()), fontsize=12)
                ax.set_title(title, fontsize=14)

                # Handle significance brackets
                significance_brackets = plot_params.get("significance_brackets", [])
                if significance_brackets:
                    self._draw_significance_brackets(ax, significance_brackets)

                # Handle custom annotations
                annotations = plot_params.get("annotations", [])
                if isinstance(annotations, list):
                    for ann_config in annotations:
                        if isinstance(ann_config, dict) and "text" in ann_config:
                            ax.text(ann_config.get('x', 0.05), ann_config.get('y', 0.95), 
                                                    ann_config["text"], # type: ignore
                                    transform=ann_config.get('transform', ax.transAxes), 
                                    fontsize=ann_config.get('fontsize', 10),
                                    verticalalignment=ann_config.get('va', 'top'),
                                    bbox=ann_config.get('bbox')) # Get bbox from ann_config if provided
            else:
                ax.text(0.5, 0.5, "Bar Plot (No Data or Missing Columns)", ha='center', va='center')
                ax.set_title(title, fontsize=14)
            
            plt.xticks(rotation=plot_params.get('xticks_rotation', self.DEFAULT_XTICKS_ROTATION), 
                       ha=plot_params.get('xticks_ha', self.DEFAULT_XTICKS_HA)) # type: ignore
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate barplot for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Bar Plot (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_horizontal_barplot(self, 
                                    participant_id_or_group: str,
                                    title: str,
                                    plot_data_df: Optional[pd.DataFrame],
                                    y_col: str, # Categories on the y-axis
                                    x_col: str, # Values (bar lengths) on the x-axis
                                    **plot_params):
        """
        Generates a generic horizontal bar plot.
        Args:
            participant_id_or_group (str): Participant ID or "GROUP".
            title (str): The fully formatted title for the plot.
            plot_data_df (Optional[pd.DataFrame]): DataFrame containing the data for plotting.
            y_col (str): Name of the column for the y-axis categories.
            x_col (str): Name of the column for the x-axis values (bar lengths).
            **plot_params: Additional parameters such as:
                           - hue_col (str, optional): Column for color grouping.
                           - errorbar (str or tuple, optional): Seaborn errorbar argument (e.g., 'se', ('ci', 95)).
                           - palette (str or list or dict, optional): Color palette.
                           - x_axis_label (str, optional): Custom label for the x-axis.
                           - y_axis_label (str, optional): Custom label for the y-axis.
                           - annotations (list of dict, optional): For custom text.
                           - bar_height (float, optional): Height of the bars.
                           - dodge (bool, optional): Whether to dodge bars for hue. Default True.
                           - ytick_labelsize (int, optional): Fontsize for y-tick labels.
                           - show_zeroline (bool, optional): Whether to show a vertical line at x=0. Default True.
        """
        fig = None # Initialize fig to ensure it's bound at the very start
        if plot_data_df is None or plot_data_df.empty or \
           y_col not in plot_data_df.columns or x_col not in plot_data_df.columns:
            self.logger.warning(f"Horizontal bar plot for '{title}' cannot be generated: Invalid or empty data, or missing x/y columns.")
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.text(0.5, 0.5, "Horizontal Bar Plot\n(Invalid/Empty Data or Missing Columns)", ha='center', va='center', fontsize=10)
            ax.set_title(title)
            return fig

        try: # type: ignore
            # Dynamically adjust figure height based on number of y-categories if not specified
            num_categories = len(plot_data_df[y_col].unique())
            default_height = max(6, num_categories * plot_params.get('bar_height', 0.35) * (1.5 if plot_params.get('hue_col') else 1.0))
            fig_w, fig_h_default = self.DEFAULT_FIGSIZE
            figsize_h = plot_params.get('figsize_h', default_height) # Height can be dynamic
            figsize_w = plot_params.get('figsize_w', fig_w) # Width from default or param
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            sns.barplot(data=plot_data_df, y=y_col, x=x_col, # type: ignore
                        hue=plot_params.get('hue_col'), 
                        ax=ax, 
                        dodge=plot_params.get('dodge', True), 
                        palette=plot_params.get('palette', self.DEFAULT_PALETTE),
                        errorbar=plot_params.get('errorbar', 'se')) # Consistent key and default
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_',' ').title()), fontsize=12)
            ax.set_ylabel(plot_params.get('y_axis_label', y_col.replace('_',' ').title()), fontsize=12)
            if plot_params.get('show_zeroline', True):
                ax.axvline(0, color='k', linestyle='--', linewidth=0.8) 
            ax.tick_params(axis='y', labelsize=plot_params.get('ytick_labelsize', 8)) 
            fig.tight_layout()
        except Exception as e:
            self.logger.error(f"Error generating horizontal bar plot for '{title}': {e}", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig) # Close potentially corrupted figure
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.text(0.5, 0.5, "Horizontal Bar Plot\n(Plotting Error)", ha='center', va='center', fontsize=10)
            ax.set_title(title)
        return fig

    def _generate_lineplot(self, participant_id_or_group: str, title: str,
                       plot_data_df: Optional[pd.DataFrame],
                       x_col: str, y_col: str,
                       **plot_params):
        """Generates a line plot."""
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try:
            if plot_data_df is not None and not plot_data_df.empty and \
               x_col in plot_data_df.columns and y_col in plot_data_df.columns:
                 # type: ignore
                sns.lineplot(data=plot_data_df, x=x_col, y=y_col,
                             hue=plot_params.get('hue_col'),
                             style=plot_params.get('style_col'),
                             markers=plot_params.get('markers', False),
                             dashes=plot_params.get('dashes', True),
                             errorbar=plot_params.get('errorbar', ('ci', 95)),
                             palette=plot_params.get('palette', self.DEFAULT_PALETTE),
                             ax=ax) # type: ignore
                
                ax.set_title(title)
                ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_',' ').title()))
                ax.set_ylabel(plot_params.get('y_axis_label', y_col.replace('_',' ').title()))
                
                if plot_params.get('hue_col'):
                    # Ensure hue_col is a string before calling replace
                    hue_col_str = str(plot_params.get('hue_col')) if plot_params.get('hue_col') is not None else ""
                    legend_title = plot_params.get('legend_title', hue_col_str.replace('_',' ').title())
                    ax.legend(title=legend_title)

                annotations = plot_params.get("annotations", [])
                if isinstance(annotations, list):
                    for ann_config in annotations:
                        if isinstance(ann_config, dict) and "text" in ann_config:
                            ax.text(ann_config.get('x', 0.05), ann_config.get('y', 0.95), 
                                    ann_config["text"], 
                                    transform=ann_config.get('transform', ax.transAxes),
                                    fontsize=ann_config.get('fontsize', 10),
                                    verticalalignment=ann_config.get('va', 'top'), 
                                    bbox=ann_config.get('bbox')) # Get bbox from ann_config if provided
            else:
                ax.text(0.5, 0.5, "Line Plot (No Data or Missing Columns)", ha='center', va='center')
                ax.set_title(title)
            
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate lineplot for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Line Plot (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_histogram(self, participant_id_or_group: str, title: str,
                            plot_data_df: Optional[pd.DataFrame],
                            x_col: str,
                            **plot_params):
        """Generates a histogram."""
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try:
            if plot_data_df is not None and not plot_data_df.empty and x_col in plot_data_df.columns:
                sns.histplot(data=plot_data_df, x=x_col, # type: ignore
                             hue=plot_params.get('hue_col'),
                             bins=plot_params.get('bins', 'auto'),
                             kde=plot_params.get('kde', False),
                             stat=plot_params.get('stat', 'count'),
                             multiple=plot_params.get('multiple', 'layer'), 
                             element=plot_params.get('element', 'bars'), 
                             palette=plot_params.get('palette', self.DEFAULT_PALETTE),
                             ax=ax) # type: ignore
                ax.set_title(title)
                ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_',' ').title()))
                ax.set_ylabel(plot_params.get('y_axis_label', plot_params.get('stat', 'count').capitalize()))
            else:
                ax.text(0.5, 0.5, "Histogram (No Data or Missing X-column)", ha='center', va='center')
                ax.set_title(title)
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate histogram for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Histogram (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_boxplot(self, participant_id_or_group: str, title: str,
                          plot_data_df: Optional[pd.DataFrame],
                          x_col: str, y_col: str,
                          **plot_params):
        """Generates a box plot."""
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try:
            if plot_data_df is not None and not plot_data_df.empty and \
               x_col in plot_data_df.columns and y_col in plot_data_df.columns: # type: ignore
                sns.boxplot(data=plot_data_df, x=x_col, y=y_col,
                            hue=plot_params.get('hue_col'),
                            orient=plot_params.get('orient', 'v'),
                            palette=plot_params.get('palette', self.DEFAULT_PALETTE),
                            ax=ax)
                ax.set_title(title)

                # Handle significance brackets
                significance_brackets = plot_params.get("significance_brackets", [])
                if significance_brackets:
                    self._draw_significance_brackets(ax, significance_brackets)
                ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_',' ').title()))
                ax.set_ylabel(plot_params.get('y_axis_label', y_col.replace('_',' ').title()))
                plt.xticks(rotation=plot_params.get('xticks_rotation', self.DEFAULT_XTICKS_ROTATION if plot_params.get('orient', 'v') == 'v' else 0), 
                           ha=plot_params.get('xticks_ha', self.DEFAULT_XTICKS_HA if plot_params.get('orient', 'v') == 'v' else 'center'))
            else:
                ax.text(0.5, 0.5, "Box Plot (No Data or Missing Columns)", ha='center', va='center')
                ax.set_title(title)
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate boxplot for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Box Plot (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_violinplot(self, participant_id_or_group: str, title: str,
                             plot_data_df: Optional[pd.DataFrame],
                             x_col: str, y_col: str,
                             **plot_params):
        """Generates a violin plot."""
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try:
            if plot_data_df is not None and not plot_data_df.empty and \
               x_col in plot_data_df.columns and y_col in plot_data_df.columns: # type: ignore
                sns.violinplot(data=plot_data_df, x=x_col, y=y_col,
                               hue=plot_params.get('hue_col'),
                               orient=plot_params.get('orient', 'v'),
                               split=plot_params.get('split', False),
                               inner=plot_params.get('inner', 'box'), # "box", "quartile", "point", "stick", None
                               palette=plot_params.get('palette', self.DEFAULT_PALETTE),
                               ax=ax)
                ax.set_title(title)
                ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_',' ').title()))
                ax.set_ylabel(plot_params.get('y_axis_label', y_col.replace('_',' ').title()))
                plt.xticks(rotation=plot_params.get('xticks_rotation', self.DEFAULT_XTICKS_ROTATION if plot_params.get('orient', 'v') == 'v' else 0),
                           ha=plot_params.get('xticks_ha', self.DEFAULT_XTICKS_HA if plot_params.get('orient', 'v') == 'v' else 'center'))
            else:
                ax.text(0.5, 0.5, "Violin Plot (No Data or Missing Columns)", ha='center', va='center')
                ax.set_title(title)
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate violinplot for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig) # Close potentially corrupted figure
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Violin Plot (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_heatmap(self, participant_id_or_group: str, title: str,
                          data_matrix: Optional[Union[pd.DataFrame, np.ndarray]],
                          **plot_params):
        """Generates a heatmap. Expects data_matrix directly via data_mapping."""
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try:
            if data_matrix is not None:
                # If it's a DataFrame, use its index/columns for labels unless overridden # type: ignore
                xticklabels = plot_params.get('xticklabels', 'auto')
                yticklabels = plot_params.get('yticklabels', 'auto')
                if isinstance(data_matrix, pd.DataFrame):
                    if xticklabels == 'auto':
                        # Convert pandas Index to list of strings for explicit typing
                        xticklabels = list(data_matrix.columns.astype(str))
                    if yticklabels == 'auto':
                        # Convert pandas Index to list of strings for explicit typing
                        yticklabels = list(data_matrix.index.astype(str))

                sns.heatmap(data_matrix, 
                            annot=plot_params.get('annot', False),
                            fmt=plot_params.get('fmt', '.2f'),
                            cmap=plot_params.get('cmap', 'viridis'),
                            cbar=plot_params.get('cbar', True),
                            square=plot_params.get('square', False),
                            linewidths=plot_params.get('linewidths', 0),
                            linecolor=plot_params.get('linecolor', 'black'),
                            xticklabels=xticklabels,
                            yticklabels=yticklabels,
                            ax=ax) # type: ignore
                ax.set_title(title)
                plt.xticks(rotation=plot_params.get('xticks_rotation', self.DEFAULT_XTICKS_ROTATION))
                plt.yticks(rotation=plot_params.get('yticks_rotation', 0))
            else:
                ax.text(0.5, 0.5, "Heatmap (No Data Matrix Provided)", ha='center', va='center')
                ax.set_title(title)
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate heatmap for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "Heatmap (Plotting Error)", ha='center', va='center')
        return fig

    def _generate_kdeplot(self, participant_id_or_group: str, title: str,
                          plot_data_df: Optional[pd.DataFrame],
                          x_col: str,
                          **plot_params):
        """Generates a Kernel Density Estimate plot (1D or 2D)."""
        fig = None # Initialize fig to ensure it's bound
        fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
        try:
            y_col_kde = plot_params.get('y_col_kde') 

            if plot_data_df is not None and not plot_data_df.empty and x_col in plot_data_df.columns:
                if y_col_kde and y_col_kde not in plot_data_df.columns:
                    self.logger.warning(f"KDE Plot: y_col_kde '{y_col_kde}' not found. Generating 1D KDE for x_col '{x_col}'.")
                    y_col_kde = None 
 # type: ignore
                sns.kdeplot(data=plot_data_df, x=x_col, y=y_col_kde,
                            hue=plot_params.get('hue_col'),
                            fill=plot_params.get('fill', True),
                            levels=plot_params.get('levels', 10), 
                            thresh=plot_params.get('thresh', 0.05), 
                            cmap=plot_params.get('cmap', 'viridis' if y_col_kde else None), 
                            palette=plot_params.get('palette', self.DEFAULT_PALETTE),
                            ax=ax)
                ax.set_title(title)
                ax.set_xlabel(plot_params.get('x_axis_label', x_col.replace('_',' ').title()))
                if y_col_kde:
                    ax.set_ylabel(plot_params.get('y_axis_label', y_col_kde.replace('_',' ').title()))
                else:
                    ax.set_ylabel(plot_params.get('y_axis_label', 'Density'))
            else:
                ax.text(0.5, 0.5, "KDE Plot (No Data or Missing X-column)", ha='center', va='center') 
                ax.set_title(title)
            fig.tight_layout()
        except Exception as e:
            self.logger.warning(f"Could not generate KDE plot for '{participant_id_or_group}' due to: {e}. Plotting placeholder.", exc_info=True)
            if 'fig' in locals() and fig is not None and isinstance(fig, Figure): plt.close(fig)
            fig, ax = plt.subplots(figsize=plot_params.get('figsize', self.DEFAULT_FIGSIZE))
            ax.set_title(title)
            ax.text(0.5, 0.5, "KDE Plot (Plotting Error)", ha='center', va='center')
        return fig