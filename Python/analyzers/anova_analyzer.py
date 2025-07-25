import pandas as pd
import pingouin as pg
import numpy as np
import re # For parsing column names

class ANOVAAnalyzer:
    # Default parameters for ANOVA methods
    DEFAULT_ANOVA_EFFSIZE = "np2"
    DEFAULT_ANOVA_DETAILED = True
    DEFAULT_POSTHOC_P_ADJUST = "holm"
    DEFAULT_POSTHOC_EFFSIZE = "cohen"

    # Default parameters for PLV data preparation
    DEFAULT_PLV_METRIC_PREFIX = "plv_avg_"
    DEFAULT_PLV_ID_VARS_COL = 'participant_id'
    DEFAULT_PLV_VALUE_NAME_OUTPUT = 'plv_value'
    DEFAULT_PLV_CONDITION_CAPTURE_GROUP = 'condition'
    DEFAULT_PLV_MIDDLE_FACTOR_CAPTURE_GROUP = 'middle_factor'
    DEFAULT_PLV_ALLOW_OPTIONAL_MIDDLE_FACTOR = True

    # Default parameters for FAI data preparation
    DEFAULT_FAI_BAND_FILTER = "alpha" # Note: This is a filter value, not just a column name part
    DEFAULT_FAI_METRIC_PREFIX = "fai_"
    DEFAULT_FAI_ID_VARS_COL = 'participant_id'
    DEFAULT_FAI_VALUE_NAME_OUTPUT = 'fai_value'
    DEFAULT_FAI_PAIR_CAPTURE_GROUP = 'hemisphere_pair'
    DEFAULT_FAI_CONDITION_CAPTURE_GROUP = 'condition'
    DEFAULT_FAI_COL_PATTERN_TEMPLATE = r"^{metric_prefix_val}{band_filter_val}_(?P<{pair_group_val}>F[pAF]?\d_F[pAF]?\d)_(?P<{condition_group_val}>.*)$"


    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ANOVAAnalyzer initialized.")

    def perform_rm_anova(self, data_df, dv, within, subject, effsize=DEFAULT_ANOVA_EFFSIZE, detailed=DEFAULT_ANOVA_DETAILED):
        """
        Performs a Repeated Measures ANOVA.
        Args:
            data_df (pd.DataFrame): DataFrame in long format.
            dv (str): Dependent variable column name.
            within (str or list): Within-subject factor column name(s).
            subject (str): Subject identifier column name.
            effsize (str): Effect size to compute. Defaults to ANOVAAnalyzer.DEFAULT_ANOVA_EFFSIZE.
            detailed (bool): Whether to return detailed output. Defaults to ANOVAAnalyzer.DEFAULT_ANOVA_DETAILED.
        Returns:
            pd.DataFrame: ANOVA results table, or None if error.
        """
        self.logger.info(f"ANOVAAnalyzer - Performing RM ANOVA: DV='{dv}', Within='{within}', Subject='{subject}'.")
        try:
            required_cols = [dv, subject]
            if isinstance(within, str): required_cols.append(within)
            elif isinstance(within, list): required_cols.extend(within)
            
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                self.logger.error(f"ANOVAAnalyzer - Missing columns for RM ANOVA: {missing_cols}")
                return None
            
            # Drop rows with NaNs in relevant columns to avoid errors in pingouin
            data_df_cleaned = data_df.dropna(subset=required_cols)
            if data_df_cleaned.empty:
                self.logger.warning("ANOVAAnalyzer - DataFrame is empty after dropping NaNs for RM ANOVA.")
                return None

            aov_results = pg.rm_anova(data=data_df_cleaned, dv=dv, within=within, subject=subject, 
                                      detailed=detailed, effsize=effsize)
            self.logger.info("ANOVAAnalyzer - RM ANOVA completed.")
            return aov_results
        except Exception as e:
            self.logger.error(f"ANOVAAnalyzer - Error performing RM ANOVA: {e}", exc_info=True)
            return None

    def perform_posthoc_tests(self, data_df, dv, within, subject, p_adjust=DEFAULT_POSTHOC_P_ADJUST, effsize=DEFAULT_POSTHOC_EFFSIZE):
        """
        Performs pairwise t-tests as post-hoc tests, typically after a significant RM ANOVA.
        Args:
            data_df (pd.DataFrame): DataFrame in long format.
            dv (str): Dependent variable column name.
            within (str or list): Within-subject factor column name(s) for pairwise comparisons.
            subject (str): Subject identifier column name.
            p_adjust (str): Method for p-value correction. Defaults to ANOVAAnalyzer.DEFAULT_POSTHOC_P_ADJUST.
            effsize (str): Effect size to compute for pairwise tests. Defaults to ANOVAAnalyzer.DEFAULT_POSTHOC_EFFSIZE.
        Returns:
            pd.DataFrame: Post-hoc test results, or None if error or no significant factor.
        """
        self.logger.info(f"ANOVAAnalyzer - Performing post-hoc tests: DV='{dv}', Within='{within}', Subject='{subject}', P-adjust='{p_adjust}'.")
        try:
            required_cols = [dv, subject]
            if isinstance(within, str): required_cols.append(within)
            elif isinstance(within, list): required_cols.extend(within)
            
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                self.logger.error(f"ANOVAAnalyzer - Missing columns for post-hoc tests: {missing_cols}")
                return None

            data_df_cleaned = data_df.dropna(subset=required_cols)
            if data_df_cleaned.empty:
                self.logger.warning("ANOVAAnalyzer - DataFrame is empty after dropping NaNs for post-hoc tests.")
                return None

            posthoc_results = pg.pairwise_tests(data=data_df_cleaned, dv=dv, within=within, subject=subject, padjust=p_adjust, effsize=effsize)
            self.logger.info("ANOVAAnalyzer - Post-hoc tests completed.")
            return posthoc_results
        except Exception as e:
            self.logger.error(f"ANOVAAnalyzer - Error performing post-hoc tests: {e}", exc_info=True)
            return None

    def prepare_plv_data_for_anova(self, df_agg, 
                                   eeg_band_filter, 
                                   autonomic_type_filter,
                                   metric_prefix=DEFAULT_PLV_METRIC_PREFIX,
                                   id_vars_col=DEFAULT_PLV_ID_VARS_COL,
                                   value_name_output=DEFAULT_PLV_VALUE_NAME_OUTPUT,
                                   condition_capture_group_name=DEFAULT_PLV_CONDITION_CAPTURE_GROUP,
                                   middle_factor_capture_group_name=DEFAULT_PLV_MIDDLE_FACTOR_CAPTURE_GROUP,
                                   allow_optional_middle_factor=DEFAULT_PLV_ALLOW_OPTIONAL_MIDDLE_FACTOR):
        """
        Prepares (melts and parses) PLV data from wide aggregated format to long format for ANOVA.
        This method is designed to be flexible with column naming conventions.

        Args:
            df_agg (pd.DataFrame): DataFrame with aggregated PLV data in wide format.
            eeg_band_filter (str): Specific EEG band to filter for (e.g., "Alpha").
            autonomic_type_filter (str): Specific autonomic type to filter for (e.g., "HRV").
            metric_prefix (str): The prefix of the PLV columns. Defaults to ANOVAAnalyzer.DEFAULT_PLV_METRIC_PREFIX.
            id_vars_col (str): Name of the participant identifier column. Defaults to ANOVAAnalyzer.DEFAULT_PLV_ID_VARS_COL.
            value_name_output (str): Name for the column containing PLV values. Defaults to ANOVAAnalyzer.DEFAULT_PLV_VALUE_NAME_OUTPUT.
            condition_capture_group_name (str): Regex capture group name for condition. Defaults to ANOVAAnalyzer.DEFAULT_PLV_CONDITION_CAPTURE_GROUP.
            middle_factor_capture_group_name (str): Regex capture group name for middle factor. Defaults to ANOVAAnalyzer.DEFAULT_PLV_MIDDLE_FACTOR_CAPTURE_GROUP.
            allow_optional_middle_factor (bool): If regex handles optional middle factor. Defaults to ANOVAAnalyzer.DEFAULT_PLV_ALLOW_OPTIONAL_MIDDLE_FACTOR.
        Returns:
            pd.DataFrame: DataFrame in long format.
        """
        if df_agg is None or df_agg.empty:
            self.logger.warning("ANOVAAnalyzer - Input df_agg is empty or None for PLV data preparation.")
            return pd.DataFrame()
        if not isinstance(eeg_band_filter, str) or not eeg_band_filter.strip():
            self.logger.error("ANOVAAnalyzer - eeg_band_filter must be a non-empty string for PLV data preparation.")
            return pd.DataFrame()
        if not isinstance(autonomic_type_filter, str) or not autonomic_type_filter.strip():
            self.logger.error("ANOVAAnalyzer - autonomic_type_filter must be a non-empty string for PLV data preparation.")
            return pd.DataFrame()
        if id_vars_col not in df_agg.columns:
            self.logger.error(f"ANOVAAnalyzer - ID column '{id_vars_col}' not found in df_agg for PLV data preparation.")
            return pd.DataFrame()

        self.logger.info(f"ANOVAAnalyzer - Preparing PLV data for EEG band '{eeg_band_filter}', autonomic type '{autonomic_type_filter}'.")

        # This pattern assumes the structure: prefix_band_(optional_middleFactor_)autonomic_condition
        escaped_prefix = re.escape(metric_prefix)
        escaped_band = re.escape(eeg_band_filter)
        escaped_autonomic = re.escape(autonomic_type_filter)

        if allow_optional_middle_factor:
            # Allows for an optional middle factor (e.g., an ROI)
            # Example: plv_avg_Alpha_DLPFC_HRV_Positive or plv_avg_Alpha_HRV_Positive
            pattern_str = rf"^{escaped_prefix}{escaped_band}_(?:(?P<{middle_factor_capture_group_name}>.*?)_)?{escaped_autonomic}_(?P<{condition_capture_group_name}>.*)$"
        else:
            # Expects no middle factor
            # Example: plv_avg_Alpha_HRV_Positive
            pattern_str = rf"^{escaped_prefix}{escaped_band}_{escaped_autonomic}_(?P<{condition_capture_group_name}>.*)$"
        
        plv_cols_to_melt = []
        parsed_col_data = []

        for col_name in df_agg.columns:
            if col_name == id_vars_col:
                continue
            match = re.match(pattern_str, col_name)
            if match:
                plv_cols_to_melt.append(col_name)
                group_dict = match.groupdict()
                parsed_data_for_col = {'metric_full_name': col_name}
                parsed_data_for_col.update(group_dict)
                parsed_col_data.append(parsed_data_for_col)

        if not plv_cols_to_melt:
            self.logger.warning(f"ANOVAAnalyzer - No PLV columns found matching pattern for prefix '{metric_prefix}', band '{eeg_band_filter}', autonomic '{autonomic_type_filter}'. Pattern: {pattern_str}")
            return pd.DataFrame() # Return empty DataFrame

        df_plv_long = df_agg[[id_vars_col] + plv_cols_to_melt].melt(
            id_vars=id_vars_col,
            value_vars=plv_cols_to_melt,
            var_name='metric_full_name', 
            value_name=value_name_output
        )
        df_plv_long.dropna(subset=[value_name_output], inplace=True)
        if df_plv_long.empty:
            self.logger.warning(f"ANOVAAnalyzer - No valid PLV data after melt for band '{eeg_band_filter}', autonomic '{autonomic_type_filter}'.")
            return pd.DataFrame()
        
        # Merge the parsed factor columns
        if parsed_col_data:
            df_parsed_factors = pd.DataFrame(parsed_col_data)
            df_plv_long = pd.merge(df_plv_long, df_parsed_factors, on='metric_full_name', how='left')
        
        # Add original band and autonomic type for reference, though they are implicit in the selection
        df_plv_long['eeg_band_filter'] = eeg_band_filter
        df_plv_long['autonomic_type_filter'] = autonomic_type_filter

        # Ensure the main condition column was parsed
        if condition_capture_group_name not in df_plv_long.columns:
            self.logger.error(f"ANOVAAnalyzer - Condition column '{condition_capture_group_name}' not found after parsing. Check regex and group names.")
            return pd.DataFrame()
        df_plv_long.dropna(subset=[condition_capture_group_name], inplace=True)
        return df_plv_long

    def prepare_fai_data_for_anova(self, df_agg, 
                                   band_name_filter=DEFAULT_FAI_BAND_FILTER,
                                   metric_prefix=DEFAULT_FAI_METRIC_PREFIX,
                                   id_vars_col=DEFAULT_FAI_ID_VARS_COL,
                                   value_name_output=DEFAULT_FAI_VALUE_NAME_OUTPUT,
                                   pair_capture_group_name=DEFAULT_FAI_PAIR_CAPTURE_GROUP,
                                   condition_capture_group_name=DEFAULT_FAI_CONDITION_CAPTURE_GROUP,
                                   # Default regex pattern for FAI columns like "fai_alpha_F4_F3_Positive".
                                   fai_col_pattern_template=DEFAULT_FAI_COL_PATTERN_TEMPLATE):
        """
        Prepares (melts and parses) FAI data from wide aggregated format to long format for ANOVA.
        """
        if df_agg is None or df_agg.empty:
            self.logger.warning("ANOVAAnalyzer - Input df_agg is empty or None for FAI data preparation.")
            return pd.DataFrame()
        if not isinstance(band_name_filter, str) or not band_name_filter.strip():
            self.logger.error("ANOVAAnalyzer - band_name_filter must be a non-empty string for FAI data preparation.")
            return pd.DataFrame()
        if id_vars_col not in df_agg.columns:
            self.logger.error(f"ANOVAAnalyzer - ID column '{id_vars_col}' not found in df_agg for FAI data preparation.")
            return pd.DataFrame()

        self.logger.info(f"ANOVAAnalyzer - Preparing FAI data for band '{band_name_filter}' for ANOVA.")
        
        # Construct the specific regex pattern to find columns and parse them
        col_selection_prefix = f"{metric_prefix}{band_name_filter}_"
        fai_cols = [col for col in df_agg.columns if col.startswith(col_selection_prefix)]
        
        if not fai_cols:
            self.logger.warning(f"ANOVAAnalyzer - No FAI columns found with prefix '{col_selection_prefix}'.")
            return pd.DataFrame()
        
        # The fai_col_pattern_template is for the *full* column name.
        parsing_regex = fai_col_pattern_template.format(
            metric_prefix_val=re.escape(metric_prefix),
            band_filter_val=re.escape(band_name_filter), # This makes the band part fixed in the regex
            pair_group_val=pair_capture_group_name,
            condition_group_val=condition_capture_group_name
        )

        df_fai_long = df_agg[[id_vars_col] + fai_cols].melt(
            id_vars=id_vars_col,
            value_vars=fai_cols,
            var_name='fai_metric_full',
            value_name=value_name_output
        )
        df_fai_long.dropna(subset=[value_name_output], inplace=True)
        if df_fai_long.empty:
            self.logger.warning("ANOVAAnalyzer - No valid FAI data after melt and NaN drop.")
            return pd.DataFrame()

        # Apply the regex to parse out factors
        parsed_data = df_fai_long['fai_metric_full'].str.extract(parsing_regex)
        
        for group_name in [pair_capture_group_name, condition_capture_group_name]:
            if group_name not in parsed_data.columns:
                self.logger.warning(f"ANOVAAnalyzer - Capture group '{group_name}' not found in FAI parsed data. Regex might be incorrect or not matching. Regex: {parsing_regex}")
                parsed_data[group_name] = pd.NA 

        df_fai_long = pd.concat([df_fai_long.reset_index(drop=True), parsed_data.reset_index(drop=True)], axis=1)
        
        # Drop rows where essential factors (pair, condition) couldn't be parsed
        df_fai_long.dropna(subset=[pair_capture_group_name, condition_capture_group_name], inplace=True)
        df_fai_long['band_name_filter'] = band_name_filter # Add the band used for filtering

        return df_fai_long

    def perform_mixed_anova(self, data_df, dv, within, subject, between, effsize=DEFAULT_ANOVA_EFFSIZE, detailed=DEFAULT_ANOVA_DETAILED):
        """
        Performs a Mixed Design ANOVA.
        Args:
            data_df (pd.DataFrame): DataFrame in long format.
            dv (str): Dependent variable column name.
            within (str or list): Within-subject factor column name(s).
            subject (str): Subject identifier column name.
            between (str or list): Between-subject factor column name(s).
            effsize (str): Effect size to compute. Defaults to ANOVAAnalyzer.DEFAULT_ANOVA_EFFSIZE.
            detailed (bool): Whether to return detailed output. Defaults to ANOVAAnalyzer.DEFAULT_ANOVA_DETAILED.
        Returns:
            pd.DataFrame: ANOVA results table, or None if error.
        """
        self.logger.info(f"ANOVAAnalyzer - Mixed ANOVA: DV='{dv}', Within='{within}', Subject='{subject}', Between='{between}'.")
        try:
            required_cols = [dv, subject]
            if isinstance(within, str): required_cols.append(within)
            elif isinstance(within, list): required_cols.extend(within)
            if isinstance(between, str): required_cols.append(between)
            elif isinstance(between, list): required_cols.extend(between)
            
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                self.logger.error(f"ANOVAAnalyzer - Missing columns for Mixed ANOVA: {missing_cols}")
                return None
            
            data_df_cleaned = data_df.dropna(subset=required_cols)
            if data_df_cleaned.empty:
                self.logger.warning("ANOVAAnalyzer - DataFrame is empty after dropping NaNs for Mixed ANOVA.")
                return None

            aov_results = pg.mixed_anova(data=data_df_cleaned, dv=dv, within=within, subject=subject, between=between,
                                         detailed=detailed, effsize=effsize) # type: ignore
            self.logger.info("ANOVAAnalyzer - Mixed ANOVA completed.")
            return aov_results
        except Exception as e:
            self.logger.error(f"ANOVAAnalyzer - Error performing Mixed ANOVA: {e}", exc_info=True)
            return None

    def perform_between_subjects_anova(self, data_df, dv, between, effsize=DEFAULT_ANOVA_EFFSIZE, detailed=DEFAULT_ANOVA_DETAILED):
        """
        Performs a Between-Subjects ANOVA.
        Args:
            data_df (pd.DataFrame): DataFrame in long format.
            dv (str): Dependent variable column name.
            between (str or list): Between-subject factor column name(s).
            effsize (str): Effect size to compute. Defaults to ANOVAAnalyzer.DEFAULT_ANOVA_EFFSIZE.
            detailed (bool): Whether to return detailed output. Defaults to ANOVAAnalyzer.DEFAULT_ANOVA_DETAILED.
        Returns:
            pd.DataFrame: ANOVA results table, or None if error.
        """
        self.logger.info(f"ANOVAAnalyzer - Between-Subjects ANOVA: DV='{dv}', Between='{between}'.")
        try:
            required_cols = [dv]
            if isinstance(between, str): required_cols.append(between)
            elif isinstance(between, list): required_cols.extend(between)
            
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                self.logger.error(f"ANOVAAnalyzer - Missing columns for Between-Subjects ANOVA: {missing_cols}")
                return None
            
            data_df_cleaned = data_df.dropna(subset=required_cols)
            if data_df_cleaned.empty:
                self.logger.warning("ANOVAAnalyzer - DataFrame is empty after dropping NaNs for Between-Subjects ANOVA.")
                return None

            aov_results = pg.anova(data=data_df_cleaned, dv=dv, between=between, detailed=detailed, effsize=effsize)
            self.logger.info("ANOVAAnalyzer - Between-Subjects ANOVA completed.")
            return aov_results
        except Exception as e:
            self.logger.error(f"ANOVAAnalyzer - Error performing Between-Subjects ANOVA: {e}", exc_info=True)
            return None