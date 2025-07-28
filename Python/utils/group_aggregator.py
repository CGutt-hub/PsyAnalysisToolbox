import os
import pandas as pd

class GroupAggregator:
    def __init__(self, logger):
        self.logger = logger

    def aggregate(self, participant_data_list, config, output_dir=None):
        group_level_data_artifacts = {}
        for section_name in config.sections():
            if section_name.startswith('GroupLevel_Aggregate_'):
                step_name = section_name.replace('GroupLevel_Aggregate_', '')
                preproc_config = dict(config.items(section_name))
                method = preproc_config.get('method')
                output_key = preproc_config.get('output_artifact_key')
                artifact_data_key = preproc_config.get('artifact_data_key')
                if not all([method, output_key, artifact_data_key]):
                    self.logger.error(f"Skipping preprocessing step '{step_name}': config section is missing 'method', 'output_artifact_key', or 'artifact_data_key'.")
                    continue
                if method == 'concat_dataframes_from_artifacts':
                    self.logger.info(f"Running group preprocessing step: {step_name}")
                    dfs_to_concat = []
                    for p_data in participant_data_list:
                        if artifact_data_key in p_data and isinstance(p_data[artifact_data_key], pd.DataFrame):
                            df = p_data[artifact_data_key].copy()
                            if 'participant_id' not in df.columns:
                                df['participant_id'] = p_data['participant_id']
                            dfs_to_concat.append(df)
                        else:
                            self.logger.warning(f"Missing or invalid {artifact_data_key} for {p_data['participant_id']}")
                    if dfs_to_concat:
                        aggregated_df = pd.concat(dfs_to_concat, ignore_index=True)
                        group_level_data_artifacts[output_key] = aggregated_df
                        self.logger.info(f"Step '{step_name}' successful. Aggregated data shape: {aggregated_df.shape}. Stored as '{output_key}'.")
                        if output_dir and config.getboolean('GroupLevel', 'save_intermediate', fallback=False):
                            intermediate_path = os.path.join(output_dir, f"intermediate_{output_key}.csv")
                            aggregated_df.to_csv(intermediate_path, index=False)
                            self.logger.info(f"Saved intermediate DataFrame to {intermediate_path}")
                    else:
                        self.logger.warning(f"No data found for preprocessing step '{step_name}' with key '{artifact_data_key}'.")
        return group_level_data_artifacts 