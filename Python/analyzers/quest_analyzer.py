import polars as pl
import sys
import os

def main():
    if len(sys.argv) < 5:
        print("Usage: python quest_analyzer.py <input_parquet> <questionnaire_types> <questionnaire_patterns> <output_prefix>")
        sys.exit(1)
    
    input_parquet = sys.argv[1]
    questionnaire_types = sys.argv[2]
    questionnaire_patterns = sys.argv[3]
    output_prefix = sys.argv[4]
    
    # Read input data
    df = pl.read_parquet(input_parquet)
    
    if 'key' not in df.columns:
        print(f"[QUEST] ERROR: Expected key-value format with 'key' column not found")
        sys.exit(1)
    
    # Process questionnaire types
    quest_types = questionnaire_types.split(',')
    
    # Extract questionnaire responses
    all_responses = []
    for row in df.to_dicts():
        if any(q_type.lower() in row["key"].lower() for q_type in quest_types):
            all_responses.append(row)
    
    # Create structured output with plot metadata
    analysis_data = []
    for questionnaire_type in quest_types:
        for row in all_responses:
            if any(q_type.lower() in row["key"].lower() for q_type in [questionnaire_type]):
                analysis_data.append({
                    "questionnaire_type": questionnaire_type,
                    "item_id": row["key"],
                    "question_text": row["key"],
                    "response_value": row["value"],
                    "scale_numeric": int(row["value"]) if str(row["value"]).isdigit() else 0,
                    "x_axis": f"{questionnaire_type} Items",
                    "y_axis": "Response Value",
                    "plot_type": "bar",
                    "plot_weight": 1.0,
                    "x_scale": "categorical",
                    "y_scale": "linear",
                    "x_data": row["key"],
                    "y_data": int(row["value"]) if str(row["value"]).isdigit() else 0,
                    # Preserve trial context if available
                    **{k: v for k, v in row.items() if k in ["trial_number", "condition", "stimulus_file", "trigger", "procedure"]}
                })
    
    # Create DataFrame
    if analysis_data:
        structured_df = pl.DataFrame(analysis_data)
    else:
        structured_df = pl.DataFrame({
            "questionnaire_type": [], "item_id": [], "question_text": [],
            "response_value": [], "scale_numeric": [],
            "x_axis": [], "y_axis": [], "plot_type": [], "plot_weight": [],
            "x_scale": [], "y_scale": [], "x_data": [], "y_data": []
        })
    
    # Generate output filename
    output_filename = f"{os.path.splitext(os.path.basename(input_parquet))[0]}_{output_prefix}_analysis.parquet"
    
    # Single consolidated output line to minimize Nextflow verbosity
    print(f"[QUEST] {os.path.basename(input_parquet)} | Types: {questionnaire_types} | Prefix: {output_prefix} | Input: {df.shape[0]} rows → Responses: {len(all_responses)} → Analysis: {structured_df.shape[0]} rows → {output_filename}")
    
    # Save output
    structured_df.write_parquet(output_filename)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[QUEST] ERROR: {e}")
        sys.exit(1)