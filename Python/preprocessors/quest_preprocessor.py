import polars as pl, sys, os, re

usage = lambda: print("[PREPROC] Usage: python quest_preprocessor.py <input_parquet.parquet> [level_key] [pair_delimiter] [key_value_delimiter]") or sys.exit(1)
get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_quest.parquet"


def preprocess_run(input_parquet, level_key='Level', key_value_delimiter=':'):
    """
    Transform flat key:value text lines into nested hierarchy structure.
    
    Steps:
    1. Parquet → Polars: Load parquet and extract text column
    2. Parse to (key, value) tuples: Split by delimiter, drop invalid lines
    3. Group by level: Nest based on level markers
    4. Handle recursion: Duplicate keys create new dicts in list
    5. Polars → Parquet: Write output
    """
    print(f"[PREPROC] Questionnaire preprocessor started for: {input_parquet}")
    
    # Step 1: Parquet → Polars
    df = pl.read_parquet(input_parquet)
    print(f"[PREPROC] Read DataFrame with shape: {df.shape}")
    
    # Extract text column
    if 'text' in df.columns:
        lines = df['text'].to_list()
    elif 'line' in df.columns:
        lines = df['line'].to_list()
    else:
        lines = df.to_series(0).to_list()
    
    lines = [str(line).strip() for line in lines if line is not None and str(line).strip()]
    print(f"[PREPROC] Extracted {len(lines)} text lines")
    
    # Step 2: Parse to (key, value) tuples (all strings)
    level_pattern = re.compile(rf'^\s*{re.escape(level_key)}\s*{re.escape(key_value_delimiter)}\s*(\d+)\s*$', re.IGNORECASE)
    
    parsed_entries = []
    for line in lines:
        # Check if it's a level marker
        match = level_pattern.match(line)
        if match:
            parsed_entries.append((level_key, match.group(1)))  # Store level as string
            continue
        
        # Try to split by delimiter
        if key_value_delimiter in line:
            key, value = line.split(key_value_delimiter, 1)
            parsed_entries.append((key.strip(), value.strip()))
        else:
            print(f"[PREPROC] Dropping non key:value line: '{line}'")
    
    print(f"[PREPROC] Parsed {len(parsed_entries)} key:value entries")
    
    # Steps 3 & 4: Group by level with nesting and recursion handling
    def build_hierarchy(entries):
        """
        Build nested hierarchy with level-based nesting and recursion handling.
        Returns list of dicts (outermost is always a list).
        """
        if not entries:
            return []
        
        result = []
        stack = [(0, result)]  # (level_num, current_list)
        current_dict = {}
        current_level = 0
        
        for key, value in entries:
            if key == level_key:
                # Level marker - determines nesting depth
                new_level = int(value)
                
                # Save current dict BEFORE changing levels
                if current_dict:
                    stack[-1][1].append(current_dict)
                
                # Pop stack until we're at the right parent level
                while len(stack) > 1 and stack[-1][0] >= new_level:
                    stack.pop()
                
                # If level increases, create _branch in last dict of parent level
                if new_level > stack[-1][0]:
                    parent_list = stack[-1][1]
                    if parent_list:
                        # Nest under last dict in parent
                        if '_branch' not in parent_list[-1]:
                            parent_list[-1]['_branch'] = []
                        stack.append((new_level, parent_list[-1]['_branch']))
                    else:
                        # Empty parent list - shouldn't happen but handle it
                        stack.append((new_level, parent_list))
                
                # Start new dict with Level key
                current_dict = {level_key: value}
                current_level = new_level
                
            else:
                # Regular key:value pair
                # Handle recursion: if key exists, save dict and start new
                if key in current_dict:
                    stack[-1][1].append(current_dict)
                    current_dict = {key: value}
                else:
                    current_dict[key] = value
        
        # Append last dict if it exists
        if current_dict:
            stack[-1][1].append(current_dict)
        
        return result
    
    hierarchy_list = build_hierarchy(parsed_entries)
    
    print(f"[PREPROC] Built hierarchy with {len(hierarchy_list)} top-level entries")
    
    # DEBUG: Show first entry before Polars
    if hierarchy_list:
        import json
        print("[DEBUG] First entry before Polars:")
        print(json.dumps(hierarchy_list[0], indent=2))
    
    # Step 5: Polars → Parquet
    output_name = get_output_filename(input_parquet)
    output_df = pl.DataFrame({
        'hierarchy': [{'level': hierarchy_list}]
    })
    output_df.write_parquet(output_name)
    
    print(f"[PREPROC] Questionnaire preprocessing finished. Output: {output_name}")
    return output_name



if __name__ == '__main__':
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_file = args[1]
            # Pipeline passes: <level_key> . <key_value_delimiter>
            # Accept flexible positions and ignore '.' placeholders
            extra = [a for a in args[2:] if a != '.']
            level_key = extra[0] if len(extra) >= 1 else 'Level'
            kv_delimiter = extra[1] if len(extra) >= 2 else ':'
            
            try:
                preprocess_run(input_file, level_key=level_key, key_value_delimiter=kv_delimiter)
            except Exception as e:
                # On error, write diagnostic parquet
                out_name = get_output_filename(input_file) if input_file else 'failed_quest.parquet'
                try:
                    print(f"[PREPROC] ERROR: {e}")
                    safe_hierarchy = {level_key.lower(): []}
                    err_df = pl.DataFrame({'hierarchy': [safe_hierarchy], 'error': [str(e)]})
                    err_df.write_parquet(out_name)
                    print(f"[PREPROC] Wrote diagnostic output to {out_name}")
                    sys.exit(1)
                except Exception as ee:
                    print(f"[PREPROC] FATAL: Could not write diagnostic parquet: {ee}")
                    sys.exit(2)
    except Exception as e:
        print(f"[PREPROC] ERROR: {e}")
        sys.exit(1)
