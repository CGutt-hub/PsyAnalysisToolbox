import pyxdf
import numpy as np
import json
from typing import Any, Dict, List, Union, Optional

import datetime # Import datetime for timestamp
import sys # Import sys for stdout redirection
import io # Import io for StringIO
# Define maximum display widths for channel details columns
MAX_LABEL_DISPLAY_WIDTH = 15
MAX_TYPE_DISPLAY_WIDTH = 10
MAX_UNIT_DISPLAY_WIDTH = 15
MAX_SCALE_DISPLAY_WIDTH = 10

def inspect_xdf(file_path: str, show_all_channels: bool, show_all_full_json: bool) -> None:
    """
    Loads an XDF file and prints detailed, nicely formatted information about each stream,
    including data integrity checks. Displays channel details and full JSON based on global flags.
    """
    # Capture original stdout
    original_stdout = sys.stdout # Store the original stdout
    # Redirect stdout to an in-memory text buffer
    output_buffer = io.StringIO()
    sys.stdout = output_buffer # All print() calls will now go to the buffer
    
    # This outer try block wraps almost the entire function to ensure finally is always executed
    try: 
        print(f"\n{'='*70}")
        print(f"{'XDF Stream Inspector':^70}")
        print(f"{'='*70}\n")
        print(f"Attempting to load XDF file: {file_path}\n")

        try: # Inner try block for XDF loading (errors here will be printed to buffer)
            streams, header = pyxdf.load_xdf(file_path)
        except FileNotFoundError:
            print(f"ERROR: File not found at '{file_path}'. Please check the path.")
            return # Exit function, output_buffer will be handled by outer finally block
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading XDF file: {e}")
            return # Exit function, output_buffer will be handled by outer finally block
        
        if not streams:
            print("No streams found in the XDF file. The file might be empty or corrupted.")
            return # Exit function, output_buffer will be handled by outer finally

        print(f"Successfully loaded XDF file. Found {len(streams)} stream(s).\n")

        # Helper to safely get nested values
        def get_nested_value(data_dict: Any, path: List[Union[str, int]], default: str = 'N/A') -> str:
            """
            Safely retrieves a nested value from a dictionary or list using a list of keys/indices.
            Handles missing keys, None values, and type mismatches gracefully.
            """
            current = data_dict
            for part in path:
                if current is None:
                    return default
                
                if isinstance(current, dict):
                    current = current.get(part, None)
                elif isinstance(current, list):
                    if isinstance(part, int):
                        try:
                            current = current[part]
                        except IndexError:
                            return default
                    else: # Tried to use string key on a list
                        return default
                else: # current is neither dict nor list, cannot traverse further
                    return default

            # Handle the final value
            if current is None:
                return default
            if isinstance(current, list) and current:
                return str(current[0]) # Take first element if it's a non-empty list
            return str(current)

        for i, stream in enumerate(streams):
            stream_info: Dict[str, Any] = stream.get('info', {})
            stream_name = stream_info.get('name', ['N/A'])[0]
            stream_type = stream_info.get('type', ['N/A'])[0]
            
            print(f"\n{'='*70}")
            print(f"--- Stream {i+1}: {stream_name} (Type: {stream_type}) ---")
            print(f"{'-'*70}")

            print("\n  [Basic Stream Information]")
            print(f"    Name: {stream_name}")
            print(f"    Type: {stream_type}")
            print(f"    Source ID: {stream_info.get('source_id', ['N/A'])[0]}")
            print(f"    Nominal Sampling Rate: {stream_info.get('nominal_srate', ['N/A'])[0]} Hz")
            
            # Effective Sampling Rate (approx)
            effective_srate_val = stream_info.get('effective_srate')
            if effective_srate_val is not None:
                print(f"    Effective Sampling Rate (approx): {effective_srate_val:.2f} Hz")
            else:
                print("    Effective Sampling Rate (approx): Not available")

            # Channel Count
            channel_count_str = stream_info.get('channel_count', ['N/A'])[0]
            channel_count = 0
            try:
                channel_count = int(channel_count_str)
            except (ValueError, TypeError):
                pass # Keep channel_count as 0 if conversion fails
            print(f"    Channel Count: {channel_count_str}")

            print("\n  [Data Integrity Check]")
            data_array = stream['time_series']
            num_samples: int = 0
            if isinstance(data_array, np.ndarray):
                num_samples = data_array.shape[0]
            elif isinstance(data_array, list):
                num_samples = len(data_array)
            
            print(f"    Number of Samples/Events: {num_samples}")
            if num_samples == 0:
                print("    Status: No data samples found in this stream.")
            else:
                if isinstance(data_array, np.ndarray) and np.issubdtype(data_array.dtype, np.number):
                    num_nans = np.sum(np.isnan(data_array))
                    num_infs = np.sum(np.isinf(data_array))
                    total_elements = data_array.size

                    if num_nans == 0 and num_infs == 0:
                        print("    Status: Data appears clean (no NaNs or Infs detected).")
                    else:
                        print("    Status: Potential data issues detected!")
                        if num_nans > 0:
                            print(f"      - Contains {num_nans} NaN (Not a Number) values ({num_nans/total_elements:.2%}).")
                        if num_infs > 0:
                            print(f"      - Contains {num_infs} Inf (Infinity) values ({num_infs/total_elements:.2%}).")
                else:
                    print("    Status: Data is not a numeric NumPy array (e.g., markers, strings). No NaN/Inf check performed.")
            
            print(f"\n  [Detailed Stream Metadata (Extracted for Readability)]")
            
            # Check for 'desc' and its contents before trying to extract deeper metadata
            desc_info = stream_info.get('desc')
            if isinstance(desc_info, list) and desc_info and isinstance(desc_info[0], dict):
                desc_dict = desc_info[0]

                # Acquisition Details
                acquisition_info = desc_dict.get('acquisition')
                if isinstance(acquisition_info, list) and acquisition_info and isinstance(acquisition_info[0], dict):
                    print("\n    [Acquisition Details]")
                    print(f"      Manufacturer: {get_nested_value(desc_dict, ['acquisition', 0, 'manufacturer', 0])}")
                    print(f"      Model: {get_nested_value(desc_dict, ['acquisition', 0, 'model', 0])}")
                    print(f"      Serial Number: {get_nested_value(desc_dict, ['acquisition', 0, 'serial_number', 0])}")
                
                # Amplifier Settings
                amplifier_info = desc_dict.get('amplifier')
                if isinstance(amplifier_info, list) and amplifier_info and isinstance(amplifier_info[0], dict):
                    print("\n    [Amplifier Settings]")
                    print(f"      Resolution: {get_nested_value(desc_dict, ['amplifier', 0, 'settings', 0, 'resolution', 0])}")
                    print(f"      Resolution Factor: {get_nested_value(desc_dict, ['amplifier', 0, 'settings', 0, 'resolutionfactor', 0])}")
                    print(f"      DC Coupling: {get_nested_value(desc_dict, ['amplifier', 0, 'settings', 0, 'dc_coupling', 0])}")
                    print(f"      Low Impedance Mode: {get_nested_value(desc_dict, ['amplifier', 0, 'settings', 0, 'low_impedance_mode', 0])}")
                
                # Version Information
                versions_info = desc_dict.get('versions')
                if isinstance(versions_info, list) and versions_info and isinstance(versions_info[0], dict):
                    print("\n    [Version Information]")
                    print(f"      LSL Protocol Version: {get_nested_value(desc_dict, ['versions', 0, 'lsl_protocol', 0])}")
                    print(f"      LibLSL Version: {get_nested_value(desc_dict, ['versions', 0, 'liblsl', 0])}")
                    print(f"      App Version: {get_nested_value(desc_dict, ['versions', 0, 'App', 0])}")
            else:
                print("    No detailed metadata (acquisition, amplifier, versions) found in stream description.")

            # --- Channel Details (Conditional Display based on global flag) ---
            if channel_count > 0 and show_all_channels:
                print("\n  [Channel Details]")
                channel_details_list = []
                try:
                    channels_desc = stream_info.get('desc', [{}])[0].get('channels', [{}])[0].get('channel', [])
                    if isinstance(channels_desc, list): # Ensure it's a list
                        for ch_idx, ch in enumerate(channels_desc):
                            if isinstance(ch, dict):
                                # Robustly get label, type, unit, and scaling_factor
                                label_val = ch.get('label', ['N/A'])[0]
                                label = str(label_val) if label_val is not None else 'N/A'
        
                                ch_type_val = ch.get('type', ['N/A'])[0]
                                ch_type = str(ch_type_val) if ch_type_val is not None else 'N/A'
        
                                unit_val = ch.get('unit', ['N/A'])[0]
                                unit = str(unit_val) if unit_val is not None else 'N/A'
        
                                scaling_factor_raw_val = ch.get('scaling_factor', ['N/A'])[0]
                                
                                scaling_factor = "N/A"
                                if scaling_factor_raw_val is not None:
                                    try:
                                        scaling_factor = f"{float(scaling_factor_raw_val):.6f}" # Format to 6 decimal places
                                    except ValueError:
                                        # If conversion to float fails, use the string representation if it's not empty
                                        if isinstance(scaling_factor_raw_val, str) and scaling_factor_raw_val.strip():
                                            scaling_factor = scaling_factor_raw_val.strip()
                                        # else: it remains "N/A"
        
                                channel_details_list.append({
                                    "Label": label,
                                    "Type": ch_type,
                                    "Unit": unit,
                                    "Scale": scaling_factor
                                })
                except (IndexError, TypeError, AttributeError):
                    pass # Labels not found or parsing failed, channel_details_list remains empty
                
                if channel_details_list: # Only print if there are channel details
                    # Calculate actual content width first
                    actual_max_label_len = max(len(d['Label']) for d in channel_details_list)
                    actual_max_type_len = max(len(d['Type']) for d in channel_details_list)
                    actual_max_unit_len = max(len(d['Unit']) for d in channel_details_list)
                    actual_max_scale_len = max(len(d['Scale']) for d in channel_details_list)
        
                    # Then determine final column width, considering header and max display width
                    final_label_col_width = min(max(actual_max_label_len, len("Label")), MAX_LABEL_DISPLAY_WIDTH)
                    final_type_col_width = min(max(actual_max_type_len, len("Type")), MAX_TYPE_DISPLAY_WIDTH)
                    final_unit_col_width = min(max(actual_max_unit_len, len("Unit")), MAX_UNIT_DISPLAY_WIDTH)
                    final_scale_col_width = min(max(actual_max_scale_len, len("Scale")), MAX_SCALE_DISPLAY_WIDTH)
        
                    # Print header
                    print(f"    {'Label':<{final_label_col_width}}  {'Type':<{final_type_col_width}}  {'Unit':<{final_unit_col_width}}  {'Scale':<{final_scale_col_width}}")
                    print(f"    {'-'*final_label_col_width}  {'-'*final_type_col_width}  {'-'*final_unit_col_width}  {'-'*final_scale_col_width}")
        
                    # Print rows
                    for ch_detail in channel_details_list:
                        label_to_print = ch_detail['Label']
                        type_to_print = ch_detail['Type']
                        unit_to_print = ch_detail['Unit']
                        scale_to_print = ch_detail['Scale']
        
                        # Apply truncation if necessary
                        if len(label_to_print) > final_label_col_width:
                            label_to_print = label_to_print[:final_label_col_width-3] + "..."
                        if len(type_to_print) > final_type_col_width:
                            type_to_print = type_to_print[:final_type_col_width-3] + "..."
                        if len(unit_to_print) > final_unit_col_width:
                            unit_to_print = unit_to_print[:final_unit_col_width-3] + "..."
                        if len(scale_to_print) > final_scale_col_width:
                            scale_to_print = scale_to_print[:final_scale_col_width-3] + "..."
        
                        print(f"    {label_to_print:<{final_label_col_width}}  {type_to_print:<{final_type_col_width}}  {unit_to_print:<{final_unit_col_width}}  {scale_to_print:<{final_scale_col_width}}")
                else:
                    print("    No detailed channel information available in stream description.")
            elif channel_count > 0: # Channels exist, but user chose not to show all
                print("\n  Channel details hidden (chosen not to show all).")
            else: # No channels to show
                print("\n  [Channel Details]")
                print("    No channels to display for this stream.")

            # --- Full JSON (Conditional Display based on global flag) ---
            if stream_info and show_all_full_json:
                print(f"\n  [Full Raw Stream Info Dictionary (for advanced details and custom configuration)]")
                print(f"    This section contains the complete metadata for the stream in raw JSON format.")
                print(f"    It's useful for finding specific, less common keys or nested values for your configuration.")
                try:
                    print(json.dumps(stream_info, indent=2, ensure_ascii=False, default=str))
                except Exception as e_json:
                    print(f"    Could not pretty-print full info dict: {e_json}")
                    print(f"    Raw info dict: {stream_info}")
            elif stream_info: # stream_info exists, but user chose not to show all
                print("\n  Full Raw Stream Info Dictionary hidden (chosen not to show all).")
            else: # No stream_info to show
                print("\n  No Full Raw Stream Info Dictionary available for this stream.")
            
            print(f"\n{'='*70}")

    finally:
        # Print the collected buffer content to the actual console
        original_stdout.write(output_buffer.getvalue())
        
        # Restore original stdout before asking to save log
        sys.stdout = original_stdout
        
        # Ask to save log
        save_log_input = input("\nDo you want to save this output to a log file? (y/n): ").strip().lower()
        if save_log_input == 'y':
            log_content = output_buffer.getvalue() # Get content again, as it was printed to original_stdout
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"xdf_inspector_log_{timestamp}.txt"
            try:
                with open(log_filename, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                print(f"Output saved to {log_filename}")
            except Exception as e:
                print(f"ERROR: Could not save log file: {e}")
        else:
            print("Log not saved.")

        # Ensure buffer is closed
        output_buffer.close()


if __name__ == "__main__":
    print("\n--- XDF File Inspector ---")
    print("Please enter the full path to the XDF file you want to inspect.")
    print("Example: D:\\rawData\\EV_P001\\EV_P001.xdf")
    
    xdf_file_path = input("Enter XDF file path: ").strip()

    show_all_channels_pref = False
    if input("\nDo you want to see detailed channel lists for ALL streams? (y/n): ").strip().lower() == 'y':
        show_all_channels_pref = True

    show_all_full_json_pref = False
    if input("Do you want to see the Full Raw Stream Info Dictionary for ALL streams? (y/n): ").strip().lower() == 'y':
        show_all_full_json_pref = True

    inspect_xdf(xdf_file_path, show_all_channels_pref, show_all_full_json_pref)

    print("\nInspection complete. Use the 'Stream Name', 'Type', and 'Channel Labels' ")
    print("details to configure your pipeline's data loaders.")