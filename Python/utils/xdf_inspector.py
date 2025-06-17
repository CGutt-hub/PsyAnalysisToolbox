import pyxdf
import argparse
import json # For pretty printing the info dictionary
import numpy as np # Import numpy to check array type
from typing import Any, Dict, List # For type hinting

def inspect_xdf(file_path: str) -> None:
    """
    Loads an XDF file and prints detailed information about each stream.
    """
    print(f"Attempting to load XDF file: {file_path}\n")
    try:
        streams, header = pyxdf.load_xdf(file_path)
    except Exception as e:
        print(f"Error loading XDF file: {e}")
        return

    print(f"Successfully loaded XDF file. Found {len(streams)} stream(s).\n")
    print("="*50)

    for i, stream in enumerate(streams):
        print(f"\n--- Stream {i+1} ({stream.get('info', {}).get('name', ['N/A'])[0]}) ---") # Add name to header
        
        stream_info: Dict[str, Any] = stream.get('info', {})
        
        # Print selected fields from stream['info'] for quick overview
        print("\nKey Stream Info:")
        print(f"  Name: {stream_info.get('name', ['N/A'])[0]}")
        print(f"  Type: {stream_info.get('type', ['N/A'])[0]}")
        print(f"  Channel Count (from info): {stream_info.get('channel_count', ['N/A'])[0]}")
        print(f"  Nominal Sampling Rate: {stream_info.get('nominal_srate', ['N/A'])[0]}")
        print(f"  Effective Sampling Rate (approx): {stream_info.get('effective_srate', 'N/A')}")
        
        # Attempt to print channel labels if available
        try:
            channels_desc = stream_info.get('desc', [{}])[0].get('channels', [{}])[0].get('channel', [])
            if channels_desc and isinstance(channels_desc, list):
                channel_labels = [ch.get('label', ['N/A'])[0] for ch in channels_desc if isinstance(ch, dict)]
                if channel_labels:
                    print(f"  Channel Labels (from desc): {channel_labels}")
        except (IndexError, TypeError, AttributeError):
            print("  Channel Labels (from desc): Not found or could not parse.")
        
        # Check if time_series is a numpy array before accessing shape/ndim
        print("\nTime Series Data:")
        if isinstance(stream['time_series'], np.ndarray):
            print(f"  Shape: {stream['time_series'].shape}")
            print(f"  Number of channels (from data shape): {stream['time_series'].shape[1] if stream['time_series'].ndim > 1 else 1}")
        else:
            print("  Data is not a standard NumPy array (e.g., markers, strings).")
        print(f"  Number of samples/events: {len(stream['time_series'])}")
        
        # Print the full stream['info'] dictionary for detailed inspection
        print("\nFull Stream Info Dictionary (info):")
        try:
            print(json.dumps(stream_info, indent=4, ensure_ascii=False, default=str)) # Use default=str for non-serializable
        except Exception as e_json:
            print(f"  Could not pretty-print full info dict: {e_json}")
            print(f"  Raw info dict: {stream_info}")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect streams within an XDF file.")
    # Make the argument optional and set the default to the specified pilot file path
    parser.add_argument("--xdf_file_path", type=str,
                        default=r"D:\pilotRawData\EV_P005\EV_P005.xdf",
                        help="Path to the .xdf file to inspect (defaults to a pilot file if not specified).")
    
    args = parser.parse_args()
    
    inspect_xdf(args.xdf_file_path)

    print("\nInspection complete. Use the 'Stream Info' details (especially 'name', 'type', ")
    print("and channel labels within 'desc') to configure your DataLoader in the pipeline.")