import sys
import json
from pathlib import Path

PATHS_FILE = Path("./config_paths.json")

def set_index_paths(paths_list):
    """Overwrites config_paths.json with the provided list of directory paths."""
    clean_paths = [str(Path(p).resolve()) if Path(p).exists() else p for p in paths_list]
    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_paths, f, indent=2)
    print(f">>> Successfully updated {len(clean_paths)} index path(s) in {PATHS_FILE.name}")

if __name__ == "__main__":
    # If paths are passed as arguments, use them
    if len(sys.argv) > 1:
        target_paths = sys.argv[1:]
    else:
        # Fallback default list if no arguments are provided
        target_paths = [
            r"C:\Users\raviw\syncthing",
            r"\\192.168.68.102\Sys"
        ]
    
    set_index_paths(target_paths)