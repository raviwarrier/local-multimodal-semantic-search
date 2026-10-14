import json
from pathlib import Path

PATHS_FILE = Path("./config_paths.json")

def set_index_paths(paths_list):
    """
    Overwrites config_paths.json with the provided list of directory paths.
    """
    clean_paths = [str(Path(p).resolve()) if Path(p).exists() else p for p in paths_list]
    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_paths, f, indent=2)
    print(f">>> Successfully updated {len(clean_paths)} index path(s) in {PATHS_FILE.name}")

if __name__ == "__main__":
    # Specify the local and network directories you want indexed
    target_paths = [
        r"C:\Users\raviw\syncthing",
        r"B:\\",  # Bookshelf SMB Drive
        r"D:\\"   # Documents SMB Drive
    ]
    
    set_index_paths(target_paths)