import argparse
import json
import os
import sys
from pathlib import Path

CONFIG_FILE = Path("./config_paths.json")

def load_paths():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Error loading {CONFIG_FILE}: {e}")
    return []

def save_paths(paths):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    print(f"Successfully saved {len(paths)} path(s) to {CONFIG_FILE.name}")

def sanitize_path(path_str):
    """Strips quotes and normalizes slashes without breaking root drive letters."""
    clean = path_str.strip().strip('"').strip("'")
    norm = os.path.normpath(clean)
    return norm

def main():
    parser = argparse.ArgumentParser(description="Manage index paths in config_paths.json")
    parser.add_argument("--add", "-a", nargs="+", help="One or more paths to append")
    parser.add_argument("--remove", "-r", nargs="+", help="One or more paths to remove")
    parser.add_argument("--list", "-l", action="store_true", help="List all currently configured paths")
    
    args = parser.parse_args()

    # Load existing list from JSON
    existing_paths = load_paths()

    if args.list:
        print(f"\n--- Configured Paths ({len(existing_paths)}) ---")
        if not existing_paths:
            print("  (No paths configured)")
        else:
            for p in existing_paths:
                print(f"  • {p}")
        print("-----------------------------------\n")
        if not args.add and not args.remove:
            return

    # Use a case-insensitive map to track existing paths while preserving original formatting
    path_map = {sanitize_path(p).lower(): sanitize_path(p) for p in existing_paths}
    modified = False

    # Process removals
    if args.remove:
        for r_item in args.remove:
            clean_r = sanitize_path(r_item).lower()
            if clean_r in path_map:
                removed_val = path_map.pop(clean_r)
                print(f"- Removed path: {removed_val}")
                modified = True
            else:
                print(f"? Path not found (skipped removal): {r_item}")

    # Process additions (appends to existing)
    if args.add:
        for a_item in args.add:
            clean_a = sanitize_path(a_item)
            clean_a_lower = clean_a.lower()
            if clean_a_lower not in path_map:
                path_map[clean_a_lower] = clean_a
                print(f"+ Appended path: {clean_a}")
                modified = True
            else:
                print(f"= Path already exists (skipped): {clean_a}")

    # Save only if items were added or removed
    if modified:
        save_paths(list(path_map.values()))

if __name__ == "__main__":
    main()