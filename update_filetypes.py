import argparse
import json
from pathlib import Path

CONFIG_FILE = Path("./config_filetypes.json")

def load_filetypes():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Error loading {CONFIG_FILE}: {e}")
    return []

def save_filetypes(filetypes):
    filetypes = sorted(list(set(filetypes)))
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(filetypes, f, indent=2)
    print(f"Successfully saved {len(filetypes)} extension(s) to {CONFIG_FILE.name}")

def format_ext(ext_str):
    ext = ext_str.strip().strip('"').strip("'").lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext

def main():
    parser = argparse.ArgumentParser(description="Manage allowed file extensions in config_filetypes.json")
    parser.add_argument("--add", "-a", nargs="+", help="One or more file extensions to append")
    parser.add_argument("--remove", "-r", nargs="+", help="One or more file extensions to remove")
    parser.add_argument("--list", "-l", action="store_true", help="List all currently allowed file extensions")
    
    args = parser.parse_args()

    existing_types = set(load_filetypes())

    if args.list:
        sorted_exts = sorted(list(existing_types))
        print(f"\n--- Allowed File Extensions ({len(sorted_exts)}) ---")
        if not sorted_exts:
            print("  (No extensions configured)")
        else:
            print("  " + ", ".join(sorted_exts))
        print("----------------------------------------------\n")
        if not args.add and not args.remove:
            return

    modified = False

    # Process removals
    if args.remove:
        for r_ext in args.remove:
            clean_r = format_ext(r_ext)
            if clean_r in existing_types:
                existing_types.remove(clean_r)
                print(f"- Removed extension: {clean_r}")
                modified = True
            else:
                print(f"? Extension not found (skipped removal): {clean_r}")

    # Process additions (appends to existing)
    if args.add:
        for a_ext in args.add:
            clean_a = format_ext(a_ext)
            if clean_a not in existing_types:
                existing_types.add(clean_a)
                print(f"+ Appended extension: {clean_a}")
                modified = True
            else:
                print(f"= Extension already exists (skipped): {clean_a}")

    if modified:
        save_filetypes(list(existing_types))

if __name__ == "__main__":
    main()