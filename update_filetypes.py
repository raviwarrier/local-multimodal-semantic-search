import json
from pathlib import Path

FILETYPES_FILE = Path("./config_filetypes.json")

def set_filetypes(ext_list):
    """
    Normalizes extensions (ensuring lowercase and leading dots) and saves to config_filetypes.json.
    """
    clean_exts = sorted(list(set(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" 
        for ext in ext_list if ext.strip()
    )))
    
    with open(FILETYPES_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_exts, f, indent=2)
    print(f">>> Successfully updated {len(clean_exts)} allowed extension(s) in {FILETYPES_FILE.name}")

if __name__ == "__main__":
    # Specify all supported file extensions
    allowed_extensions = [
        ".txt", ".md", ".pdf", ".docx",
        ".png", ".jpg", ".jpeg"
    ]
    
    set_filetypes(allowed_extensions)