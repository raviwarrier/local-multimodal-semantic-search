# System Setup & Deployment Instructions (v2.6)

This document details step-by-step setup, configuration management, and system maintenance guidelines for the local search engine service.

---

## Architecture Overview

1. **Ingestion Loop**: Scans configured directories every 24 hours (`CRAWL_INTERVAL_SECONDS = 86400`). Adds new vector embeddings to ChromaDB, extracts faces via ArcFace (`InsightFace`), and prunes deleted records.
2. **Dynamic Configuration via CLI Helpers**: `update_paths.py` and `update_filetypes.py` safely update `config_paths.json` and `config_filetypes.json`, allowing decoupled path/type management independently from `server.py`.
3. **Cluster-First Hybrid Engine**:
   - Matches aliases in search queries case-insensitively.
   - Extracts associated cluster image paths.
   - For name-only searches, directly returns person image sets.
   - For descriptive queries, performs vector search constrained to person image sets with automated fallback.

---

## Step 1: Directory Setup

Create the target system folder structure:
```powershell
mkdir -p C:\Users\raviw\.localsearch\face_crops
mkdir -p C:\Users\raviw\.localsearch\local_search_db
```

---

## Step 2: Deploy CLI Helpers & Initialize Configurations

Create and run the CLI helper scripts to initialize `config_paths.json` and `config_filetypes.json`.

### 1. `update_paths.py`
Create `update_paths.py` in `C:\Users\raviw\.localsearch\`:
```python
import json
from pathlib import Path

PATHS_FILE = Path("./config_paths.json")

def set_index_paths(paths_list):
    with open(PATHS_FILE, "w") as f:
        json.dump(paths_list, f, indent=2)
    print(f"Successfully updated index paths in {PATHS_FILE.name}")

if __name__ == "__main__":
    new_paths = [
        r"C:\Users\raviw\syncthing",
        r"B:\\",
        r"D:\\"
    ]
    set_index_paths(new_paths)
```

Run to generate `config_paths.json`:
```powershell
python update_paths.py
```

### 2. `update_filetypes.py`
Create `update_filetypes.py` in `C:\Users\raviw\.localsearch\`:
```python
import json
from pathlib import Path

FILETYPES_FILE = Path("./config_filetypes.json")

def set_filetypes(ext_list):
    clean_exts = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in ext_list]
    with open(FILETYPES_FILE, "w") as f:
        json.dump(clean_exts, f, indent=2)
    print(f"Successfully updated allowed filetypes in {FILETYPES_FILE.name}")

if __name__ == "__main__":
    allowed = [".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg"]
    set_filetypes(allowed)
```

Run to generate `config_filetypes.json`:
```powershell
python update_filetypes.py
```

---

## Step 3: Persistent Windows Service (NSSM) Setup

1. Place `nssm.exe` in `C:\Users\raviw\.localsearch\`.
2. Open PowerShell as Administrator.
3. Configure and start the service:
   ```powershell
   cd C:\Users\raviw\.localsearch
   .\nssm.exe install LocalSearchService "C:\Users\raviw\AppData\Local\Programs\Python\Python310\python.exe" "C:\Users\raviw\.localsearch\server.py"
   .\nssm.exe set LocalSearchService AppDirectory "C:\Users\raviw\.localsearch"
   .\nssm.exe set LocalSearchService Start SERVICE_AUTO_START
   .\nssm.exe start LocalSearchService
   ```

---

## Step 4: Maintenance & Configuration Updates

### Updating Index Paths or Supported File Types
To add new drives or file extensions:
1. Open `update_paths.py` or `update_filetypes.py`.
2. Modify the target lists in the `if __name__ == "__main__":` block.
3. Execute the helper script (`python update_paths.py` or `python update_filetypes.py`).
4. The crawler loop will read the updated JSON configs on its next sweep without requiring a service restart.

### Checking Service Health & Logs
- Check service status: `.\nssm.exe status LocalSearchService`
- Restart service after updating `server.py`: `.\nssm.exe restart LocalSearchService`
- Access Web Interface: Open browser to `http://localhost:8000`