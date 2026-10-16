### This project is still under development. It is still buggy and has not been tested for security issues. DO NOT install it yet. This message will be removed once I am fairly convinced that it works as expected and is safe to use.

# Self-Hosted Local Multimodal Semantic Search & Face Tagging Engine

A privacy-first, fully automated local search service built for Windows. Combines face detection/clustering via **InsightFace (ArcFace)** with natural language multimodal embeddings via **Google EmbeddingGemma-2**, persisting vectors in **ChromaDB** and running seamlessly as a persistent Windows Service via **NSSM**.

---

## Key Features

- **Multimodal Semantic Search**: Natural language query processing across local images (`.png`, `.jpg`, `.jpeg`) and documents (`.txt`, `.md`, `.pdf`, `.docx`).
- **Cluster-First Face Identification**: Automatic facial extraction (`buffalo_l` model) and clustering. Assign aliases/nicknames to face clusters in a high-visibility modal gallery.
- **Robust Path & Alias Normalization**: Case-insensitive alias matching with strict Windows path normalization across ChromaDB and `face_clusters.json`.
- **Hybrid Search Fallback**: Direct person lookup for name-only queries (e.g., `Manasa`) and ranked vector search for descriptive queries (e.g., `Manasa in bikini`).
- **Automated 24-Hour Ingestion Cycle**: Automatic daily background crawls for additive file indexing and pruning of deleted or unmounted files.
- **Decoupled Configuration via CLI Helpers**: Manage index roots (`update_paths.py`) and extension filters (`update_filetypes.py`) dynamically without modifying core server logic.

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Vector Storage**: ChromaDB
- **Facial Analytics**: InsightFace (`buffalo_l` / ArcFace)
- **Multimodal AI**: Google `google/embeddinggemma-2` (PyTorch, Transformers)
- **Persistence / OS Integration**: NSSM (Non-Sucking Service Manager) on Windows

---

## Project Structure

```text
C:\Users\raviw\.localsearch\
├── server.py                 # Core FastAPI server, ingestion engine & web UI
├── update_paths.py           # CLI helper script to update config_paths.json
├── update_filetypes.py       # CLI helper script to update config_filetypes.json
├── config_paths.json         # Managed list of indexed directory roots
├── config_filetypes.json     # Managed list of allowed file extensions
├── face_clusters.json        # Managed mapping of person clusters and aliases
├── face_crops/               # Extracted crop thumbnails for cluster management
└── local_search_db/          # ChromaDB vector store directory
```

---

## Quick Start

### 1. Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for accelerated ArcFace and Gemma inference)

### 2. Install Dependencies
```powershell
pip install fastapi uvicorn chromadb insightface torch torchvision transformers pillow pypdf python-docx numpy opencv-python
```

### 3. Initialize Configuration

Run the configuration helper scripts to generate initial target paths and allowed filetypes:

```powershell
python update_paths.py
python update_filetypes.py
```

### 4. Run Server or Service

To start manually:
```powershell
python server.py
```

To run as a background Windows Service using NSSM:
```powershell
.\nssm.exe install LocalSearchService "C:\Users\raviw\AppData\Local\Programs\Python\Python310\python.exe" "C:\Users\raviw\.localsearch\server.py"
.\nssm.exe set LocalSearchService AppDirectory "C:\Users\raviw\.localsearch"
.\nssm.exe start LocalSearchService
```

---

## Usage

- Access the web app at `http://localhost:8000`.
- **Search Tab**: Type queries like `Manasa at the beach`, `Me in shorts`, or `Dog at home`.
- **Face Tagging Gallery**: View detected face clusters, inspect image associations, and save comma-separated aliases (e.g., `Ravi, Ravi Warrier, me, my`).

---

## Managing Configuration Changes

When adding new index directories or file formats:
1. Update parameters inside `update_paths.py` or `update_filetypes.py`.
2. Run `python update_paths.py` or `python update_filetypes.py`.
3. The background crawler reads updated JSON configs dynamically during every crawl cycle without requiring a server reboot.

---

## Vibe-Coding Disclosure

This project was built through interactive AI pair programming ("vibe-coding")—iteratively designing, refining, and troubleshooting architecture, face clustering, path normalization, and vector search strategies with an AI collaborator.

---

## License

MIT License - Free for personal and commercial use.
