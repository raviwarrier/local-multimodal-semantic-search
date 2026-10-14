import os
import glob
import time
import json
import base64
import re
from pathlib import Path
import threading
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import chromadb
from pypdf import PdfReader
from docx import Document
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import insightface
from insightface.app import FaceAnalysis

# -------------------------------------------------------------------
# 1. Configuration & Dynamic JSON Loaders
# -------------------------------------------------------------------
print(">>> SERVICE LAUNCHING v2.7: Robust Hybrid Search Engine...", flush=True)

MODEL_ID = "google/embeddinggemma-2"
DB_PATH = "./local_search_db"
FACE_CROPS_DIR = Path("./face_crops")
FACE_CROPS_DIR.mkdir(exist_ok=True)
FACE_DATA_FILE = Path("./face_clusters.json")

PATHS_CONFIG_FILE = Path("./config_paths.json")
FILETYPES_CONFIG_FILE = Path("./config_filetypes.json")

CRAWL_INTERVAL_SECONDS = 86400

DEFAULT_INDEX_PATHS = [
    r"C:\users\raviw\syncthing",
    r"B:\\",  # Bookshelf SMB Drive
    r"D:\\",  # Documents SMB Drive
]
DEFAULT_ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}

def normalize_path(path_str: str) -> str:
    """Standardize Windows path format for consistent dictionary lookups."""
    if not path_str:
        return ""
    return os.path.normpath(path_str).lower().strip()

def load_index_paths():
    if PATHS_CONFIG_FILE.exists():
        try:
            with open(PATHS_CONFIG_FILE, "r") as f:
                paths = json.load(f)
                if isinstance(paths, list) and paths:
                    print(f">>> Config Loaded: {len(paths)} index paths from {PATHS_CONFIG_FILE.name}", flush=True)
                    return paths
        except Exception as e:
            print(f">>> Failed to load {PATHS_CONFIG_FILE.name}: {e}. Using defaults.", flush=True)
    return DEFAULT_INDEX_PATHS

def load_allowed_extensions():
    if FILETYPES_CONFIG_FILE.exists():
        try:
            with open(FILETYPES_CONFIG_FILE, "r") as f:
                exts = json.load(f)
                if isinstance(exts, list) and exts:
                    print(f">>> Config Loaded: {len(exts)} file extensions from {FILETYPES_CONFIG_FILE.name}", flush=True)
                    return set(exts)
        except Exception as e:
            print(f">>> Failed to load {FILETYPES_CONFIG_FILE.name}: {e}. Using defaults.", flush=True)
    return DEFAULT_ALLOWED_EXTENSIONS

SKIP_DIRS = {".git", ".venv", "$recycle.bin", "appdata", "rwpi_backups", "node_modules"}

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

print(f">>> Loading {MODEL_ID} on {device} ({torch_dtype})...", flush=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch_dtype).to(device)

print(">>> Initializing InsightFace ArcFace model...", flush=True)
try:
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    print(f">>> InsightFace init notice: {e}. Falling back to CPU provider.", flush=True)
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

print(">>> Initializing ChromaDB persistence...", flush=True)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="multimodal_files")

# -------------------------------------------------------------------
# 2. Multi-Alias Face Clustering & Storage Logic
# -------------------------------------------------------------------
face_db = {"clusters": {}, "image_to_clusters": {}}

def load_face_db():
    global face_db
    if FACE_DATA_FILE.exists():
        try:
            with open(FACE_DATA_FILE, "r") as f:
                raw_db = json.load(f)
                
            cleaned_clusters = {}
            for cid, cdata in raw_db.get("clusters", {}).items():
                raw_aliases = cdata.get("aliases", [])
                if isinstance(cdata.get("name"), str) and cdata.get("name"):
                    raw_aliases.extend(cdata["name"].split(","))
                    
                unique_aliases = list(dict.fromkeys([a.strip() for a in raw_aliases if a and a.strip()]))
                cdata["aliases"] = unique_aliases
                cdata["name"] = ", ".join(unique_aliases)
                cleaned_clusters[cid] = cdata
                
            face_db["clusters"] = cleaned_clusters
            
            normalized_mapping = {}
            for img_path, clusters in raw_db.get("image_to_clusters", {}).items():
                clean_key = normalize_path(img_path)
                normalized_mapping[clean_key] = clusters
                
            face_db["image_to_clusters"] = normalized_mapping
            save_face_db()
            print(f">>> Loaded & Cleaned Face DB: {len(face_db['clusters'])} clusters, {len(face_db['image_to_clusters'])} mapped images.", flush=True)
        except Exception as e:
            print(f">>> Error loading face_clusters.json: {e}", flush=True)

def save_face_db():
    with open(FACE_DATA_FILE, "w") as f:
        json.dump(face_db, f, indent=2)

load_face_db()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def match_or_create_cluster(face_embedding, cropped_img, file_path):
    norm_path = normalize_path(file_path)
    emb_norm = face_embedding / np.linalg.norm(face_embedding)
    
    best_cluster = None
    best_sim = -1.0
    
    for cluster_id, data in face_db["clusters"].items():
        rep_emb = np.array(data["representative_emb"])
        sim = cosine_similarity(emb_norm, rep_emb)
        if sim > best_sim:
            best_sim = sim
            best_cluster = cluster_id
            
    if best_sim >= 0.45 and best_cluster:
        cluster_id = best_cluster
    else:
        cluster_id = f"person_{len(face_db['clusters']) + 1}"
        crop_filename = f"{cluster_id}.jpg"
        crop_file_path = FACE_CROPS_DIR / crop_filename
        cv2.imwrite(str(crop_file_path), cropped_img)
        
        face_db["clusters"][cluster_id] = {
            "aliases": [],
            "representative_emb": emb_norm.tolist(),
            "crop_path": str(crop_file_path)
        }
        
    if norm_path not in face_db["image_to_clusters"]:
        face_db["image_to_clusters"][norm_path] = []
        
    if cluster_id not in face_db["image_to_clusters"][norm_path]:
        face_db["image_to_clusters"][norm_path].append(cluster_id)
        
    save_face_db()
    return cluster_id

def process_image_faces(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            return
        faces = face_app.get(img)
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            cropped = img[y1:y2, x1:x2]
            match_or_create_cluster(face.embedding, cropped, file_path)
    except Exception as e:
        print(f">>> Face processing error on {file_path}: {e}", flush=True)

# -------------------------------------------------------------------
# 3. Text Extraction & Multimodal Embeddings
# -------------------------------------------------------------------
def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    try:
        if os.path.getsize(file_path) > 10 * 1024 * 1024:
            return ""
            
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(4000)
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages[:3]:
                text += page.extract_text() or ""
            return text[:4000]
        elif ext == ".docx":
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])[:4000]
    except Exception as e:
        print(f"Skipping corrupt or unreadable file {file_path}: {e}", flush=True)
    return ""

def get_file_embedding(file_path: str):
    ext = Path(file_path).suffix.lower()
    
    if ext in [".png", ".jpg", ".jpeg"]:
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            process_image_faces(file_path)
        except Exception as e:
            return None
    else:
        text_content = extract_text(file_path)
        if not text_content.strip():
            return None
            
        title = Path(file_path).name
        formatted_text = f"title: {title} | text: {text_content}"
        inputs = processor(text=formatted_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
    return embedding.cpu().to(torch.float32).numpy().tolist()[0]

def get_query_embedding(query_text: str):
    formatted_query = f"task: search result | query: {query_text}"
    inputs = processor(text=formatted_query, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
    return embedding.cpu().to(torch.float32).numpy().tolist()[0]

# -------------------------------------------------------------------
# 4. Recurring Ingestion & Automatic Pruning Routine
# -------------------------------------------------------------------
def start_scheduled_crawler():
    first_run = True
    while True:
        try:
            if not first_run:
                print(f">>> SCHEDULED CRAWLER: Sleeping for {CRAWL_INTERVAL_SECONDS // 3600} hours before next sweep...", flush=True)
                time.sleep(CRAWL_INTERVAL_SECONDS)
            first_run = False

            index_paths = load_index_paths()
            allowed_extensions = load_allowed_extensions()

            print(f"\n>>> CRAWLER SWEEP STARTING: Processing {len(index_paths)} root paths...", flush=True)
            start_time = time.time()
            count = 0
            scanned_folders = 0
            
            for root_dir in index_paths:
                if not os.path.exists(root_dir):
                    print(f">>> SKIPPING UNMOUNTED PATH: {root_dir}", flush=True)
                    continue
                    
                for root, dirs, files in os.walk(root_dir):
                    dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
                    scanned_folders += 1
                    
                    if scanned_folders % 50 == 0:
                        elapsed_sec = int(time.time() - start_time)
                        print(f"Scanning active... Scanned {scanned_folders} folders | Last folder: {root} | New files indexed: {count}", flush=True)

                    for file in files:
                        ext = Path(file).suffix.lower()
                        if ext in allowed_extensions:
                            full_path = os.path.join(root, file)
                            norm_path = normalize_path(full_path)
                            
                            existing = collection.get(ids=[norm_path])
                            if len(existing["ids"]) > 0:
                                continue
                                
                            vec = get_file_embedding(full_path)
                            if vec:
                                collection.add(
                                    ids=[norm_path],
                                    embeddings=[vec],
                                    metadatas=[{
                                        "path": full_path, 
                                        "filename": file, 
                                        "type": "image" if ext in [".png", ".jpg", ".jpeg"] else "text"
                                    }]
                                )
                                count += 1

            print(">>> CRAWLER: Checking for deleted files to prune...", flush=True)
            all_records = collection.get(include=["metadatas"])
            deleted_ids = []

            if all_records and all_records.get("ids"):
                for record_id in all_records["ids"]:
                    if not os.path.exists(record_id):
                        deleted_ids.append(record_id)

            if deleted_ids:
                print(f">>> PRUNING: Removing {len(deleted_ids)} deleted file(s) from vector index...", flush=True)
                collection.delete(ids=deleted_ids)
                
                for d_id in deleted_ids:
                    norm_d_id = normalize_path(d_id)
                    if norm_d_id in face_db["image_to_clusters"]:
                        del face_db["image_to_clusters"][norm_d_id]
                save_face_db()

            total_elapsed = time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start_time)))
            print(f">>> CRAWLER SWEEP COMPLETE. Indexed {count} new files, pruned {len(deleted_ids)} deleted files across {scanned_folders} folders in {total_elapsed}.", flush=True)
            
        except Exception as e:
            print(f">>> Error during crawler loop execution: {e}", flush=True)
            time.sleep(60)

# -------------------------------------------------------------------
# 5. FastAPI Endpoints & UI
# -------------------------------------------------------------------
app = FastAPI()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=start_scheduled_crawler, daemon=True).start()

@app.get("/api/search")
def search(q: str = Query(...), limit: int = 50):
    matched_cluster_ids = []
    clean_query = q.lower().strip()
    
    for cid, cdata in face_db["clusters"].items():
        clean_aliases = [a.strip().lower() for a in cdata.get("aliases", []) if a and a.strip()]
        for alias in clean_aliases:
            if alias in clean_query:
                matched_cluster_ids.append(cid)
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                clean_query = pattern.sub("", clean_query).strip()

    cluster_photo_paths = set()
    if matched_cluster_ids:
        for img_path, clusters in face_db["image_to_clusters"].items():
            if any(c in clusters for c in matched_cluster_ids):
                cluster_photo_paths.add(normalize_path(img_path))

    remaining_visual_prompt = re.sub(r'\b(in|and|at|wearing|of|with|a|the)\b', '', clean_query, flags=re.IGNORECASE).strip()

    if matched_cluster_ids and not remaining_visual_prompt:
        if not cluster_photo_paths:
            return {"metadatas": [[]]}
            
        direct_results = [
            {"path": p, "filename": Path(p).name, "type": "image"} 
            for p in cluster_photo_paths
        ]
        return {"metadatas": [direct_results[:limit]]}

    visual_search_term = remaining_visual_prompt if remaining_visual_prompt else "photo"
    query_vec = get_query_embedding(visual_search_term)

    results = collection.query(query_embeddings=[query_vec], n_results=200)

    filtered_metadatas = []
    if results.get("metadatas") and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            p_norm = normalize_path(meta["path"])
            if matched_cluster_ids:
                if p_norm in cluster_photo_paths:
                    filtered_metadatas.append(meta)
            else:
                filtered_metadatas.append(meta)

    if matched_cluster_ids and not filtered_metadatas:
        for p in cluster_photo_paths:
            filtered_metadatas.append({
                "path": p,
                "filename": Path(p).name,
                "type": "image"
            })

    return {"metadatas": [filtered_metadatas[:limit]]}

@app.get("/api/indexed_files")
def get_indexed_files():
    try:
        count = collection.count()
        if count == 0:
            return {"files": []}

        all_records = collection.get(limit=count, include=["metadatas"])
        metadatas = all_records.get("metadatas", []) or []

        files = []
        seen_paths = set()

        for meta in metadatas:
            if not meta or not meta.get("path"):
                continue

            file_path = meta["path"]
            if file_path in seen_paths:
                continue

            seen_paths.add(file_path)
            ext = Path(file_path).suffix.lower()

            if ext in [".png", ".jpg", ".jpeg"]:
                category = "image"
            elif ext == ".pdf":
                category = "pdf"
            elif ext in [".txt", ".md"]:
                category = "text"
            elif ext in [".docx", ".doc"]:
                category = "document"
            else:
                category = "other"

            files.append({
                "name": meta.get("filename") or Path(file_path).name,
                "path": file_path,
                "category": category
            })

        files.sort(key=lambda x: x["name"].lower())
        return {"files": files}
    except Exception as e:
        print(f">>> Error fetching indexed files: {e}", flush=True)
        return {"files": []}

@app.get("/api/clusters")
def get_clusters():
    result = []
    for cid, cdata in face_db["clusters"].items():
        matching_photos = [img for img, clist in face_db["image_to_clusters"].items() if cid in clist]
        display_aliases = ", ".join(cdata.get("aliases", []))
        
        result.append({
            "id": cid,
            "aliases": display_aliases,
            "crop_url": f"/crops/{cid}.jpg",
            "photo_count": len(matching_photos),
            "photos": matching_photos
        })
    return sorted(result, key=lambda x: x["photo_count"], reverse=True)

@app.post("/api/tag_cluster")
def tag_cluster(payload: dict = Body(...)):
    cluster_id = payload.get("cluster_id")
    raw_input = payload.get("name", "").strip()
    
    if cluster_id in face_db["clusters"]:
        aliases = list(dict.fromkeys([a.strip() for a in raw_input.split(",") if a and a.strip()]))
        face_db["clusters"][cluster_id]["aliases"] = aliases
        face_db["clusters"][cluster_id]["name"] = ", ".join(aliases)
        save_face_db()
        return {"status": "success", "cluster_id": cluster_id, "aliases": aliases}
        
    raise HTTPException(status_code=404, detail="Cluster not found")

@app.get("/crops/{crop_filename}")
def serve_crop(crop_filename: str):
    target = FACE_CROPS_DIR / crop_filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Crop image not found")
    return FileResponse(path=target)

@app.get("/files/{file_path:path}")
def serve_file(file_path: str):
    target = Path(file_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=target, filename=target.name)

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Local Multimodal Search v2.7</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f4f4f9; }
            .container { max-width: 950px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .nav { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            .nav-button { padding: 8px 16px; border: none; background: #e0e0e0; cursor: pointer; border-radius: 4px; font-weight: bold; }
            .nav-button.active { background: #007bff; color: white; }
            
            input { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
            .result-item { padding: 12px; border-bottom: 1px solid #eee; text-decoration: none; display: block; color: inherit; }
            .result-item:hover { background: #f8f9fa; }
            
            .badge { display: inline-block; padding: 3px 8px; font-size: 11px; font-weight: bold; border-radius: 4px; color: #fff; background: #007bff; text-transform: uppercase; }
            .badge-image { background: #28a745; }
            .badge-pdf { background: #dc3545; }
            .badge-text { background: #17a2b8; }
            .badge-document { background: #ffc107; color: #333; }
            
            .path { font-size: 13px; color: #666; word-break: break-all; margin-top: 4px; }
            .filename { font-size: 16px; color: #007bff; font-weight: 600; text-decoration: underline; }
            
            .filter-bar { display: flex; gap: 8px; margin-bottom: 15px; align-items: center; }
            .filter-btn { padding: 6px 14px; background: #eee; border: 1px solid #ccc; border-radius: 20px; cursor: pointer; font-size: 13px; }
            .filter-btn.active { background: #007bff; color: #fff; border-color: #007bff; }

            .cluster-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
            .cluster-card { border: 1px solid #ddd; padding: 12px; border-radius: 8px; text-align: center; background: #fafafa; }
            .crop-container { cursor: pointer; display: inline-block; position: relative; }
            .crop-container:hover img { border-color: #28a745; }
            .crop-container img { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #007bff; transition: border-color 0.2s; }
            .cluster-card input { width: 90%; margin-top: 8px; padding: 6px; font-size: 12px; text-align: center; }
            .cluster-card button { margin-top: 6px; padding: 5px 12px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
            .photo-count { font-size: 12px; color: #007bff; margin-top: 6px; cursor: pointer; text-decoration: underline; font-weight: bold; }
            
            .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); overflow-y: auto; }
            .modal-content { background: #fff; margin: 5% auto; padding: 20px; border-radius: 8px; max-width: 850px; width: 90%; }
            .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .close-btn { font-size: 24px; font-weight: bold; cursor: pointer; border: none; background: none; }
            
            .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-top: 15px; max-height: 550px; overflow-y: auto; }
            .gallery-grid img { width: 100%; height: 200px; object-fit: contain; background: #181818; border-radius: 6px; cursor: pointer; transition: transform 0.2s; }
            .gallery-grid img:hover { transform: scale(1.03); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav">
                <button class="nav-button active" onclick="switchTab('search', event)">Search</button>
                <button class="nav-button" onclick="switchTab('files', event)">Indexed Files</button>
                <button class="nav-button" onclick="switchTab('tagger', event)">Face Tagging Gallery</button>
            </div>

            <!-- Tab 1: Search -->
            <div id="search-tab">
                <h2>Multimodal Search</h2>
                <input type="text" id="query" placeholder="e.g., '[Person name] at the beach, Me in shorts, [Pet name] at home'..." onkeydown="if(event.key==='Enter') runSearch()">
                <div id="results" style="margin-top: 15px;"></div>
            </div>

            <!-- Tab 2: Indexed Files -->
            <div id="files-tab" style="display: none;">
                <h2>Indexed Documents & Media</h2>
                <div class="filter-bar">
                    <strong>Filter:</strong>
                    <button class="filter-btn active" onclick="filterFiles('all', event)">All</button>
                    <button class="filter-btn" onclick="filterFiles('image', event)">Images</button>
                    <button class="filter-btn" onclick="filterFiles('pdf', event)">PDFs</button>
                    <button class="filter-btn" onclick="filterFiles('text', event)">Text / Markdown</button>
                    <button class="filter-btn" onclick="filterFiles('document', event)">Docs</button>
                </div>
                <p id="file-status-label">Loading indexed files...</p>
                <div id="indexed-files-list"></div>
            </div>

            <!-- Tab 3: Face Tagging Gallery -->
            <div id="tagger-tab" style="display: none;">
                <h2>Face Clusters & Aliases</h2>
                <p>Click any thumbnail or photo count to inspect all photos in that cluster. Enter <code>name and/or nicknames</code> (comma-separated) to register search tags.</p>
                <div class="cluster-grid" id="cluster-grid"></div>
            </div>
        </div>

        <div id="gallery-modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="modal-title">Cluster Preview</h3>
                    <button class="close-btn" onclick="closeModal()">&times;</button>
                </div>
                <div class="gallery-grid" id="modal-gallery"></div>
            </div>
        </div>

        <script>
            let allClusters = [];
            let allIndexedFiles = [];

            function switchTab(tab, evt) {
                document.querySelectorAll('.nav-button').forEach(b => b.classList.remove('active'));
                evt.target.classList.add('active');

                document.getElementById('search-tab').style.display = 'none';
                document.getElementById('files-tab').style.display = 'none';
                document.getElementById('tagger-tab').style.display = 'none';

                if (tab === 'search') {
                    document.getElementById('search-tab').style.display = 'block';
                } else if (tab === 'files') {
                    document.getElementById('files-tab').style.display = 'block';
                    loadIndexedFiles();
                } else if (tab === 'tagger') {
                    document.getElementById('tagger-tab').style.display = 'block';
                    loadClusters();
                }
            }

            async function runSearch() {
                const q = document.getElementById('query').value;
                if(!q) return;
                const res = await fetch('/api/search?q=' + encodeURIComponent(q));
                const data = await res.json();
                let html = '';
                if(data.metadatas && data.metadatas[0]) {
                    data.metadatas[0].forEach((item) => {
                        const badgeClass = item.type === 'image' ? 'badge badge-image' : 'badge';
                        const fileUrl = '/files/' + encodeURIComponent(item.path);
                        html += `
                            <a class="result-item" href="${fileUrl}" target="_blank">
                                <span class="${badgeClass}">${item.type.toUpperCase()}</span>
                                <span class="filename">${item.filename}</span>
                                <div class="path">${item.path}</div>
                            </a>`;
                    });
                }
                document.getElementById('results').innerHTML = html || '<p>No matching files or photos found.</p>';
            }

            async function loadIndexedFiles() {
                const statusLabel = document.getElementById('file-status-label');
                statusLabel.innerText = 'Loading...';
                try {
                    const res = await fetch('/api/indexed_files');
                    const data = await res.json();
                    allIndexedFiles = data.files || [];
                    renderIndexedFiles(allIndexedFiles);
                } catch (err) {
                    statusLabel.innerText = 'Failed to load indexed files.';
                }
            }

            function renderIndexedFiles(files) {
                const statusLabel = document.getElementById('file-status-label');
                const listContainer = document.getElementById('indexed-files-list');
                listContainer.innerHTML = '';

                statusLabel.innerText = `Showing ${files.length} indexed item(s)`;

                if (files.length === 0) {
                    listContainer.innerHTML = '<p>No files match the selected filter.</p>';
                    return;
                }

                let html = '';
                files.forEach(item => {
                    let badgeClass = 'badge';
                    if (item.category === 'image') badgeClass += ' badge-image';
                    else if (item.category === 'pdf') badgeClass += ' badge-pdf';
                    else if (item.category === 'text') badgeClass += ' badge-text';
                    else if (item.category === 'document') badgeClass += ' badge-document';

                    const fileUrl = '/files/' + encodeURIComponent(item.path);

                    html += `
                        <a class="result-item" href="${fileUrl}" target="_blank">
                            <span class="${badgeClass}">${item.category.toUpperCase()}</span>
                            <span class="filename">${item.name}</span>
                            <div class="path">${item.path}</div>
                        </a>`;
                });
                listContainer.innerHTML = html;
            }

            function filterFiles(category, evt) {
                document.querySelectorAll('.filter-bar .filter-btn').forEach(btn => btn.classList.remove('active'));
                evt.target.classList.add('active');

                if (category === 'all') {
                    renderIndexedFiles(allIndexedFiles);
                } else {
                    const filtered = allIndexedFiles.filter(f => f.category === category);
                    renderIndexedFiles(filtered);
                }
            }

            async function loadClusters() {
                const res = await fetch('/api/clusters');
                allClusters = await res.json();
                let html = '';
                allClusters.forEach(c => {
                    html += `
                        <div class="cluster-card">
                            <div class="crop-container" onclick="openGallery('${c.id}')">
                                <img src="${c.crop_url}" alt="Face crop">
                            </div>
                            <div class="photo-count" onclick="openGallery('${c.id}')">${c.photo_count} photo(s) (Inspect)</div>
                            <input type="text" id="input-${c.id}" value="${c.aliases}" placeholder="name and/or nicknames">
                            <button onclick="saveTag('${c.id}')">Save Aliases</button>
                        </div>`;
                });
                document.getElementById('cluster-grid').innerHTML = html || '<p>No faces detected yet.</p>';
            }

            function openGallery(clusterId) {
                const cluster = allClusters.find(c => c.id === clusterId);
                if (!cluster) return;
                
                document.getElementById('modal-title').innerText = `Photos for ${cluster.aliases || cluster.id} (${cluster.photo_count} photos)`;
                let html = '';
                cluster.photos.forEach(path => {
                    const fileUrl = '/files/' + encodeURIComponent(path);
                    html += `<a href="${fileUrl}" target="_blank"><img src="${fileUrl}" title="${path}"></a>`;
                });
                document.getElementById('modal-gallery').innerHTML = html || '<p>No photos available.</p>';
                document.getElementById('gallery-modal').style.display = 'block';
            }

            function closeModal() {
                document.getElementById('gallery-modal').style.display = 'none';
            }

            async function saveTag(clusterId) {
                const rawName = document.getElementById(`input-${clusterId}`).value;
                const res = await fetch('/api/tag_cluster', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cluster_id: clusterId, name: rawName })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    alert(`Saved aliases: ${data.aliases.join(', ')}`);
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)