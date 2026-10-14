import os
import glob
import time
from pathlib import Path
import threading
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import chromadb
from pypdf import PdfReader
from docx import Document
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# -------------------------------------------------------------------
# 1. Configuration & Paths
# -------------------------------------------------------------------
print(">>> SERVICE LAUNCHING: Initializing components...", flush=True)

MODEL_ID = "google/embeddinggemma-2"
DB_PATH = "./local_search_db"

INDEX_PATHS = [
    r"C:\users\raviw\syncthing",
    r"B:\\",  # Bookshelf SMB Drive
    r"D:\\",  # Documents SMB Drive
]

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}
SKIP_DIRS = {".git", ".venv", "$recycle.bin", "appdata", "rwpi_backups", "node_modules"}

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

print(f">>> Loading {MODEL_ID} on {device} ({torch_dtype})...", flush=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch_dtype).to(device)

print(">>> Initializing ChromaDB persistence...", flush=True)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="multimodal_files")

print(">>> System ready. Starting FastAPI server on port 8000.", flush=True)

# -------------------------------------------------------------------
# 2. Text Extraction Helpers
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

# -------------------------------------------------------------------
# 3. Vector Embedding Generators
# -------------------------------------------------------------------
def get_file_embedding(file_path: str):
    ext = Path(file_path).suffix.lower()
    
    if ext in [".png", ".jpg", ".jpeg"]:
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
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
# 4. Ingestion / Crawler Routine
# -------------------------------------------------------------------
def index_directories():
    print(">>> BACKGROUND CRAWLER: Starting file sweep...", flush=True)
    start_time = time.time()
    count = 0
    scanned_folders = 0
    
    for root_dir in INDEX_PATHS:
        if not os.path.exists(root_dir):
            print(f">>> SKIPPING UNMOUNTED PATH: {root_dir}", flush=True)
            continue
            
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            scanned_folders += 1
            
            # Periodic heartbeats every 50 folders scanned so service.log updates even if 0 new files found
            if scanned_folders % 50 == 0:
                elapsed_sec = int(time.time() - start_time)
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
                print(f"Scanning active... Scanned {scanned_folders} folders | Last folder: {root} | New files indexed: {count}", flush=True)

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in ALLOWED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    
                    existing = collection.get(ids=[full_path])
                    if len(existing["ids"]) > 0:
                        continue
                        
                    vec = get_file_embedding(full_path)
                    if vec:
                        collection.add(
                            ids=[full_path],
                            embeddings=[vec],
                            metadatas=[{
                                "path": full_path, 
                                "filename": file, 
                                "type": "image" if ext in [".png", ".jpg", ".jpeg"] else "text"
                            }]
                        )
                        count += 1
                        if count % 10 == 0:
                            elapsed_sec = int(time.time() - start_time)
                            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
                            print(f"Indexed {count} new files in {elapsed_str} | Last folder worked on: {root}", flush=True)
                            
    total_elapsed = time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start_time)))
    print(f">>> CRAWLER SWEEP COMPLETE. Indexed {count} new files across {scanned_folders} folders in {total_elapsed}.", flush=True)

# -------------------------------------------------------------------
# 5. FastAPI Endpoints & Search UI
# -------------------------------------------------------------------
app = FastAPI()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=index_directories, daemon=True).start()

@app.get("/api/search")
def search(q: str = Query(...), limit: int = 10):
    query_vec = get_query_embedding(q)
    results = collection.query(query_embeddings=[query_vec], n_results=limit)
    return results

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
        <title>Local Multimodal Search</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f4f4f9; }
            .container { max-width: 700px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            input { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
            .result-item { padding: 12px; border-bottom: 1px solid #eee; text-decoration: none; display: block; color: inherit; }
            .result-item:hover { background: #f8f9fa; }
            .badge { display: inline-block; padding: 3px 8px; font-size: 12px; font-weight: bold; border-radius: 4px; color: #fff; background: #007bff; }
            .badge-image { background: #28a745; }
            .path { font-size: 13px; color: #666; word-break: break-all; margin-top: 4px; }
            .filename { font-size: 16px; color: #007bff; font-weight: 600; text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Multimodal Local Search</h2>
            <input type="text" id="query" placeholder="Type query (e.g. 'project notes', 'sunset beach')..." onkeydown="if(event.key==='Enter') runSearch()">
            <div id="results"></div>
        </div>

        <script>
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
                document.getElementById('results').innerHTML = html || '<p>No matching files found.</p>';
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)