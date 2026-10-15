import os
from pathlib import Path
import chromadb

DB_PATH = "./local_search_db"

def dedupe():
    print(">>> Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="multimodal_files")

    total_count = collection.count()
    print(f">>> Fetching {total_count} records from database...")

    records = collection.get(include=["metadatas"])
    ids = records.get("ids", [])
    metadatas = records.get("metadatas", [])

    seen_canonical_paths = {}
    ids_to_delete = []

    for rec_id, meta in zip(ids, metadatas):
        if not meta or not meta.get("path"):
            ids_to_delete.append(rec_id)
            continue

        # Canonical normalization: convert double backslashes, lowercase
        clean_path = os.path.normpath(meta["path"]).lower().strip()

        # Check for duplicate path formats OR duplicate Syncthing vault copies
        if clean_path in seen_canonical_paths:
            ids_to_delete.append(rec_id)
        else:
            seen_canonical_paths[clean_path] = rec_id

    if ids_to_delete:
        print(f">>> Found {len(ids_to_delete)} duplicate/malformed entries out of {total_count}.")
        print(">>> Pruning duplicates from ChromaDB...")
        
        # Batch delete duplicates
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i + batch_size]
            collection.delete(ids=batch)
            
        print(f">>> Cleaned! Remaining unique records: {collection.count()}")
    else:
        print(">>> No duplicates found in database.")

if __name__ == "__main__":
    dedupe()