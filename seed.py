import os
import shutil
from app import RAW_DOCS_DIR, process_documents

def seed_corpus():
    print("=== Testing Seed Script ===")
    print("[+] Starting RAG Corpus Seeding...")
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)
    
    sample_files = ["lic.pdf", "fepr102.pdf"]
    for sample in sample_files:
        src = os.path.join(os.getcwd(), sample)
        if os.path.exists(src):
            dst = os.path.join(RAW_DOCS_DIR, sample)
            shutil.copy(src, dst)
            print(f"  [V] Copied {sample} to raw_documents/")
            
    print("\n[*] Processing and indexing corpus chunks...")
    result = process_documents(chunk_size=3000, chunk_overlap=600, strategy="Header-Aware Splitting")
    if result:
        print("  [V] ChromaDB Vector Store & BM25 Index successfully seeded!")
    else:
        print("  [X] Seeding failed: No documents processed.")

if __name__ == "__main__":
    seed_corpus()
