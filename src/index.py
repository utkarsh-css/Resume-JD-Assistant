import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_index():
    input_path = "processed/jobs_cleaned.json"
    index_path = "embeddings/faiss_index.bin"
    metadata_path = "embeddings/metadata.json"
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    print(f"Loading cleaned data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run ingest.py first.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records. Instantiating SentenceTransformer model...")
    # Load the specified model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Extract combined text fields but strongly weight the job title in the semantic representation
    # This prevents generic responsibilities from overpowering the specific role identity
    texts = [f"Job Title: {rec['job_title']}. Description: {rec['combined_text']}" for rec in records]
    
    print("Generating dense embeddings...")
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}, type: {embeddings.dtype}")
    
    # Convert to float32
    embeddings = embeddings.astype(np.float32)
    
    # Normalize embeddings to unit vectors for cosine similarity (Inner Product = Cosine Similarity for normalized vectors)
    print("Normalizing embeddings for cosine similarity using inner product...")
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    
    print(f"Initializing FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    
    print("Adding embeddings to FAISS index...")
    index.add(embeddings)
    
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    # Saving metadata map: position to record (so we can retrieve title/text)
    metadata = {}
    for i, rec in enumerate(records):
        metadata[str(i)] = rec
        
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"Index creation complete. Total vectors in index: {index.ntotal}")

if __name__ == "__main__":
    create_index()
