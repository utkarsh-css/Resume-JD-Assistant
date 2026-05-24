import os
import sys
import json
import faiss
from src.ingest import ingest_data
from src.index import create_index
from src.retrieve_explain import retrieve_top_5, generate_explanation

def main():
    print("=== RAG Resume-Job Semantic Matching System ===")
    
    # Check if processed JSON exists; if not, suggest running ingest
    if not os.path.exists("processed/jobs_cleaned.json"):
        print("Processed data not found. Running ingestion pipeline...")
        try:
            ingest_data()
        except ValueError as e:
            print(f"Ingestion failed: {e}")
            print("Please ensure the CSV dataset meets the strict requirements.")
            sys.exit(1)
            
    # Check if FAISS index exists; if not, run index creation
    if not os.path.exists("embeddings/faiss_index.bin"):
        print("FAISS index not found. Building index...")
        create_index()
        
    # Validation step as requested
    index_path = "embeddings/faiss_index.bin"
    metadata_path = "embeddings/metadata.json"
    processed_path = "processed/jobs_cleaned.json"
    
    if os.path.exists(index_path) and os.path.exists(processed_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        with open(processed_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        print(f"Validation: FAISS index size is {index.ntotal}. Processed dataset size is {len(processed_data)}.")
        if index.ntotal != len(processed_data):
            print("Warning: FAISS index size does not match dataset size!")
        else:
            print("Validation Passed: Index size perfectly matches processed dataset size.")
    else:
        print("Critical files are missing! Exiting.")
        sys.exit(1)
        
    # Interactive CLI Prompt
    print("\n" + "-"*50)
    print("Please paste your resume text below.")
    print("Press Enter, type 'EOF' on a new line, and press Enter again to finish:")
    resume_lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'EOF':
                break
            resume_lines.append(line)
        except EOFError:
            break
            
    resume_text = "\n".join(resume_lines).strip()
    
    if not resume_text:
        print("No resume text provided. Exiting.")
        sys.exit(0)
        
    print("\nRunning retrieval process...")
    top_5_jobs = retrieve_top_5(resume_text, index_path, metadata_path)
    
    if not top_5_jobs:
        print("Failed to retrieve jobs.")
        sys.exit(1)
        
    # Validation: ensure retrieval always returns five results
    if len(top_5_jobs) != 5:
        print(f"Warning: Expected 5 results, but got {len(top_5_jobs)}.")
    else:
        print("Validation Passed: Retrieval returned exactly 5 results.")
        
    print("\n" + "="*50)
    print("TOP 5 MATCHING JOBS:")
    print("="*50)
    for i, job in enumerate(top_5_jobs, start=1):
        print(f"{i}. {job['job_title']} (ID: {job['job_id']})")
        print(f"   Similarity: {job['similarity_percentage']}%")
        
    print("\n" + "="*50)
    # Generate explanation for the top-ranked job
    top_job = top_5_jobs[0]
    print(f"Generating detailed explanation for top match: {top_job['job_title']}...\n")
    
    explanation = generate_explanation(resume_text, top_job)
    print(explanation)
    print("="*50)

if __name__ == "__main__":
    main()
