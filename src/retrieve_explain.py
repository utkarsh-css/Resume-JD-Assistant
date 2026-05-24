import subprocess
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def retrieve_top_5(resume_text, index_path="embeddings/faiss_index.bin", metadata_path="embeddings/metadata.json"):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading FAISS index and metadata...")
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Error: Index or metadata not found. Run index.py first.")
        return []

    print("Generating resume embedding...")
    resume_embedding = model.encode([resume_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(resume_embedding)
    
    print("Searching for top-5 matches...")
    # Search top 5
    D, I = index.search(resume_embedding, 5)
    
    results = []
    for i in range(5):
        if I[0][i] == -1:
            continue # No match
        
        sim_score = D[0][i] # inner product range [-1, 1], though usually [0, 1] for text
        # Convert to percentage
        sim_percent = max(0.0, min(100.0, float(sim_score) * 100))
        
        idx = str(I[0][i])
        record = metadata.get(idx)
        if record:
            results.append({
                "job_title": record["job_title"],
                "similarity_percentage": round(sim_percent, 2),
                "combined_text": record["combined_text"],
                "job_id": record["id"]
            })
            
    return results

def generate_explanation(resume_text, job_record):
    print(f"Generating explanation using Ollama Mistral for top match: {job_record['job_title']}...")
    
    prompt = f"""You are an expert technical recruiter analyzing a candidate's fit for a role.
Please analyze the following Resume against the Job Description.

Job Title: {job_record['job_title']}
Job Match Percentage: {job_record['similarity_percentage']}%
Job Description: {job_record['combined_text']}

Candidate Resume:
{resume_text}

Provide output strictly in these four sections:
1. Explanation of Match
2. Matched Skills
3. Missing Skills
4. Improvement Suggestions
"""
    
    command = ['ollama', 'run', 'phi3:mini']
    
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt)
        
        if process.returncode != 0:
            return f"Error running Ollama: {stderr}"
        return stdout.strip()
            
    except Exception as e:
        return f"Error executing Ollama subprocess: {str(e)}"
