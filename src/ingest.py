import os
import re
import json
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ingest_data():
    input_path = "data/jobs.csv"
    output_path = "processed/jobs_cleaned.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Dataset loaded. Total rows: {len(df)}")
    print("Columns in dataset:", list(df.columns))
    
    # Identify title column
    title_col = None
    for col in df.columns:
        if 'title' in col.lower():
            title_col = col
            break
            
    if not title_col:
        # Fallback if no title containing 'title'
        print("Warning: No column containing 'title' found. Using the first string column as a fallback or raise error.")
        # Requirements say: "Automatically detect the job title column by searching for a column name containing the word “title” (case insensitive)"
        raise ValueError("No column containing the word 'title' was found.")
        
    # Identify description column
    desc_cols = [col for col in df.columns if 'description' in col.lower()]
    if not desc_cols:
        # Fallback to responsibilities or skills if description is missing
        desc_cols = [col for col in df.columns if 'responsibilities' in col.lower() or 'skills' in col.lower()]
        
    desc_col = None
    if not desc_cols:
        raise ValueError("No valid description column found containing the word 'description' or 'responsibilities'.")
    elif len(desc_cols) == 1:
        desc_col = desc_cols[0]
    else:
        # Multiple description columns
        max_avg_length = -1
        for col in desc_cols:
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > max_avg_length:
                max_avg_length = avg_length
                desc_col = col
                
    print(f"Selected Job Title Column: '{title_col}'")
    print(f"Selected Job Description Column: '{desc_col}'")
    
    # Filtering null, empty, or short descriptions
    initial_count = len(df)
    df = df.dropna(subset=[desc_col])
    # Compute length of string representation
    df['desc_len'] = df[desc_col].astype(str).str.strip().str.len()
    df = df[df['desc_len'] >= 100].copy()
    print(f"Rows dropped due to null/empty/short description: {initial_count - len(df)}")
    
    # Clean text columns
    df['cleaned_title'] = df[title_col].apply(clean_text)
    df['cleaned_desc'] = df[desc_col].apply(clean_text)
    
    # Combine title and description
    df['combined_text'] = df['cleaned_title'] + ". " + df['cleaned_desc']
    
    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['combined_text'])
    print(f"Rows dropped due to exact duplicates in combined text: {before_dedup - len(df)}")
    
    # Limit to first 300 rows
    df = df.head(300).copy()
    print(f"Rows after limiting to 300: {len(df)}")
    
    # Assign integer ID
    df['clean_id'] = range(1, len(df) + 1)
    
    # Prepare JSON output
    output_records = []
    for _, row in df.iterrows():
        output_records.append({
            "id": row['clean_id'],
            "job_title": row['cleaned_title'], # Keep cleaned title separately as required "where each entry contains an integer ID, job title, and combined cleaned text"
            "combined_text": row['combined_text']
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed data. Saved to {output_path}.")
    
if __name__ == "__main__":
    ingest_data()
