import streamlit as st
import fitz  # PyMuPDF
import re
import json
import faiss
import numpy as np
import subprocess
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

# 1. Page Configuration & Theme
st.set_page_config(page_title="AI Resume Intelligence Engine", layout="wide")

# Custom CSS for dark theme styling, gradients, cards, and badges
st.markdown("""
    <style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .header-container {
        text-align: center;
        padding: 3rem 1rem 1rem 1rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #d1e8ff;
        font-size: 1.2rem;
        font-weight: 400;
    }
    .subtle-divider {
        border-top: 1px solid #333;
        margin: 2rem 0;
    }
    .stats-card {
        background-color: #1a1c24;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        border: 1px solid #2d303e;
    }
    .match-card {
        background-color: #1a1c24;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .match-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5);
    }
    .match-card-accent {
        border: 2px solid #00d2ff;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.3);
    }
    .job-title-card {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.4em 0.8em;
        font-size: 0.9em;
        font-weight: 600;
        border-radius: 20px;
        margin: 0.2em;
    }
    .badge-matched { background-color: rgba(40, 167, 69, 0.2); color: #4ade80; border: 1px solid #28a745; }
    .badge-missing { background-color: rgba(220, 53, 69, 0.2); color: #ff6b6b; border: 1px solid #dc3545; }
    .badge-extra { background-color: rgba(0, 123, 255, 0.2); color: #60a5fa; border: 1px solid #007bff; }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #888;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid #333;
    }
    /* Clean spacing in tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Centered hero section
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Semantic Resume–Job Matching</h1>
        <p class="subtitle">Vector-powered AI matching using FAISS and local LLM reasoning</p>
    </div>
    <hr class="subtle-divider">
""", unsafe_allow_html=True)

# 7. Performance & Caching
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_models():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading FAISS Index...")
def load_faiss_index():
    index_path = "embeddings/faiss_index.bin"
    metadata_path = "embeddings/metadata.json"
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return index, metadata
    except Exception:
        return None, None

def extract_pdf_text(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_blocks = []
        for page in doc:
            # Extract blocks of text rather than raw streams to maintain localized context
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5 and block[4]:
                    text_blocks.append(block[4].strip())
        
        # Join blocks maintaining slight structure
        text = " \n ".join(text_blocks)
        
        # Clean up excessive newlines/whitespaces but keep paragraph bounds
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        
        return text
    except Exception as e:
        print(f"PDF extract error: {e}")
        return None

# Load Resources safely
embed_model = load_models()
faiss_index, metadata = load_faiss_index()

# 2. Layout Structure
left_col, right_col = st.columns([2, 1])

resume_text = ""

with left_col:
    st.subheader("📄 Resume Upload")
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
    
    if uploaded_file is not None:
        raw_text = extract_pdf_text(uploaded_file)
        if not raw_text:
            st.error("Failed to parse PDF. Please ensure it is a valid document and not corrupted.")
        else:
            resume_text = raw_text
            st.success("Resume parsed successfully!")
            st.write(f"**Character Count:** {len(resume_text)}")
            
            with st.expander("Resume Preview (First 500 chars)"):
                st.write(resume_text[:500] + "...")
                
            with st.expander("Expand Full Text"):
                st.write(resume_text)

with right_col:
    st.markdown('<div class="stats-card">', unsafe_allow_html=True)
    st.subheader("📊 System Stats")
    
    total_jobs = len(metadata) if metadata else 0
    st.metric(label="Total Indexed Jobs", value=total_jobs)
    st.metric(label="Embedding Model", value="all-MiniLM-L6-v2")
    st.metric(label="Vector DB", value="FAISS IndexFlatIP")
    st.metric(label="Similarity Metric", value="Cosine Similarity")
    st.metric(label="LLM Model", value="phi3:mini")
    st.metric(label="Top-K Retrieval", value=5)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

# 5. Retrieval Execution
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    if not resume_text:
        st.warning("Please upload and preview a valid resume PDF first.")
    elif not faiss_index or not metadata:
        st.error("FAISS index or metadata not found. Execute the ingestion backend pipeline first.")
    else:
        with st.spinner("Embedding resume and searching FAISS index..."):
            resume_embedding = embed_model.encode([resume_text], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(resume_embedding)
            
            D, I = faiss_index.search(resume_embedding, 5)
            
            top_matches = []
            for i in range(5):
                if I[0][i] == -1:
                    continue
                sim_score = D[0][i]
                sim_percent = max(0.0, min(100.0, float(sim_score) * 100))
                
                record = metadata.get(str(I[0][i]))
                if record:
                    top_matches.append({
                        "job_title": record["job_title"],
                        "similarity_percent": sim_percent,
                        "combined_text": record["combined_text"],
                        "id": record["id"]
                    })
        
        st.success(f"Retrieved {len(top_matches)} matches!")
        
        # 6. Results Section (Tabs)
        if top_matches:
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Top Matches", "📊 Similarity Dashboard", "⚡ Skill Gap Analysis", "🤖 AI Explanation"])
            
            with tab1:
                for idx, match in enumerate(top_matches):
                    accent_class = "match-card-accent" if idx == 0 else ""
                    st.markdown(f'<div class="match-card {accent_class}">', unsafe_allow_html=True)
                    st.markdown(f'<div class="job-title-card">{match["job_title"]}</div>', unsafe_allow_html=True)
                    
                    st.metric("Similarity", f"{match['similarity_percent']:.1f}%")
                    st.progress(match['similarity_percent'] / 100.0)
                    
                    st.write("**Preview:**")
                    st.write(match["combined_text"][:300] + "...")
                    
                    with st.expander("View Full Description"):
                        st.write(match["combined_text"])
                    st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.subheader("Similarity Comparison")
                
                titles = [f"{m['job_title']} ({m['id']})" for m in top_matches][::-1]
                scores = [m["similarity_percent"] for m in top_matches][::-1]
                
                colors = ['#3a506b'] * len(scores)
                if colors:
                    colors[-1] = '#00d2ff' # Top result highlighted
                    
                fig = go.Figure(go.Bar(
                    x=scores,
                    y=titles,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{s:.1f}%" for s in scores],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                    xaxis=dict(title='Similarity %', range=[0, 100], showgrid=False),
                    yaxis=dict(showgrid=False),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with tab3:
                st.subheader("Skill Gap Analysis")
                st.write("Comparing candidate technical skills with the highest matched job.")
                
                target_skills = [
                    "Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "TensorFlow",
                    "PyTorch", "Pandas", "NumPy", "Docker", "Kubernetes", "AWS", "Azure", "Git",
                    "REST API", "FastAPI", "Flask", "React", "Node.js", "Java", "C++", 
                    "Data Analysis", "Power BI", "Tableau", "Scikit-learn", "LangChain", "FAISS"
                ]
                
                resume_lower = resume_text.lower()
                top_job_lower = top_matches[0]["combined_text"].lower()
                
                found_in_resume = set([skill for skill in target_skills if skill.lower() in resume_lower])
                found_in_job = set([skill for skill in target_skills if skill.lower() in top_job_lower])
                
                matched_skills = found_in_resume.intersection(found_in_job)
                missing_skills = found_in_job - found_in_resume
                extra_skills = found_in_resume - found_in_job
                
                col_match, col_miss, col_extra = st.columns(3)
                
                with col_match:
                    st.markdown("#### ✅ Matched Skills")
                    if matched_skills:
                        badges = "".join([f'<span class="badge badge-matched">{s}</span>' for s in sorted(matched_skills)])
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.write("None")
                        
                with col_miss:
                    st.markdown("#### ❌ Missing Skills")
                    if missing_skills:
                        badges = "".join([f'<span class="badge badge-missing">{s}</span>' for s in sorted(missing_skills)])
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.write("None")
                        
                with col_extra:
                    st.markdown("#### 💡 Extra Skills")
                    if extra_skills:
                        badges = "".join([f'<span class="badge badge-extra">{s}</span>' for s in sorted(extra_skills)])
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.write("None")

            with tab4:
                st.subheader("AI Analyst Explanation")
                
                top_job = top_matches[0]
                prompt = f"""You are an AI career analyst.

Resume:
{resume_text}

Job Description:
{top_job['combined_text']}

Similarity Score:
{top_job['similarity_percent']:.1f}%

Provide:
- Why this job is a strong or a weak match.
- Key strengths alignment.
- Missing competencies.
- Actionable improvement suggestions.
"""
                
                with st.spinner("Generating AI explanation... (This uses local computation and may take up to a minute)"):
                    try:
                        process = subprocess.Popen(
                            ['ollama', 'run', 'phi3:mini'],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8'
                        )
                        stdout, stderr = process.communicate(input=prompt)
                        
                        if process.returncode != 0:
                            st.error(f"Error running Ollama: {stderr}")
                        else:
                            st.markdown(stdout.strip())
                    except Exception as e:
                        st.error(f"Failed to execute Ollama subprocess: {e}")

st.markdown("<div class='footer'>Powered by FAISS vector search, SentenceTransformers embeddings, and Phi3-mini local reasoning.</div>", unsafe_allow_html=True)
