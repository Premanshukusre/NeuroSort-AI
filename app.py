"""
NeuroSort AI: Professional Intelligent Document Classifier.
Enterprise-grade dashboard built for academic evaluation and production-level AI processing.
"""

import streamlit as st
import time
import os
import datetime
import pandas as pd
import random
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroSort AI | Intelligent Sorting",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" # Changed to expanded for visibility
)

# 2. NeuroSort Professional Styling
st.markdown("""
<style>
    /* Premium Reset */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}
    .block-container {
        padding-top: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1350px;
    }
    
    /* Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Manrope:wght@800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background-color: #f7f9fc; /* Clean professional surface */
    }

    /* Branding & Header */
    .brand-title {
        font-family: 'Manrope', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #0d1117;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .brand-sub {
        color: #57606a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Stepper UI for Pipeline */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #d0d7de;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    .step {
        flex: 1;
        text-align: center;
        position: relative;
    }
    .step:not(:last-child)::after {
        content: '→';
        position: absolute;
        right: -10%;
        top: 20%;
        color: #afb8c1;
        font-size: 1.2rem;
    }
    .step-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .step-label {
        font-size: 0.7rem;
        font-weight: 800;
        color: #0969da;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .step-desc {
        font-size: 0.65rem;
        color: #57606a;
        padding: 0 0.5rem;
    }

    /* Metric Cards */
    .neuro-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid #d0d7de;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .neuro-label {
        font-size: 0.7rem;
        font-weight: 800;
        color: #0969da;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .neuro-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f2328;
    }
    
    .stButton>button {
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
    }

</style>
""", unsafe_allow_html=True)

# 3. Enhanced State Management
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processed_ids' not in st.session_state:
    st.session_state.processed_ids = set()
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'avg_confidence' not in st.session_state:
    st.session_state.avg_confidence = 0.0

# Official V6.0 Labels
CANDIDATE_LABELS = [
    "Internship Offer Letter",
    "Medical Report",
    "Academic Assignment",
    "Financial Document",
    "Technical Documentation"
]

@st.cache_resource
def load_neuro_engine():
    """Loads BART-Large Semantic Engine."""
    with st.spinner("Initializing NeuroSort AI Core..."):
        return pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def render_pipeline_flow():
    st.markdown("""
    <div class="pipeline-container">
        <div class="step">
            <div class="step-icon">📄</div>
            <div class="step-label">Text Extraction</div>
            <div class="step-desc">High-Fidelity PDF/Doc Analysis</div>
        </div>
        <div class="step">
            <div class="step-icon">🧹</div>
            <div class="step-label">Regex Clearing</div>
            <div class="step-desc">Metadata & Noise Removal</div>
        </div>
        <div class="step">
            <div class="step-icon">🤖</div>
            <div class="step-label">Hybrid Engine</div>
            <div class="step-desc">Model + Heuristic Fusion</div>
        </div>
        <div class="step">
            <div class="step-icon">🎯</div>
            <div class="step-label">Final Label</div>
            <div class="step-desc">Rounded High-Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    col_nav, col_main = st.columns([1.8, 8.2], gap="large")
    
    with col_nav:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Manrope;font-weight:800;font-size:1.5rem;color:#0d1117;'>NeuroSort AI</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.65rem;font-weight:800;color:#57606a;letter-spacing:0.1em;text-transform:uppercase;'>V6.0 Refined Logic</div>", unsafe_allow_html=True)
        
        st.button("🏠 Home Dashboard", use_container_width=True)
        if st.button("📊 Analytics Insights", use_container_width=True):
            st.switch_page("pages/2_📊_Analytics.py")
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.session_state.results:
            st.markdown("<div style='font-size:0.7rem; font-weight:800; color:#57606a; margin-bottom:8px;'>DATA EXPORT</div>", unsafe_allow_html=True)
            df_export = pd.DataFrame(st.session_state.results)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=df_export.to_csv(index=False),
                file_name="NeuroSort_Results.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.65rem; color:#8c959f; font-weight:800; text-align:center;'>© 2026 NEUROSORT AI <br> DEPLOYMENT STABLE V7</div>", unsafe_allow_html=True)

    with col_main:
        st.markdown("<div class='brand-title'>NeuroSort Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-sub'>Professional Document Classification with Hybrid ML + Heuristic Intelligence.</div>", unsafe_allow_html=True)
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='neuro-card'><div class='neuro-label'>Engine Status</div><div class='neuro-value' style='color:#1a7f37;'>ACTIVE</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='neuro-card'><div class='neuro-label'>Avg Certainty</div><div class='neuro-value'>{st.session_state.avg_confidence:.1f}%</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='neuro-card'><div class='neuro-label'>Total Analysed</div><div class='neuro-value'>{st.session_state.total_processed}</div></div>", unsafe_allow_html=True)
        
        # Visual Pipeline
        render_pipeline_flow()
        
        from src.document_parser import parse_document, clean_text
        from src.heuristics import detect_document_type
        
        st.markdown("##### Ingestion Terminal")
        uploaded_files = st.file_uploader(
            "Upload Records (PDF, DOCX, TXT)", 
            type=['pdf', 'docx', 'txt'], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_ids]
            
            if new_files:
                classifier = load_neuro_engine()
                progress_bar = st.progress(0, text="Initializing Pipeline...")
                
                for i, file in enumerate(new_files):
                    progress_text = f"Analysing {file.name} (Batch Item {i+1}/{len(new_files)})..."
                    progress_bar.progress((i + 1) / len(new_files), text=progress_text)
                    
                    temp_path = f"tmp_{file.name}"
                    with open(temp_path, "wb") as f: f.write(file.getbuffer())
                    raw_text = parse_document(temp_path)
                    os.remove(temp_path)
                    
                    if raw_text:
                        text = clean_text(raw_text)
                        heuristic_label, heuristic_score = detect_document_type(text)
                        
                        # Decide if ML fallback is needed (Point 2: Heuristic-First)
                        if heuristic_score > 0.8:
                            # HIGH CONFIDENCE HEURISTIC: Skip heavy transformers
                            final_label = heuristic_label
                            confidence = round(heuristic_score * 100, 2)
                        else:
                            # LOW CONFIDENCE HEURISTIC: Fallback to Transformers
                            # Optimized snippet size for faster inference
                            snippet = text[:400] 
                            model_pred = classifier(snippet, CANDIDATE_LABELS)
                            model_label = model_pred['labels'][0]
                            model_confidence = model_pred['scores'][0]
                            
                            # Final decision merger
                            if model_confidence > heuristic_score:
                                final_label = model_label
                                confidence = round(model_confidence * 100, 2)
                            else:
                                final_label = heuristic_label
                                confidence = round(heuristic_score * 100, 2)
                        
                        st.session_state.results.append({
                            "file": file.name,
                            "type": final_label,
                            "confidence": confidence,
                            "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                        })
                        st.session_state.processed_ids.add(file.name)
                        st.session_state.total_processed += 1
                
                if st.session_state.results:
                    st.session_state.avg_confidence = sum([r['confidence'] for r in st.session_state.results]) / len(st.session_state.results)
                
                progress_bar.empty()
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Real-time Analysis Feed")
        
        if not st.session_state.results:
            st.info("System idle. Waiting for document ingestion.")
        else:
            for item in reversed(st.session_state.results):
                conf = item['confidence']
                conf_color = "#1a7f37" if conf > 70 else ("#bf8700" if conf > 50 else "#cf222e")
                
                st.markdown(f"""
                <div style="background:white; border:1px solid #d0d7de; border-radius:1rem; padding:20px; margin-bottom:15px; box-shadow:0 2px 5px rgba(0,0,0,0.02);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:0.7rem; font-weight:800; color:#57606a; text-transform:uppercase;">Source File</div>
                            <div style="font-size:1.1rem; font-weight:700; color:#0d1117;">{item['file']}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.7rem; font-weight:800; color:#57606a; text-transform:uppercase;">AI Confidence</div>
                            <div style="font-size:1.5rem; font-weight:800; color:{conf_color};">{conf}%</div>
                        </div>
                    </div>
                    <div style="margin-top:15px; padding-top:15px; border-top:1px solid #f6f8fa; display:flex; gap:30px;">
                        <div>
                            <div style="font-size:0.7rem; font-weight:800; color:#0969da; text-transform:uppercase;">Classification</div>
                            <div style="font-size:1rem; font-weight:600;">{item['type']}</div>
                        </div>
                        <div>
                            <div style="font-size:0.7rem; font-weight:800; color:#57606a; text-transform:uppercase;">Timestamp</div>
                            <div style="font-size:1rem; color:#57606a;">{item['timestamp']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
