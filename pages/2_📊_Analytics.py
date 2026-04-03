import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroSort AI | Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inherit Exact Same Advanced CSS
st.markdown("""
<style>
    /* Premium Reset */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}
    .block-container {
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1350px;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background-color: #fcfdfe; /* New surface */
    }

    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Manrope:wght@800&display=swap');
    
    .title-h1 {
        font-family: 'Manrope', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #0d1117;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #57606a;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Cards & Containers */
    .chart-container {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 2rem;
        border: 1px solid #d0d7de;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
    }
    
    .insight-card {
        background: #f6f8fa;
        border-radius: 0.75rem;
        padding: 1rem;
        border-left: 4px solid #0969da;
        margin-bottom: 0.75rem;
    }
    
    /* Architecture Diagram Mockup */
    .arch-step {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 0.5rem;
        padding: 10px;
        text-align: center;
        font-size: 0.75rem;
        font-weight: bold;
        width: 100px;
    }
</style>
""", unsafe_allow_html=True)

# 3. State Requirements
if 'history' not in st.session_state:
    st.session_state.history = []

def generate_ai_insights(df):
    if df.empty: return []
    insights = []
    
    # Majority document type
    major_type = df['type'].value_counts().idxmax()
    percent = (df['type'].value_counts().max() / len(df)) * 100
    insights.append(f"<b>Dominant Classification:</b> {major_type} ({percent:.0f}% of total).")
    
    # average confidence
    avg_conf = df['confidence'].mean()
    insights.append(f"<b>Engine Reliability:</b> {avg_conf:.1f}% mean certainty across current batch (V6.0 Logic).")
    
    # variance note
    if len(df['type'].unique()) > 1:
        insights.append(f"<b>Batch Diversity:</b> Multiple distinct document archetypes detected in recent logs.")
    
    return insights

def main():
    col_nav, col_main = st.columns([1.5, 8.5], gap="large")
    
    with col_nav:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Manrope;font-weight:800;font-size:1.3rem;color:#0d1117;'>NeuroSort AI</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.6rem;font-weight:800;color:#57606a;letter-spacing:0.1em;text-transform:uppercase;'>Robust Refined V8.0</div>", unsafe_allow_html=True)
        
        if st.button("🏠 Global Dash", key="n_back", use_container_width=True):
            st.switch_page("app.py")
        st.button("📊 Advanced Analytics", key="analytics_active", use_container_width=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Architecture Modal Trigger
        @st.dialog("🧠 System Architecture V6.0")
        def show_architecture():
            st.markdown("""
            <div style="background:#f6f8fa; padding:2rem; border-radius:1rem; border:1px solid #d0d7de; text-align:center;">
                <div style="display:flex; flex-direction:column; align-items:center; gap:15px;">
                    <div style="background:white; border:2px solid #0969da; border-radius:0.75rem; padding:15px; width:100%; font-weight:800;">👤 User Batch Upload</div>
                    <div style="color:#afb8c1; font-size:1.5rem;">↓</div>
                    <div style="background:white; border:1px solid #d0d7de; border-radius:0.75rem; padding:12px; width:90%; font-size:0.85rem;"><b>Regex Cleaning:</b> Browser/IP/Page Noise Removal</div>
                    <div style="color:#afb8c1; font-size:1.5rem;">↓</div>
                    <div style="background:white; border:1px solid #d0d7de; border-radius:0.75rem; padding:12px; width:90%; font-size:0.85rem;"><b>Hybrid Core:</b> BART-Large + Detection Heuristics</div>
                    <div style="color:#afb8c1; font-size:1.5rem;">↓</div>
                    <div style="background:white; border:1px solid #d0d7de; border-radius:0.75rem; padding:12px; width:90%; font-size:0.85rem;"><b>Max Confidence:</b> (Score > 0.8 Heuristic Override)</div>
                    <div style="color:#afb8c1; font-size:1.5rem;">↓</div>
                    <div style="background:white; border:2px solid #1a7f37; border-radius:0.75rem; padding:15px; width:100%; font-weight:800;">🎯 Specific Document Classification</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("V6.0 Refined features strictly cleaner text and optimized heuristic signals.")

        if st.button("🌐 View System Architecture", use_container_width=True):
            show_architecture()
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.65rem; color:#8c959f; font-weight:800; text-align:center;'>© 2026 NEUROSORT AI <br> PERFORMANCE EDITION v8</div>", unsafe_allow_html=True)

    with col_main:
        st.markdown("<div class='title-h1'>Analytics & Library Insights</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Production-grade intelligence on refined V8.0 taxonomy and engine performance metrics.</div>", unsafe_allow_html=True)
        
        # Point: Use st.session_state.results (v6 key)
        if not st.session_state.get('results'):
            st.warning("Neural database is currently empty. Initialize a classification pipeline to view insights.")
            return
            
        df = pd.DataFrame(st.session_state.results)
        
        # Performance Charts
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.markdown("#### Document Type Distribution")
            type_counts = df['type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig_pie = px.pie(type_counts, names='Type', values='Count', hole=0.4,
                             color_discrete_sequence=['#0969da', '#1a7f37', '#8250df', '#bf8700', '#cf222e'])
            fig_pie.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=250, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.markdown("#### Neural Reliability Index")
            # Dynamic representation based on current scores
            avg_c = df['confidence'].mean()
            perf_df = pd.DataFrame({
                "Metric": ["Target", "Current Avg", "Peak", "V6.0 Stability"],
                "Value": [90, avg_c, max(df['confidence']), 95]
            })
            fig_bar = px.bar(perf_df, x='Metric', y='Value', color='Metric', text_auto='.0f',
                            color_discrete_sequence=['#afb8c1', '#1a7f37', '#0969da', '#8250df'])
            fig_bar.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250, showlegend=False, yaxis_range=[0,100])
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # AI INSIGHTS SECTION
        st.markdown("#### 🧠 AI Insights & Heuristics")
        insights = generate_ai_insights(df)
        ic1, ic2 = st.columns(2)
        for i, text in enumerate(insights):
            col = ic1 if i % 2 == 0 else ic2
            with col:
                st.markdown(f"<div class='insight-card'>{text}</div>", unsafe_allow_html=True)
        
        # Global Library
        st.markdown("<br><div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("#### Refined Global Library")
        st.markdown("<div style='font-size:0.8rem; color:#57606a; margin-bottom:1rem;'>Search and filter high-confidence predictions validated by the V6.0 Hybrid Core.</div>", unsafe_allow_html=True)
        
        st.dataframe(
            df[['file', 'type', 'confidence', 'timestamp']].sort_values('timestamp', ascending=False),
            column_config={
                "file": "Document Name",
                "type": "Specific Classification",
                "confidence": st.column_config.ProgressColumn("Confidence Index (%)", min_value=0, max_value=100, format="%d%%"),
                "timestamp": "Batch Time"
            },
            use_container_width=True,
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
