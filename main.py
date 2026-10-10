import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import logging
import warnings
import asyncio
import nest_asyncio
import streamlit as st
from datetime import datetime

# =========================================================
# CONFIG & SUPPRESSION
# =========================================================
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

nest_asyncio.apply()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ShopAI | Autonomous Shopping Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load local image as Base64 data URI to display local images in Streamlit markdown
import base64

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].replace(".", "")
        mime = f"image/{ext}"
        if ext == "svg":
            mime = "image/svg+xml"
        return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"
    return ""

fire_logo_base64 = get_image_base64("assets/fire.svg")

# Custom CSS for a more modern look
# ================= UI Styling =================
st.markdown(
    """
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.main-header { font-size: 2.8rem; color: #00D4B1; text-align: center; font-weight: 800; margin-bottom: 0.2rem; letter-spacing: -0.5px; }
.sub-header { text-align: center; color: #888; margin-bottom: 2.5rem; font-size: 1.1rem; }
.stButton>button { width: 100%; background-color: #555555; color: #FAFAFA; border: none; padding: 0.8rem 1rem; border-radius: 8px; font-weight: 600; font-size: 1rem; margin-top: 1rem; }
.stButton>button:hover { background-color: #777777; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.stTextInput>div>div>input { background-color: #262730; color: #FAFAFA; border: 1px solid #393946; border-radius: 8px; padding: 0.8rem; }
.stTextInput>div>div>input:focus { border-color: #00D4B1; box-shadow: 0 0 0 2px rgba(0, 212, 177, 0.2); }
.css-1d391kg, .css-1d391kg>div { background-color: #0E1117 !important; border-right: 1px solid #262730; }
.css-1d391kg h1,h2,h3,h4,h5,h6,p,label { color: #FAFAFA !important; }
.stProgress > div > div > div > div { background-color: #00D4B1; }
.streamlit-expanderHeader { background-color: #262730; color: #FAFAFA; border-radius: 8px; font-weight: 600; }
.streamlit-expanderContent { background-color: #1A1D25; border-radius: 0 0 8px 8px; }
.card { background-color: #262730; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #00D4B1; }
.stRadio > div { background-color: #262730; padding: 1rem; border-radius: 8px; }
label { font-weight: 600 !important; margin-bottom: 0.5rem; display: block; color: #CCC !important; }
.main-title { font-size: 40px; font-weight: bold; display: flex; align-items: center; }
.main-title img { height: 50px; margin-left: 20px; vertical-align: middle; }
.subtitle { font-size: 20px; color: #AAAAAA; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header">
  <div class="main-title">
    RAGCart AI With
    <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*QR3Jl4jUu326U2p2.png" alt="ScrapeGraphAI Logo">
    <span style="margin-left:40px;">&</span>
    <img src="{fire_logo_base64}" alt="Firecrawl Logo">
  </div>
  <div class="subtitle"> Your intelligent shopping assistant</div>
  <br>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# IMPORTS (After config)
# =========================================================
from agent import AgenticShoppingAI

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.image("./assets/Groq.svg", width=150)
    
    # Model select dropdown to bypass Groq rate limits (Rate Limit 429 TPD)
    model_name = st.selectbox(
        "Select Groq LLM",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0,
        help="If you hit a Daily Token limit (Rate Limit 429), swap to Llama 3.1 8B or Mixtral for fresh quotas!"
    )
    
    groq_key = st.text_input(
        "Enter your Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    smartscrape_key = st.text_input(
        "Smartscrape Key", value=os.getenv("SCRAPEGRAPH_API_KEY", ""), type="password"
    )

    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state["GROQ_API_KEY"] = groq_key
        st.session_state["SCRAPEGRAPH_API_KEY"] = smartscrape_key
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["SCRAPEGRAPH_API_KEY"] = smartscrape_key
        
        # Force agent re-initialization with new credentials
        if 'agent' in st.session_state:
            del st.session_state['agent']
            
        st.success("Credentials updated! Re-initializing agent...")
        st.rerun()
        
    if groq_key or smartscrape_key:
        st.caption("Credentials loaded for active session.")

    st.markdown("---")

# =========================================================
# AGENT DYNAMIC INITIALIZATION
# =========================================================
if 'agent' not in st.session_state or st.session_state.get('active_model') != model_name:
    with st.spinner(f"🚀 Initializing Agent with {model_name}..."):
        st.session_state.agent = AgenticShoppingAI(model_name=model_name)
        st.session_state['active_model'] = model_name

# =========================================================
# SIDEBAR RAG DATABASE INSPECTOR
# =========================================================
with st.sidebar:
    st.subheader("📦 RAG Database Inspector")
    
    # Get total document count programmatically
    try:
        doc_count = st.session_state.agent.vector_db._collection.count()
        st.metric(label="Total Stored RAG Chunks", value=doc_count)
        
        # Reset button to purge database and test clean citations
        if doc_count > 0:
            if st.button("🗑️ Clear Vector Database", use_container_width=True):
                try:
                    ids = st.session_state.agent.vector_db.get()['ids']
                    if ids:
                        st.session_state.agent.vector_db.delete(ids=ids)
                    st.success("Database cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear database: {e}")
    except Exception as e:
        st.write("No database initialized yet.")
        doc_count = 0
        
    # Quick database query test tool
    st.write("🔍 **Test Vector DB Retrieval**")
    test_query = st.text_input("Query RAG directly:", placeholder="e.g. laptop", key="rag_test_input")
    if test_query:
        try:
            retriever = st.session_state.agent.vector_db.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(test_query)
            if retrieved_docs:
                st.success(f"Found {len(retrieved_docs)} matching chunks!")
                for idx, doc in enumerate(retrieved_docs):
                    source_url = doc.metadata.get("source", "")
                    with st.expander(f"Chunk #{idx+1} (Preview)"):
                        if source_url:
                            st.markdown(f"🌐 **Source Link**: [Visit Site]({source_url})")
                        st.text(doc.page_content[:200] + "...")
            else:
                st.warning("No matching chunks found for this query.")
        except Exception as e:
            st.error(f"Error querying RAG: {e}")

# =========================================================
# MAIN UI
# =========================================================
query = st.chat_input(
    "What are you looking for today?",
)
if query:
    if not query:
        st.error("Please enter a product to search.")
    else:
        try:
            # Use st.status for a modern progress indicator
            with st.status("gent at work...", expanded=True) as status:
                st.write("Searching the web...")
                result = asyncio.run(st.session_state.agent.run(query))
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            st.markdown(result)
            
            # Add a download button for the report
            st.download_button(
                label="📥 Download Report",
                data=result,
                file_name=f"shopai_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

