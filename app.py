import os
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OPENAI_API_KEY
from ingestion import run_ingestion
from retrieval import run_hybrid_retrieval
import streamlit as st

st.set_page_config(
    page_title="Hybrid RAG: Vector + Graph + Guardrails + DeepEval",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ingestion_complete" not in st.session_state:
  st.session_state.ingestion_complete = False
if "vector_db" not in st.session_state:
  st.session_state.vector_db = None
if "neo4j_driver" not in st.session_state:
  st.session_state.neo4j_driver = None

st.title(
    "⚡ Hybrid RAG Application (FAISS + Neo4j + Guardrails + DeepEval)"
)
st.write(
    "Runtime PDF Upload ➔ Ingestion Pipeline (Vector DB + Knowledge Graph) ➔"
    " Guardrails ➔ Hybrid Retrieval ➔ Cross-Encoder Reranking ➔ DeepEval."
)

# --- SIDEBAR: Ingestion Module ---
st.sidebar.header("📁 Module 1: Ingestion Pipeline")

if not OPENAI_API_KEY or not NEO4J_PASSWORD:
  st.sidebar.error("⚠️ Missing credentials in `.env` file.")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF document", type=["pdf"]
)

if uploaded_file is not None and OPENAI_API_KEY and NEO4J_PASSWORD:
  os.makedirs("data", exist_ok=True)
  file_path = os.path.join("data", uploaded_file.name)
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.sidebar.success(f"File uploaded: {uploaded_file.name}")

  if st.sidebar.button("🚀 Run Ingestion Pipeline"):
    with st.spinner(
        "Extracting PDF ➔ Chunking ➔ Building FAISS & Neo4j Knowledge Graph..."
    ):
      try:
        vector_db, driver = run_ingestion(
            file_path=file_path,
            filename=uploaded_file.name,
            openai_api_key=OPENAI_API_KEY,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
        )
        st.session_state.vector_db = vector_db
        st.session_state.neo4j_driver = driver
        st.session_state.ingestion_complete = True
        st.sidebar.success(
            "Ingestion Complete! Vector & Knowledge Graph Ready."
        )
      except Exception as e:
        st.sidebar.error(f"Ingestion failed: {e}")

# --- MAIN PAGE: Retrieval & Evaluation Module ---
st.header("🔍 Module 2: Guardrails, Hybrid Retrieval & DeepEval")

if not st.session_state.ingestion_complete:
  st.warning(
      "🔒 Retrieval section is locked. Please upload a PDF and run the"
      " Ingestion Pipeline from the sidebar."
  )
else:
  st.success("✅ Ingestion verified. You can query your document.")

  query = st.text_input(
      "Enter your question based on the document:",
      placeholder="e.g., What are the core components?",
  )

  if st.button("Generate Hybrid Answer") and query:
    with st.spinner(
        "Running Guardrails ➔ Hybrid Retrieval ➔ Reranking ➔ DeepEval..."
    ):
      try:
        answer, top_chunks, eval_scores = run_hybrid_retrieval(
            query=query,
            vector_db=st.session_state.vector_db,
            neo4j_driver=st.session_state.neo4j_driver,
            openai_api_key=OPENAI_API_KEY,
        )

        st.markdown("### 📝 Combined Single Output")
        st.write(answer)

        # Display DeepEval Evaluation Metrics
        if eval_scores:
          st.markdown("### 📊 DeepEval Metrics Dashboard")
          cols = st.columns(len(eval_scores))
          for col, (metric_name, score) in zip(cols, eval_scores.items()):
            col.metric(label=metric_name, value=str(score))

        with st.expander("View Retrieved & Reranked Context Chunks"):
          for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1}**: {chunk}")

      except Exception as e:
        st.error(f"An error occurred: {e}")