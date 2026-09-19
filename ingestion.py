import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase


def run_ingestion(
    file_path, filename, openai_api_key, neo4j_uri, neo4j_user, neo4j_password
):
  # Step 1: PDF Extraction
  loader = PyPDFLoader(file_path)
  docs = loader.load()

  # Step 2: Section / Chunking
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  splits = text_splitter.split_documents(docs)

  # Step 3A: Vector Pipeline (Embeddings -> FAISS)
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
  vector_db = FAISS.from_documents(splits, embeddings)

  # Step 3B: Knowledge Graph Pipeline (Entity Extraction -> Relationships -> Neo4j)
  driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
  llm = ChatOpenAI(
      model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key
  )

  with driver.session() as session:
    # Create Document Node
    session.run("MERGE (d:Document {name: $filename})", filename=filename)

    # Process chunks for Entity Extraction & Relationships
    for i, split in enumerate(
        splits[:15]
    ):  # Limited for performance during demo runtime
      chunk_text = split.page_content

      # Simple LLM Entity & Relationship Extractor
      extraction_prompt = f"""
            Extract up to 2 key entities and their relationship from the text below. 
            Return ONLY a valid JSON object with keys: "entity1", "relation", "entity2".
            Text: {chunk_text}
            """
      try:
        res = llm.invoke(extraction_prompt).content
        clean_res = res.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_res)
        e1 = data.get("entity1", "Concept")
        rel = data.get("relation", "RELATED_TO").upper().replace(" ", "_")
        e2 = data.get("entity2", "Topic")
      except Exception:
        e1, rel, e2 = "DocumentChunk", "CONTAINS", f"Chunk_{i}"

      # Populate Neo4j Knowledge Graph with Entities and Relationships
      session.run(
          """
                MATCH (d:Document {name: $filename})
                CREATE (c:Chunk {id: $id, text: $text})
                MERGE (e1:Entity {name: $e1})
                MERGE (e2:Entity {name: $e2})
                MERGE (d)-[:CONTAINS]->(c)
                MERGE (c)-[:MENTIONS]->(e1)
                MERGE (e1)-[r:%s]->(e2)
            """
          % rel,
          filename=filename,
          id=i,
          text=chunk_text,
          e1=e1,
          e2=e2,
      )

  return vector_db, driver