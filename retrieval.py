from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import streamlit as st


@st.cache_resource
def get_cross_encoder():
  return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def run_guardrail(query):
  """Guardrail to ensure the query is valid and non-empty."""
  if not query or len(query.strip()) < 3:
    return False, "Guardrail Triggered: Query is too short or invalid."
  return True, "Passed"


def run_hybrid_retrieval(query, vector_db, neo4j_driver, openai_api_key):
  # 1. Apply Guardrail
  is_safe, msg = run_guardrail(query)
  if not is_safe:
    return msg, [], {}

  # 2. Hybrid Retrieval (FAISS vector search + Neo4j Graph traversal)
  vector_results = vector_db.similarity_search(query, k=4)
  vector_texts = [doc.page_content for doc in vector_results]

  graph_texts = []
  if neo4j_driver:
    with neo4j_driver.session() as session:
      result = session.run(
          "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c.text AS text, e.name AS entity LIMIT 5"
      )
      graph_texts = [
          f"{record['text']} (Entity: {record['entity']})" for record in result
      ]

  combined_candidates = list(set(vector_texts + graph_texts))
  if not combined_candidates:
    return "No relevant context found across vector and graph stores.", [], {}

  # 3. Cross-Encoder Reranking
  cross_encoder = get_cross_encoder()
  pairs = [[query, candidate] for candidate in combined_candidates]
  scores = cross_encoder.predict(pairs)

  scored_candidates = sorted(
      zip(combined_candidates, scores), key=lambda x: x[1], reverse=True
  )
  top_chunks = [item[0] for item in scored_candidates[:3]]
  final_context = "\n\n---\n\n".join(top_chunks)

  # 4. LLM Generation
  llm = ChatOpenAI(
      model="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key
  )
  prompt_template = PromptTemplate(
      input_variables=["context", "question"],
      template=(
          "You are an expert technical assistant. Answer accurately using ONLY"
          " the provided context.\n\nContext:\n{context}\n\nQuestion:"
          " {question}\n\nAnswer:"
      ),
  )

  chain = prompt_template | llm
  response = chain.invoke({"context": final_context, "question": query})
  answer_text = response.content

  # 5. DeepEval Evaluation Integration
  eval_scores = {}
  try:
    test_case = LLMTestCase(
        input=query, actual_output=answer_text, retrieval_context=top_chunks
    )

    # Evaluate Faithfulness
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5, model="gpt-4o-mini", include_reason=False
    )
    faithfulness_metric.measure(test_case)
    eval_scores["Faithfulness Score"] = faithfulness_metric.score

    # Evaluate Answer Relevancy (Fixed class name)
    relevance_metric = AnswerRelevancyMetric(
        threshold=0.5, model="gpt-4o-mini", include_reason=False
    )
    relevance_metric.measure(test_case)
    eval_scores["Answer Relevancy Score"] = relevance_metric.score
  except Exception as e:
    eval_scores["Evaluation Error"] = str(e)

  return answer_text, top_chunks, eval_scores