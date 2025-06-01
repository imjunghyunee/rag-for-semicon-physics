from __future__ import annotations
import json
import requests
from typing import List
from rag_pipeline.graph_state import GraphState
from rag_pipeline import retrievers, config, utils


def node_retrieve_file_embedding(state: GraphState, pdf_path: str) -> GraphState:
    query = state["question"][-1]
    context = retrievers.retrieve_from_file_embedding(query, pdf_path)
    return {"context": context}


def node_retrieve_img_embedding(state: GraphState, img_path: str) -> GraphState:
    query = state["question"][-1]
    context = retrievers.retrieve_from_img_embedding(query, img_path)
    return {"context": context}


def node_retrieve(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context = retrievers.vectordb_retrieve(query)
    return {"context": context}


def node_retrieve_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context = retrievers.vectordb_hybrid_retrieve(query, weights=hybrid_weights)
    return {"context": context}


def node_retrieve_summary(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context, explanation = retrievers.summary_retrieve(query)
    return {"context": context, "explanation": explanation}


def node_retrieve_summary_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context, explanation = retrievers.summary_hybrid_retrieve(
        query, weights=hybrid_weights
    )
    return {"context": context, "explanation": explanation}


def node_retrieve_hyde(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context, explanation = retrievers.hyde_retrieve(query)
    return {"context": context, "explanation": explanation}


def node_retrieve_hyde_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context, explanation = retrievers.hyde_hybrid_retrieve(
        query, weights=hybrid_weights
    )
    return {"context": context, "explanation": explanation}


def node_relevance_check(state: GraphState) -> GraphState:
    with open("score/path.json", "r", encoding="utf-8") as f:
        scores: List[float] = json.load(f)
        print(scores)

    context_docs = state["context"]  # List[Document]
    contents = [d.page_content for d in context_docs]

    filtered_scores: List[float] = []
    filtered_context: List[str] = []

    for i, score in enumerate(scores):
        if score >= config.SIM_THRESHOLD:
            filtered_scores.append(score)
            filtered_context.append(contents[i])

    return {
        "filtered_context": filtered_context,
        "scores": scores,
        "filtered_scores": filtered_scores,
    }


def node_llm_answer(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    context_docs = state["context"]
    contents = [d.page_content for d in context_docs]
    context_str = "\n\n---\n\n".join(contents)

    payload = utils.build_payload_for_llm_answer(query, context_str)

    res = requests.post(config.REMOTE_LLM_URL, json=payload).json()
    answer = res["choices"][0]["message"]["content"].strip()
    if not isinstance(answer, str):
        answer = str(answer)

    return {"answer": answer, "messages": [("assistant", answer)]}


def node_simple_or_not(state: GraphState) -> dict:
    """Determine if the question is simple or requires complex multi-hop reasoning."""
    query: str = state["question"][-1]

    payload = utils.build_payload_for_complexity_check(query)
    response = requests.post(config.REMOTE_LLM_URL, json=payload).json()
    decision = response["choices"][0]["message"]["content"].strip().lower()

    # Ensure response is valid
    if decision not in ["simple", "complex"]:
        print(f"Invalid complexity decision: '{decision}', defaulting to 'simple'")
        decision = "simple"

    print(f"Question complexity determined as: {decision}")
    # Return as a dictionary with a routing key
    return {"next": decision}
