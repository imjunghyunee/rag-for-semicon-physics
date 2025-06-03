from __future__ import annotations
import json
from typing import List
from rag_pipeline.graph_state import GraphState
from rag_pipeline import retrievers, config, utils, plan_execute_langgraph
from pathlib import Path


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
    """관련성 체크 - 파일 경로 문제 수정"""
    try:
        # 올바른 파일 경로 사용
        score_file_path = config.SCORE_PATH

        if not Path(score_file_path).exists():
            print(
                f"Warning: Score file not found at {score_file_path}, skipping relevance check"
            )
            return {
                "filtered_context": state.get("context", []),
                "scores": [],
                "filtered_scores": [],
            }

        with open(score_file_path, "r", encoding="utf-8") as f:
            scores: List[float] = json.load(f)
            print(f"Loaded {len(scores)} similarity scores")

        context_docs = state["context"]  # List[Document]
        if not context_docs:
            return {
                "filtered_context": [],
                "scores": scores,
                "filtered_scores": [],
            }

        contents = [d.page_content for d in context_docs]

        filtered_scores: List[float] = []
        filtered_context: List[str] = []

        for i, score in enumerate(scores):
            if i < len(contents) and score >= config.SIM_THRESHOLD:
                filtered_scores.append(score)
                filtered_context.append(contents[i])

        print(
            f"Filtered {len(filtered_context)} documents above threshold {config.SIM_THRESHOLD}"
        )

        return {
            "filtered_context": filtered_context,
            "scores": scores,
            "filtered_scores": filtered_scores,
        }

    except Exception as e:
        print(f"Error in relevance check: {e}")
        # 에러 발생 시 원본 컨텍스트 반환
        return {
            "filtered_context": state.get("context", []),
            "scores": [],
            "filtered_scores": [],
        }


def node_llm_answer(state: GraphState) -> GraphState:
    """LLM 답변 생성 - 에러 처리 강화"""
    try:
        query: str = state["question"][-1]

        context_docs = state.get("context", [])
        if not context_docs:
            return {
                "answer": f"No context available to answer the question: {query}",
                "messages": [("assistant", "No context available")],
            }

        contents = [d.page_content for d in context_docs]
        context_str = "\n\n---\n\n".join(contents)

        if not context_str.strip():
            return {
                "answer": f"Empty context provided for question: {query}",
                "messages": [("assistant", "Empty context")],
            }

        answer = utils.generate_llm_answer(query, context_str)
        if not isinstance(answer, str):
            answer = str(answer)

        return {"answer": answer, "messages": [("assistant", answer)]}

    except Exception as e:
        print(f"Error in LLM answer generation: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "messages": [("assistant", f"Error: {str(e)}")],
        }


def node_simple_or_not(state: GraphState) -> dict:
    """Determine if the question is simple or requires complex multi-hop reasoning."""
    query: str = state["question"][-1]

    decision = utils.check_query_complexity(query).strip().lower()

    # Ensure response is valid
    if decision not in ["simple", "complex"]:
        print(f"Invalid complexity decision: '{decision}', defaulting to 'simple'")
        decision = "simple"

    print(f"Question complexity determined as: {decision}")
    # Return as a dictionary with a routing key
    return {"next": decision}


def node_plan_and_execute(state: GraphState) -> GraphState:
    """LangGraph 기반 Plan and Execute agent를 사용하여 복잡한 질문을 처리합니다."""
    try:
        query: str = state["question"][-1]

        hybrid_weight_embedding = config.HYBRID_WEIGHT
        hybrid_weight_bm25 = 1 - hybrid_weight_embedding
        hybrid_weights = [hybrid_weight_embedding, hybrid_weight_bm25]
        retrieval_type = config.RETRIEVAL_TYPE

        print(f"Starting LangGraph Plan and Execute for: {query}")
        print(
            f"Using retrieval_type: {retrieval_type}, hybrid_weights: {hybrid_weights}"
        )

        # LangGraph 기반 Plan and Execute 방식으로 복잡한 질문 처리
        result = plan_execute_langgraph.process_complex_query_with_langgraph_plan_execute(
            original_query=query,
            retrieval_type=retrieval_type,
            hybrid_weights=hybrid_weights,
            max_steps=8,  # 🔥 recursion_limit와 맞춤
        )

        # 🔥 모든 정보를 완전히 처리하여 반환
        return {
            "plan": result["plan"],
            "executed_steps": result["executed_steps"],
            "context": result["all_context_docs"],
            "combined_context": result["combined_context"],
            "explanation": f"LangGraph Plan and Execute completed with {result['total_steps']} steps",
            "answer": result["final_answer"],  # 🔥 최종 답변 포함
            "messages": [("assistant", result["final_answer"])],  # 🔥 메시지도 추가
        }

    except Exception as e:
        print(f"Error in LangGraph Plan and Execute: {e}")
        # 🔥 에러 시에도 폴백 답변 제공
        error_answer = f"Error in LangGraph Plan and Execute: {str(e)}"
        return {
            "plan": [],
            "executed_steps": [],
            "context": [],
            "combined_context": "",
            "explanation": error_answer,
            "answer": error_answer,
            "messages": [("assistant", error_answer)],
        }