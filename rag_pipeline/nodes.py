from __future__ import annotations
import json
from typing import List
from rag_pipeline.graph_state import GraphState
from rag_pipeline import retrievers, config, utils, query_decomposition
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


# def node_llm_answer_parent


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


def node_query_decomposition(state: GraphState) -> GraphState:
    """복잡한 질문을 하위 질문들로 분해하고 각각 처리합니다."""
    try:
        query: str = state["question"][-1]

        hybrid_weight_embedding = config.HYBRID_WEIGHT
        hybrid_weight_bm25 = 1 - hybrid_weight_embedding
        hybrid_weights = [hybrid_weight_embedding, hybrid_weight_bm25]
        retrieval_type = config.RETRIEVAL_TYPE

        print(f"Starting query decomposition for: {query}")
        print(
            f"Using retrieval_type: {retrieval_type}, hybrid_weights: {hybrid_weights}"
        )

        # 복잡한 질문 처리
        decomposition_result = query_decomposition.process_complex_query(
            original_query=query,
            retrieval_type=retrieval_type,
            hybrid_weights=hybrid_weights,
            max_subquestions=5,
        )

        return {
            "subquestions": decomposition_result["subquestions"],
            "subquestion_results": decomposition_result["subquestion_results"],
            "context": decomposition_result["all_context_docs"],
            "combined_context": decomposition_result["combined_context"],
            "explanation": f"Query decomposed into {len(decomposition_result['subquestions'])} sub-questions",
        }

    except Exception as e:
        print(f"Error in query decomposition: {e}")
        return {
            "subquestions": [state["question"][-1]],
            "subquestion_results": [],
            "context": [],
            "combined_context": "",
            "explanation": f"Error in decomposition: {str(e)}",
        }


def node_complex_llm_answer(state: GraphState) -> GraphState:
    """복잡한 질문에 대한 최종 답변을 생성합니다."""
    try:
        query: str = state["question"][-1]
        subquestion_results = state.get("subquestion_results", [])

        if not subquestion_results:
            return {
                "answer": f"No sub-question results available for: {query}",
                "messages": [("assistant", "No sub-question results")],
            }

        # 모든 하위 질문의 결과를 종합하여 최종 답변 생성
        final_answer = query_decomposition.aggregate_subquestion_results(
            original_query=query, subquestion_results=subquestion_results
        )

        return {"answer": final_answer, "messages": [("assistant", final_answer)]}

    except Exception as e:
        print(f"Error in complex LLM answer generation: {e}")
        return {
            "answer": f"Error generating complex answer: {str(e)}",
            "messages": [("assistant", f"Error: {str(e)}")],
        }


def node_relevance_check_parent(state: GraphState) -> GraphState:
    """Parent retrieval relevance check with multiple similarity scores"""
    print("🔍 Starting parent relevance check...")

    try:
        output_dir = Path(config.OUTPUT_DIR)

        # Load similarity scores
        content_query_score_path = output_dir / "content_query_similarity_score.json"
        summary_query_score_path = output_dir / "summary_query_similarity_score.json"
        content_expanded_query_score_path = (
            output_dir / "content_expanded_query_similarity_score.json"
        )

        # Check if score files exist
        if not all(
            [
                content_query_score_path.exists(),
                summary_query_score_path.exists(),
                content_expanded_query_score_path.exists(),
            ]
        ):
            print(
                "   ⚠️ Warning: Some score files not found, skipping parent relevance check"
            )
            return {
                "context": state.get("context", []),
                "examples": state.get("examples", []),
                "scores": "Score files not found",
                "filtered_context": state.get("context", []),
                "filtered_examples": state.get("examples", []),
            }

        print("📂 Loading similarity scores...")
        with open(content_query_score_path, "r", encoding="utf-8") as f:
            content_query_scores = json.load(f)

        with open(summary_query_score_path, "r", encoding="utf-8") as f:
            summary_query_scores = json.load(f)

        with open(content_expanded_query_score_path, "r", encoding="utf-8") as f:
            summary_expanded_query_scores = json.load(f)

        print(
            f"   ✅ Loaded scores - Content: {len(content_query_scores)}, Summary: {len(summary_query_scores)}, Expanded: {len(summary_expanded_query_scores)}"
        )

        # Create score summary string
        content_query_scores_str = " ".join(
            [str(score) for score in content_query_scores]
        )
        summary_query_scores_str = " ".join(
            [str(score) for score in summary_query_scores]
        )
        summary_expanded_query_scores_str = " ".join(
            [str(score) for score in summary_expanded_query_scores]
        )

        score_list = (
            f"Content-Query similarity scores: {content_query_scores_str}\n"
            f"Summary-Query similarity scores: {summary_query_scores_str}\n"
            f"Content-Expanded-Query similarity scores: {summary_expanded_query_scores_str}"
        )

        # Get content and examples from state
        content = state.get("context", [])
        examples = state.get("examples", [])

        if not content and not examples:
            print("   ⚠️ Warning: No content or examples found in state")
            return {
                "context": [],
                "examples": [],
                "scores": score_list,
                "filtered_context": [],
                "filtered_examples": [],
            }

        print(
            f"📊 Filtering content and examples (threshold: {config.SIM_THRESHOLD})..."
        )

        # Filter content based on content-query similarity scores
        content_texts = [
            d.page_content if hasattr(d, "page_content") else str(d) for d in content
        ]
        content_query_filtered_scores = []
        content_query_filtered_context = []

        for i, score in enumerate(content_query_scores):
            if i < len(content_texts) and float(score) >= config.SIM_THRESHOLD:
                content_query_filtered_scores.append(str(score))
                content_query_filtered_context.append(content_texts[i])

        print(
            f"   ✅ Filtered content: {len(content_query_filtered_context)}/{len(content_texts)} documents"
        )

        # Filter examples based on summary-expanded-query similarity scores
        content_examples = []
        if examples:
            if isinstance(examples[0], dict):
                content_examples = [d.get("page_content", str(d)) for d in examples]
            else:
                content_examples = [
                    d.page_content if hasattr(d, "page_content") else str(d)
                    for d in examples
                ]

        examples_expanded_query_filtered_scores = []
        examples_expanded_query_filtered_context = []

        for i, score in enumerate(summary_expanded_query_scores):
            if i < len(content_examples) and float(score) >= config.SIM_THRESHOLD:
                examples_expanded_query_filtered_scores.append(str(score))
                examples_expanded_query_filtered_context.append(content_examples[i])

        print(
            f"   ✅ Filtered examples: {len(examples_expanded_query_filtered_context)}/{len(content_examples)} documents"
        )

        # Check if we have any filtered results
        if not (
            content_query_filtered_context or examples_expanded_query_filtered_context
        ):
            print("   ⚠️ Warning: No documents passed the similarity threshold")
            return {
                "context": [],
                "examples": [],
                "scores": score_list,
                "filtered_context": [],
                "filtered_examples": [],
            }

        # Combine filtered content
        content_query_filtered_str = "\n".join(content_query_filtered_context)
        examples_expanded_query_filtered_str = "\n".join(
            examples_expanded_query_filtered_context
        )

        if content_query_filtered_str and examples_expanded_query_filtered_str:
            filtered_context = (
                content_query_filtered_str + "\n" + examples_expanded_query_filtered_str
            )
        elif content_query_filtered_str:
            filtered_context = content_query_filtered_str
        else:
            filtered_context = examples_expanded_query_filtered_str

        print("✅ Parent relevance check completed successfully")

        return {
            "context": filtered_context,
            "scores": score_list,
            "examples": examples_expanded_query_filtered_context,
            "filtered_context": content_query_filtered_context,
            "filtered_examples": examples_expanded_query_filtered_context,
            "filtered_scores": content_query_filtered_scores
            + examples_expanded_query_filtered_scores,
        }

    except Exception as e:
        print(f"❌ Error in parent relevance check: {e}")
        import traceback

        traceback.print_exc()

        # Return original state on error
        return {
            "context": state.get("context", []),
            "examples": state.get("examples", []),
            "scores": f"Error in relevance check: {str(e)}",
            "filtered_context": state.get("context", []),
            "filtered_examples": state.get("examples", []),
        }


def node_parent_retrieve(state: GraphState) -> GraphState:
    """Parent retrieval node for content and examples databases"""
    query: str = state["question"][-1]

    try:
        content_docs, examples_docs = retrievers.parent_retrieve(query)

        return {
            "context": content_docs,
            "examples": examples_docs,
        }

    except Exception as e:
        print(f"Error in parent retrieve: {e}")
        return {
            "context": [],
            "examples": [],
        }


def node_parent_retrieve_hybrid(state: GraphState) -> GraphState:
    """Parent hybrid retrieval node for content and examples databases"""
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    # 예시 데이터베이스용 하이브리드 가중치 (별도 설정 가능)
    hybrid_weights_examples = state.get("hybrid_weights_examples", [0.5, 0.5])

    try:
        content_docs, examples_docs = retrievers.parent_retrieve_hybrid(
            query, weights=hybrid_weights, weights_examples=hybrid_weights_examples
        )

        return {
            "context": content_docs,
            "examples": examples_docs,
        }

    except Exception as e:
        print(f"Error in parent hybrid retrieve: {e}")
        return {
            "context": [],
            "examples": [],
        }
