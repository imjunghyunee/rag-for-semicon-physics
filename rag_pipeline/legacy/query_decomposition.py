from __future__ import annotations
from typing import List, Dict, Any
from rag_pipeline import utils, retrievers, config
from langchain.schema import Document


def decompose_query(original_query: str, max_subquestions: int = 5) -> List[str]:
    """
    복잡한 질문을 더 작은 하위 질문들로 분해합니다.

    Args:
        original_query: 원본 복잡한 질문
        max_subquestions: 최대 하위 질문 개수

    Returns:
        하위 질문들의 리스트
    """
    try:
        response = utils.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert in semiconductor physics who excels at breaking down complex questions into simpler, manageable sub-questions.

                                    When given a complex question, break it down into {max_subquestions} or fewer sub-questions that:
                                    1. Are simpler and more focused than the original question
                                    2. Build upon each other logically
                                    3. When answered together, provide comprehensive information to solve the original question
                                    4. Are specific to semiconductor physics domain

                                    Examples of good decomposition:
                                    - Complex: "How does temperature affect both carrier concentration and mobility in silicon devices?"
                                    - Sub-questions:
                                    1. What is the relationship between temperature and intrinsic carrier concentration in silicon?
                                    2. How does temperature affect electron and hole mobility in silicon?
                                    3. What are the combined effects on device performance?

                                    Format your response as a numbered list:
                                    1. [First sub-question]
                                    2. [Second sub-question]
                                    ...

                                    Respond with ONLY the numbered list of sub-questions.""",
                },
                {
                    "role": "user",
                    "content": f"Let's break down this complex semiconductor physics question: {original_query}",
                },
            ],
            max_tokens=5000,
            temperature=0.3,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse numbered list into individual questions
        subquestions = []
        lines = response_text.split("\n")

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering and clean up
                if ". " in line:
                    question = line.split(". ", 1)[1].strip()
                elif "- " in line:
                    question = line.split("- ", 1)[1].strip()
                else:
                    question = line.strip()

                if question and len(question) > 10:  # Filter out very short responses
                    subquestions.append(question)

        # Fallback if no valid sub-questions generated
        if not subquestions:
            print(
                "Warning: No valid sub-questions generated, creating fallback questions"
            )
            # Simple fallback strategy
            subquestions = [
                f"What are the fundamental concepts related to: {original_query}?",
                f"What are the key factors that influence: {original_query}?",
                f"How can we analyze or solve: {original_query}?",
            ]

        # Limit to max_subquestions
        return subquestions[:max_subquestions]

    except Exception as e:
        print(f"Error in query decomposition: {e}")
        # Return fallback questions
        return [
            f"What are the basic principles underlying this question: {original_query}?",
            f"What specific factors should be considered for: {original_query}?",
        ]


def process_subquestion(
    subquestion: str,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
) -> Dict[str, Any]:
    """
    하위 질문에 대해 검색 및 답변 생성을 수행합니다.

    Args:
        subquestion: 처리할 하위 질문
        retrieval_type: 검색 타입 (hyde, summary, None)
        hybrid_weights: 하이브리드 검색 가중치

    Returns:
        검색된 컨텍스트와 답변을 포함한 딕셔너리
    """
    # 검색 타입에 따라 적절한 검색 함수 선택
    if retrieval_type == "hyde" and hybrid_weights:
        context_docs, explanation = retrievers.hyde_hybrid_retrieve(
            subquestion, weights=hybrid_weights
        )
    elif retrieval_type == "hyde":
        context_docs, explanation = retrievers.hyde_retrieve(subquestion)
    elif retrieval_type == "summary" and hybrid_weights:
        context_docs, explanation = retrievers.summary_hybrid_retrieve(
            subquestion, weights=hybrid_weights
        )
    elif retrieval_type == "summary":
        context_docs, explanation = retrievers.summary_retrieve(subquestion)
    elif hybrid_weights:
        context_docs = retrievers.vectordb_hybrid_retrieve(
            subquestion, weights=hybrid_weights
        )
        explanation = ""
    else:
        context_docs = retrievers.vectordb_retrieve(subquestion)
        explanation = ""

    # 컨텍스트 문서들을 문자열로 변환
    if isinstance(context_docs, tuple):
        context_docs = context_docs[0]  # tuple인 경우 첫 번째 요소가 문서들

    context_contents = [
        doc.page_content for doc in context_docs if isinstance(doc, Document)
    ]
    context_str = "\n\n---\n\n".join(context_contents)

    # 하위 질문에 대한 답변 생성
    answer = utils.generate_llm_answer(subquestion, context_str)

    return {
        "question": subquestion,
        "context": context_str,
        "answer": answer,
        "explanation": explanation,
        "context_docs": context_docs,
    }


def aggregate_subquestion_results(
    original_query: str, subquestion_results: List[Dict[str, Any]]
) -> str:
    """
    하위 질문들의 결과를 종합하여 원본 질문에 대한 최종 답변을 생성합니다.

    Args:
        original_query: 원본 복잡한 질문
        subquestion_results: 하위 질문들의 처리 결과 리스트

    Returns:
        최종 종합 답변
    """
    if not subquestion_results:
        return f"Unable to process the complex question: {original_query}"

    # 모든 하위 질문의 컨텍스트와 답변을 종합
    combined_context = ""
    combined_answers = ""

    for i, result in enumerate(subquestion_results, 1):
        combined_context += f"=== Sub-question {i}: {result['question']} ===\n"
        combined_context += f"Context: {result['context']}\n"
        combined_context += f"Answer: {result['answer']}\n\n"

        combined_answers += (
            f"{i}. Q: {result['question']}\n   A: {result['answer']}\n\n"
        )

    try:
        final_answer = (
            utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in semiconductor physics who excels at synthesizing complex information to provide comprehensive answers.

                                    Based on sub-question answers and context, provide a comprehensive, well-structured final answer that:
                                    1. Synthesizes information from all sub-questions
                                    2. Addresses the original question directly and completely
                                    3. Is technically accurate for semiconductor physics domain
                                    4. Includes relevant details and explanations
                                    5. Shows how the sub-answers connect to form the complete solution

                                    Structure your response clearly with proper reasoning and conclusions.""",
                    },
                    {
                        "role": "user",
                        "content": f"""Original Complex Question: {original_query}

                                    Sub-questions and their answers:
                                    {combined_answers}

                                    Detailed context and information:
                                    {combined_context}

                                    Please provide a comprehensive final answer to the original question.""",
                    },
                ],
                max_tokens=2000,
                temperature=0.2,
            )
            .choices[0]
            .message.content.strip()
        )

        return final_answer

    except Exception as e:
        print(f"Error in final answer generation: {e}")
        # Fallback: simple concatenation
        return f"Based on the analysis of sub-questions:\n\n{combined_answers}\n\nThese findings address the original question: {original_query}"


def process_complex_query(
    original_query: str,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
    max_subquestions: int = 5,
) -> Dict[str, Any]:
    """
    복잡한 질문에 대한 전체 처리 과정을 수행합니다.

    Args:
        original_query: 원본 복잡한 질문
        retrieval_type: 검색 타입
        hybrid_weights: 하이브리드 검색 가중치
        max_subquestions: 최대 하위 질문 개수

    Returns:
        최종 결과를 포함한 딕셔너리
    """
    print(f"Processing complex query: {original_query}")

    try:
        # 1. 질문 분해
        print("Step 1: Decomposing query into sub-questions...")
        subquestions = decompose_query(original_query, max_subquestions)

        if not subquestions:
            print(
                "Warning: No sub-questions generated, falling back to simple processing"
            )
            # 폴백: 원래 질문을 그대로 처리
            return {
                "original_query": original_query,
                "subquestions": [original_query],
                "subquestion_results": [],
                "final_answer": f"Unable to decompose query. Processing as simple query: {original_query}",
                "combined_context": "",
                "all_context_docs": [],
            }

        print(f"Generated {len(subquestions)} sub-questions:")
        for i, sq in enumerate(subquestions, 1):
            print(f"  {i}. {sq}")

        # 2. 각 하위 질문 처리
        print("Step 2: Processing each sub-question...")
        subquestion_results = []

        for i, subquestion in enumerate(subquestions, 1):
            try:
                print(f"Processing sub-question {i}/{len(subquestions)}: {subquestion}")
                result = process_subquestion(
                    subquestion, retrieval_type, hybrid_weights
                )
                subquestion_results.append(result)
                print(f"  ✓ Successfully processed sub-question {i}")
            except Exception as e:
                print(f"  ✗ Error processing sub-question {i}: {e}")
                # 에러가 발생한 하위 질문에 대해서도 기본 결과 추가
                subquestion_results.append(
                    {
                        "question": subquestion,
                        "context": "",
                        "answer": f"Error processing this sub-question: {str(e)}",
                        "explanation": "",
                        "context_docs": [],
                    }
                )

        # 3. 결과 종합
        print("Step 3: Aggregating results...")
        try:
            final_answer = aggregate_subquestion_results(
                original_query, subquestion_results
            )
        except Exception as e:
            print(f"Error in aggregation: {e}")
            # 폴백: 간단한 답변 조합
            answers = [
                result["answer"] for result in subquestion_results if result["answer"]
            ]
            final_answer = f"Based on the sub-questions analysis:\n\n" + "\n\n".join(
                answers
            )

        # 모든 컨텍스트 통합
        all_contexts = []
        all_context_docs = []

        for result in subquestion_results:
            if result.get("context"):
                all_contexts.append(result["context"])
            if result.get("context_docs"):
                all_context_docs.extend(result["context_docs"])

        combined_context = (
            "\n\n=== COMBINED CONTEXT FROM ALL SUB-QUESTIONS ===\n\n".join(all_contexts)
        )

        print("✓ Complex query processing completed successfully")

        return {
            "original_query": original_query,
            "subquestions": subquestions,
            "subquestion_results": subquestion_results,
            "final_answer": final_answer,
            "combined_context": combined_context,
            "all_context_docs": all_context_docs,
        }

    except Exception as e:
        print(f"Critical error in complex query processing: {e}")
        # 최종 폴백
        return {
            "original_query": original_query,
            "subquestions": [original_query],
            "subquestion_results": [],
            "final_answer": f"Error processing complex query: {str(e)}. Please try rephrasing your question.",
            "combined_context": "",
            "all_context_docs": [],
        }
