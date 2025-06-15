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
    previous_qa_context: str = "",
) -> Dict[str, Any]:
    """
    하위 질문에 대해 검색 및 답변 생성을 수행합니다.
    이전 단계의 질문-답변 컨텍스트를 포함하여 순차적으로 처리합니다.

    Args:
        subquestion: 처리할 하위 질문
        retrieval_type: 검색 타입 (hyde, summary, summary_mean, None)
        hybrid_weights: 하이브리드 검색 가중치
        previous_qa_context: 이전 단계들의 질문-답변 컨텍스트

    Returns:
        검색된 컨텍스트와 답변을 포함한 딕셔너리
    """
    print(f"   Processing subquestion: {subquestion}")
    if previous_qa_context:
        print(
            f"   Using previous Q&A context (length: {len(previous_qa_context)} chars)"
        )

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
    elif retrieval_type == "summary_mean" and hybrid_weights:
        context_docs, explanation = retrievers.summary_mean_retrieve_hybrid(
            subquestion, weights=hybrid_weights
        )
    elif retrieval_type == "summary_mean":
        context_docs, explanation = retrievers.summary_mean_retrieve(subquestion)
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
    retrieved_context = "\n\n---\n\n".join(context_contents)

    # 전체 컨텍스트 구성: 이전 Q&A + 현재 검색된 문서들
    if previous_qa_context:
        full_context = f"""Previous Questions and Answers:
{previous_qa_context}

Current Retrieved Documents:
{retrieved_context}"""
    else:
        full_context = retrieved_context

    # 하위 질문에 대한 답변 생성
    answer = utils.generate_llm_answer(subquestion, full_context)

    print(f"   ✓ Generated answer for subquestion (length: {len(answer)} chars)")

    return {
        "question": subquestion,
        "retrieved_context": retrieved_context,  # 현재 단계에서 검색된 문서들만
        "full_context": full_context,  # 이전 단계 + 현재 검색 결과
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
    combined_answers = ""
    all_retrieved_contexts = []

    for i, result in enumerate(subquestion_results, 1):
        combined_answers += (
            f"{i}. Q: {result['question']}\n   A: {result['answer']}\n\n"
        )

        # 각 단계에서 검색된 문서들만 수집 (중복 방지)
        if result.get("retrieved_context"):
            all_retrieved_contexts.append(
                f"=== Sub-question {i} Context ===\n{result['retrieved_context']}"
            )

    # 전체 검색된 컨텍스트
    combined_retrieved_context = "\n\n".join(all_retrieved_contexts)

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

                                    Sequential sub-questions and their answers:
                                    {combined_answers}

                                    Retrieved context from all searches:
                                    {combined_retrieved_context}

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
        return f"Based on the sequential analysis of sub-questions:\n\n{combined_answers}\n\nThese findings address the original question: {original_query}"


def process_complex_query(
    original_query: str,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
    max_subquestions: int = 5,
) -> Dict[str, Any]:
    """
    복잡한 질문에 대한 전체 처리 과정을 수행합니다.
    하위 질문들을 순차적으로 처리하며, 각 단계에서 이전 단계의 질문-답변을 컨텍스트로 활용합니다.

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

        # 2. 각 하위 질문을 순차적으로 처리
        print("Step 2: Processing sub-questions sequentially...")
        subquestion_results = []
        cumulative_qa_context = ""  # 이전 단계들의 Q&A 누적

        for i, subquestion in enumerate(subquestions, 1):
            try:
                print(f"Processing sub-question {i}/{len(subquestions)}: {subquestion}")

                # 현재 하위 질문을 이전 Q&A 컨텍스트와 함께 처리
                result = process_subquestion(
                    subquestion,
                    retrieval_type,
                    hybrid_weights,
                    previous_qa_context=cumulative_qa_context,
                )

                subquestion_results.append(result)

                # 다음 단계를 위해 현재 Q&A를 누적 컨텍스트에 추가
                current_qa = f"Q{i}: {subquestion}\nA{i}: {result['answer']}\n\n"
                cumulative_qa_context += current_qa

                print(f"  ✓ Successfully processed sub-question {i}")
                print(
                    f"  ✓ Updated cumulative context (length: {len(cumulative_qa_context)} chars)"
                )

            except Exception as e:
                print(f"  ✗ Error processing sub-question {i}: {e}")
                # 에러가 발생한 하위 질문에 대해서도 기본 결과 추가
                error_result = {
                    "question": subquestion,
                    "retrieved_context": "",
                    "full_context": cumulative_qa_context,
                    "answer": f"Error processing this sub-question: {str(e)}",
                    "explanation": "",
                    "context_docs": [],
                }
                subquestion_results.append(error_result)

                # 에러가 발생해도 기본 답변을 누적 컨텍스트에 추가
                current_qa = f"Q{i}: {subquestion}\nA{i}: {error_result['answer']}\n\n"
                cumulative_qa_context += current_qa

        # 3. 결과 종합
        print("Step 3: Aggregating sequential results...")
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
            final_answer = (
                f"Based on the sequential sub-questions analysis:\n\n"
                + "\n\n".join(answers)
            )

        # 모든 컨텍스트 통합
        all_retrieved_contexts = []
        all_context_docs = []

        for i, result in enumerate(subquestion_results, 1):
            if result.get("retrieved_context"):
                all_retrieved_contexts.append(
                    f"=== Step {i} Retrieved Context ===\n{result['retrieved_context']}"
                )
            if result.get("context_docs"):
                all_context_docs.extend(result["context_docs"])

        combined_context = "\n\n".join(all_retrieved_contexts)

        print("✓ Sequential complex query processing completed successfully")

        return {
            "original_query": original_query,
            "subquestions": subquestions,
            "subquestion_results": subquestion_results,
            "final_answer": final_answer,
            "combined_context": combined_context,
            "all_context_docs": all_context_docs,
            "cumulative_qa_context": cumulative_qa_context,  # 전체 Q&A 진행 과정
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
            "cumulative_qa_context": "",
        }


def process_complex_query_with_expansion(
    original_query: str,
    content_docs: List[Document],
    example_docs: List[Dict[str, Any]],
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
    max_subquestions: int = 5,
) -> Dict[str, Any]:
    """
    Process complex query with pre-retrieved content and examples.
    """
    print(f"Processing complex query with expansion: {original_query}")

    try:
        # 1. Create initial context from content and examples
        initial_context_parts = []

        if content_docs:
            content_texts = []
            for doc in content_docs:
                if hasattr(doc, "page_content"):
                    content_texts.append(doc.page_content)
                else:
                    content_texts.append(str(doc))
            if content_texts:
                initial_context_parts.append(
                    "Available Content:\n" + "\n".join(content_texts[:3])
                )  # Limit for context size

        if example_docs:
            example_texts = []
            for doc in example_docs:
                if isinstance(doc, dict):
                    example_texts.append(doc.get("page_content", str(doc)))
                else:
                    example_texts.append(str(doc))
            if example_texts:
                initial_context_parts.append(
                    "Available Examples:\n" + "\n".join(example_texts[:2])
                )  # Limit for context size

        initial_context = "\n\n".join(initial_context_parts)

        # 2. Enhanced query decomposition with context awareness
        print("Step 1: Decomposing query into context-aware sub-questions...")

        try:
            response = utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert in semiconductor physics who excels at breaking down complex questions into simpler, manageable sub-questions.

Given a complex question and available context, break it down into {max_subquestions} or fewer sub-questions that:
1. Are simpler and more focused than the original question
2. Build upon each other logically
3. Take advantage of the available context and examples
4. When answered together, provide comprehensive information to solve the original question
5. Are specific to semiconductor physics domain

Format your response as a numbered list:
1. [First sub-question]
2. [Second sub-question]
...

Respond with ONLY the numbered list of sub-questions.""",
                    },
                    {
                        "role": "user",
                        "content": f"""Complex Question: {original_query}

Available Context:
{initial_context}

Break down this question into focused sub-questions that can leverage the available context.""",
                    },
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()
            subquestions = []
            lines = response_text.split("\n")

            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    if ". " in line:
                        question = line.split(". ", 1)[1].strip()
                    elif "- " in line:
                        question = line.split("- ", 1)[1].strip()
                    else:
                        question = line.strip()

                    if question and len(question) > 10:
                        subquestions.append(question)

            if not subquestions:
                print("Warning: No sub-questions generated, using fallback")
                subquestions = [
                    f"What are the fundamental concepts related to: {original_query}?",
                    f"What are the key factors that influence: {original_query}?",
                ]

            subquestions = subquestions[:max_subquestions]

        except Exception as e:
            print(f"Error in enhanced query decomposition: {e}")
            subquestions = decompose_query(original_query, max_subquestions)

        if not subquestions:
            return {
                "original_query": original_query,
                "subquestions": [original_query],
                "subquestion_results": [],
                "final_answer": f"Unable to decompose query: {original_query}",
                "combined_context": initial_context,
                "all_context_docs": content_docs,
            }

        print(f"Generated {len(subquestions)} sub-questions:")
        for i, sq in enumerate(subquestions, 1):
            print(f"  {i}. {sq}")

        # 3. Process subquestions sequentially with initial context
        print("Step 2: Processing sub-questions sequentially...")
        subquestion_results = []
        cumulative_qa_context = f"Initial Context:\n{initial_context}\n\n"

        for i, subquestion in enumerate(subquestions, 1):
            try:
                print(f"Processing sub-question {i}/{len(subquestions)}: {subquestion}")

                result = process_subquestion(
                    subquestion,
                    retrieval_type,
                    hybrid_weights,
                    previous_qa_context=cumulative_qa_context,
                )

                subquestion_results.append(result)

                current_qa = f"Q{i}: {subquestion}\nA{i}: {result['answer']}\n\n"
                cumulative_qa_context += current_qa

                print(f"  ✓ Successfully processed sub-question {i}")

            except Exception as e:
                print(f"  ✗ Error processing sub-question {i}: {e}")
                error_result = {
                    "question": subquestion,
                    "retrieved_context": "",
                    "full_context": cumulative_qa_context,
                    "answer": f"Error processing: {str(e)}",
                    "explanation": "",
                    "context_docs": [],
                }
                subquestion_results.append(error_result)

                current_qa = f"Q{i}: {subquestion}\nA{i}: {error_result['answer']}\n\n"
                cumulative_qa_context += current_qa

        # 4. Final aggregation (reuse existing function)
        print("Step 3: Aggregating results...")
        try:
            final_answer = aggregate_subquestion_results(
                original_query, subquestion_results
            )
        except Exception as e:
            print(f"Error in aggregation: {e}")
            answers = [
                result["answer"] for result in subquestion_results if result["answer"]
            ]
            final_answer = f"Based on analysis:\n\n" + "\n\n".join(answers)

        # 5. Combine all contexts
        all_retrieved_contexts = []
        all_context_docs = list(content_docs)  # Start with initial content docs

        for i, result in enumerate(subquestion_results, 1):
            if result.get("retrieved_context"):
                all_retrieved_contexts.append(
                    f"=== Step {i} Retrieved Context ===\n{result['retrieved_context']}"
                )
            if result.get("context_docs"):
                all_context_docs.extend(result["context_docs"])

        combined_context = (
            initial_context + "\n\n" + "\n\n".join(all_retrieved_contexts)
        )

        print("✓ Complex query processing with expansion completed successfully")

        return {
            "original_query": original_query,
            "subquestions": subquestions,
            "subquestion_results": subquestion_results,
            "final_answer": final_answer,
            "combined_context": combined_context,
            "all_context_docs": all_context_docs,
            "cumulative_qa_context": cumulative_qa_context,
        }

    except Exception as e:
        print(f"Critical error in complex query processing: {e}")
        return {
            "original_query": original_query,
            "subquestions": [original_query],
            "subquestion_results": [],
            "final_answer": f"Error processing complex query: {str(e)}",
            "combined_context": "",
            "all_context_docs": content_docs,
            "cumulative_qa_context": "",
        }
