from __future__ import annotations
from typing import List, Dict, Any, Optional
from rag_pipeline import utils, retrievers, config
from langchain.schema import Document
import json


class PlanExecuteAgent:
    """Plan and Execute Agent for complex multi-hop questions"""
    
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        
    def create_plan(self, query: str, context_so_far: str = "") -> Dict[str, Any]:
        """
        복잡한 질문에 대한 실행 계획을 수립합니다.
        
        Args:
            query: 원본 복잡한 질문
            context_so_far: 지금까지 수집된 컨텍스트
            
        Returns:
            계획 정보를 포함한 딕셔너리
        """
        try:
            response = utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in semiconductor physics who excels at creating step-by-step plans to solve complex problems.

                        When given a complex question, create a detailed plan with specific, actionable steps that:
                        1. Break down the problem into logical components
                        2. Identify what specific information needs to be retrieved (concepts, formulas, data)
                        3. Determine the sequence of steps needed to solve the problem
                        4. Each step should be specific and focused on retrieving or analyzing particular information

                        Format your response as a JSON object with this structure:
                        {
                            "overall_goal": "What we're trying to achieve",
                            "current_step": 1,
                            "steps": [
                                {
                                    "step_number": 1,
                                    "action": "retrieve_concept|retrieve_formula|retrieve_data|analyze|calculate",
                                    "description": "What specifically to do in this step",
                                    "search_query": "Specific query to search for information",
                                    "expected_outcome": "What information we expect to get"
                                }
                            ],
                            "success_criteria": "How we'll know when the problem is solved"
                        }

                        Available actions:
                        - retrieve_concept: Search for fundamental concepts or definitions
                        - retrieve_formula: Search for specific equations or formulas
                        - retrieve_data: Search for numerical values, parameters, or experimental data
                        - analyze: Analyze gathered information to draw conclusions
                        - calculate: Perform calculations using gathered formulas and data

                        Respond with ONLY the JSON object."""
                    },
                    {
                        "role": "user",
                        "content": f"""Complex Question: {query}

                        Context gathered so far:
                        {context_so_far if context_so_far else "None"}

                        Create a detailed plan to solve this semiconductor physics problem."""
                    }
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                plan = json.loads(response_text)
                return plan
            except json.JSONDecodeError:
                print(f"Failed to parse plan JSON: {response_text}")
                # 폴백 계획
                return self._create_fallback_plan(query)
                
        except Exception as e:
            print(f"Error creating plan: {e}")
            return self._create_fallback_plan(query)
    
    def _create_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Create a simple fallback plan when main planning fails"""
        return {
            "overall_goal": f"Solve the complex question: {query}",
            "current_step": 1,
            "steps": [
                {
                    "step_number": 1,
                    "action": "retrieve_concept",
                    "description": "Gather fundamental concepts related to the question",
                    "search_query": f"fundamental concepts related to: {query}",
                    "expected_outcome": "Basic understanding of relevant concepts"
                },
                {
                    "step_number": 2,
                    "action": "retrieve_formula",
                    "description": "Find relevant formulas and equations",
                    "search_query": f"formulas equations for: {query}",
                    "expected_outcome": "Mathematical relationships needed"
                },
                {
                    "step_number": 3,
                    "action": "analyze",
                    "description": "Analyze gathered information to solve the problem",
                    "search_query": "",
                    "expected_outcome": "Solution to the original question"
                }
            ],
            "success_criteria": "All aspects of the question are addressed with supporting evidence"
        }
    
    def execute_step(
        self, 
        step: Dict[str, Any], 
        retrieval_type: str = None, 
        hybrid_weights: List[float] = None
    ) -> Dict[str, Any]:
        """
        계획의 단일 단계를 실행합니다.
        
        Args:
            step: 실행할 단계 정보
            retrieval_type: 검색 타입
            hybrid_weights: 하이브리드 검색 가중치
            
        Returns:
            실행 결과
        """
        action = step.get("action", "retrieve_concept")
        search_query = step.get("search_query", "")
        description = step.get("description", "")
        
        print(f"Executing step {step.get('step_number', '?')}: {action}")
        print(f"Description: {description}")
        print(f"Search query: {search_query}")
        
        if action in ["retrieve_concept", "retrieve_formula", "retrieve_data"] and search_query:
            # 정보 검색 단계
            try:
                context_docs = self._perform_retrieval(search_query, retrieval_type, hybrid_weights)
                
                context_contents = [
                    doc.page_content for doc in context_docs if isinstance(doc, Document)
                ]
                context_str = "\n\n---\n\n".join(context_contents)
                
                # 검색된 정보에 대한 간단한 요약 생성
                if context_str.strip():
                    summary = self._summarize_retrieved_info(search_query, context_str, action)
                else:
                    summary = f"No relevant information found for: {search_query}"
                
                return {
                    "step_number": step.get("step_number"),
                    "action": action,
                    "search_query": search_query,
                    "context": context_str,
                    "summary": summary,
                    "context_docs": context_docs,
                    "success": bool(context_str.strip())
                }
                
            except Exception as e:
                print(f"Error in retrieval step: {e}")
                return {
                    "step_number": step.get("step_number"),
                    "action": action,
                    "search_query": search_query,
                    "context": "",
                    "summary": f"Error retrieving information: {str(e)}",
                    "context_docs": [],
                    "success": False
                }
        
        elif action in ["analyze", "calculate"]:
            # 분석/계산 단계 - 이전 단계들의 결과를 사용
            return {
                "step_number": step.get("step_number"),
                "action": action,
                "search_query": "",
                "context": "",
                "summary": f"Analysis/calculation step: {description}",
                "context_docs": [],
                "success": True
            }
        
        else:
            return {
                "step_number": step.get("step_number"),
                "action": action,
                "search_query": search_query,
                "context": "",
                "summary": f"Unknown action or empty search query",
                "context_docs": [],
                "success": False
            }
    
    def _perform_retrieval(
        self, 
        query: str, 
        retrieval_type: str = None, 
        hybrid_weights: List[float] = None
    ) -> List[Document]:
        """실제 정보 검색을 수행합니다."""
        if retrieval_type == "hyde" and hybrid_weights:
            context_docs, _ = retrievers.hyde_hybrid_retrieve(query, weights=hybrid_weights)
        elif retrieval_type == "hyde":
            context_docs, _ = retrievers.hyde_retrieve(query)
        elif retrieval_type == "summary" and hybrid_weights:
            context_docs, _ = retrievers.summary_hybrid_retrieve(query, weights=hybrid_weights)
        elif retrieval_type == "summary":
            context_docs, _ = retrievers.summary_retrieve(query)
        elif hybrid_weights:
            context_docs = retrievers.vectordb_hybrid_retrieve(query, weights=hybrid_weights)
        else:
            context_docs = retrievers.vectordb_retrieve(query)
        
        # tuple인 경우 첫 번째 요소가 문서들
        if isinstance(context_docs, tuple):
            context_docs = context_docs[0]
            
        return context_docs
    
    def _summarize_retrieved_info(self, query: str, context: str, action: str) -> str:
        """검색된 정보를 요약합니다."""
        try:
            response = utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert in semiconductor physics. Summarize the retrieved information in a clear, concise way that's relevant to the search query.

                        Action type: {action}
                        - If retrieve_concept: Focus on definitions and fundamental principles
                        - If retrieve_formula: Focus on equations and mathematical relationships  
                        - If retrieve_data: Focus on numerical values and parameters

                        Keep the summary focused and technically accurate."""
                    },
                    {
                        "role": "user",
                        "content": f"""Search Query: {query}

                        Retrieved Information:
                        {context}

                        Provide a concise summary of the key information that's relevant to the search query."""
                    }
                ],
                max_tokens=300,
                temperature=0.2,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error summarizing retrieved info: {e}")
            return f"Retrieved information for: {query} (summary generation failed)"
    
    def evaluate_progress(
        self, 
        original_query: str, 
        plan: Dict[str, Any], 
        executed_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        현재까지의 진행 상황을 평가하고 다음 행동을 결정합니다.
        
        Args:
            original_query: 원본 질문
            plan: 현재 계획
            executed_steps: 지금까지 실행된 단계들
            
        Returns:
            평가 결과 및 다음 행동
        """
        try:
            # 지금까지 수집된 모든 정보 정리
            all_context = []
            all_summaries = []
            
            for step in executed_steps:
                if step.get("context"):
                    all_context.append(f"Step {step['step_number']} ({step['action']}): {step['context']}")
                if step.get("summary"):
                    all_summaries.append(f"Step {step['step_number']}: {step['summary']}")
            
            combined_context = "\n\n".join(all_context)
            combined_summaries = "\n".join(all_summaries)
            
            response = utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in semiconductor physics who evaluates problem-solving progress.

                        Analyze the progress made so far and determine the next action:

                        1. CONTINUE: If more information is needed, continue with the next planned step
                        2. REPLAN: If the current plan isn't working, create a new plan
                        3. COMPLETE: If enough information has been gathered to answer the original question

                        Respond with a JSON object:
                        {
                            "decision": "CONTINUE|REPLAN|COMPLETE",
                            "reasoning": "Why this decision was made",
                            "confidence": 0.8,
                            "missing_info": ["What information is still needed"],
                            "can_answer": true/false
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"""Original Question: {original_query}

                        Current Plan Goal: {plan.get('overall_goal', 'Not specified')}
                        Success Criteria: {plan.get('success_criteria', 'Not specified')}

                        Steps Executed So Far:
                        {combined_summaries}

                        Remaining Planned Steps: {len(plan.get('steps', [])) - len(executed_steps)}

                        Evaluate the progress and determine the next action."""
                    }
                ],
                max_tokens=500,
                temperature=0.2,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            try:
                evaluation = json.loads(response_text)
                return evaluation
            except json.JSONDecodeError:
                print(f"Failed to parse evaluation JSON: {response_text}")
                # 폴백 평가
                return {
                    "decision": "CONTINUE" if len(executed_steps) < len(plan.get('steps', [])) else "COMPLETE",
                    "reasoning": "Fallback evaluation due to parsing error",
                    "confidence": 0.5,
                    "missing_info": [],
                    "can_answer": len(executed_steps) >= 2
                }
                
        except Exception as e:
            print(f"Error in progress evaluation: {e}")
            return {
                "decision": "COMPLETE",
                "reasoning": f"Error in evaluation: {str(e)}",
                "confidence": 0.3,
                "missing_info": [],
                "can_answer": True
            }
    
    def generate_final_answer(
        self, 
        original_query: str, 
        executed_steps: List[Dict[str, Any]]
    ) -> str:
        """
        실행된 단계들의 결과를 바탕으로 최종 답변을 생성합니다.
        
        Args:
            original_query: 원본 질문
            executed_steps: 실행된 모든 단계들
            
        Returns:
            최종 답변
        """
        try:
            # 모든 단계의 정보 통합
            step_summaries = []
            all_context = []
            
            for step in executed_steps:
                step_info = f"Step {step['step_number']} ({step['action']}): {step['summary']}"
                step_summaries.append(step_info)
                
                if step.get("context"):
                    all_context.append(step["context"])
            
            combined_context = "\n\n=== STEP CONTEXT ===\n\n".join(all_context)
            step_summary = "\n".join(step_summaries)
            
            response = utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in semiconductor physics who provides comprehensive final answers.

                        Based on the step-by-step investigation and gathered information, provide a complete answer that:
                        1. Directly addresses the original question
                        2. Uses the information gathered in each step
                        3. Shows clear reasoning and connections
                        4. Is technically accurate for semiconductor physics
                        5. Includes relevant details, formulas, and explanations

                        Structure your response clearly with proper reasoning."""
                    },
                    {
                        "role": "user",
                        "content": f"""Original Question: {original_query}

                        Investigation Summary:
                        {step_summary}

                        Detailed Information Gathered:
                        {combined_context}

                        Provide a comprehensive final answer to the original question."""
                    }
                ],
                max_tokens=2000,
                temperature=0.2,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating final answer: {e}")
            # 폴백: 단계 요약만 제공
            step_info = "\n".join([f"Step {s['step_number']}: {s['summary']}" for s in executed_steps])
            return f"Based on the investigation:\n\n{step_info}\n\nError occurred while generating detailed answer: {str(e)}"


def process_complex_query_with_plan_execute(
    original_query: str,
    retrieval_type: str = None,
    hybrid_weights: List[float] = None,
    max_steps: int = 5
) -> Dict[str, Any]:
    """
    Plan and Execute agent를 사용하여 복잡한 질문을 처리합니다.
    
    Args:
        original_query: 원본 복잡한 질문
        retrieval_type: 검색 타입
        hybrid_weights: 하이브리드 검색 가중치
        max_steps: 최대 실행 단계 수
        
    Returns:
        처리 결과
    """
    agent = PlanExecuteAgent(max_steps=max_steps)
    
    print(f"🚀 Starting Plan and Execute for: {original_query}")
    
    # 1. 초기 계획 수립
    print("📋 Step 1: Creating initial plan...")
    plan = agent.create_plan(original_query)
    print(f"   Plan created with {len(plan.get('steps', []))} steps")
    print(f"   Goal: {plan.get('overall_goal', 'Not specified')}")
    
    executed_steps = []
    current_step_index = 0
    
    # 2. 계획 실행 루프
    for iteration in range(max_steps):
        print(f"\n🔄 Iteration {iteration + 1}/{max_steps}")
        
        # 현재 단계 실행
        if current_step_index < len(plan.get('steps', [])):
            current_step = plan['steps'][current_step_index]
            print(f"   Executing planned step {current_step_index + 1}")
            
            step_result = agent.execute_step(current_step, retrieval_type, hybrid_weights)
            executed_steps.append(step_result)
            current_step_index += 1
        
        # 진행 상황 평가
        print("   Evaluating progress...")
        evaluation = agent.evaluate_progress(original_query, plan, executed_steps)
        
        decision = evaluation.get('decision', 'CONTINUE')
        print(f"   Decision: {decision}")
        print(f"   Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
        
        if decision == "COMPLETE":
            print("   ✅ Problem solving complete!")
            break
        elif decision == "REPLAN":
            print("   🔄 Replanning...")
            # 지금까지의 컨텍스트를 포함하여 재계획
            context_so_far = "\n".join([s.get('summary', '') for s in executed_steps])
            plan = agent.create_plan(original_query, context_so_far)
            current_step_index = 0
            print(f"   New plan created with {len(plan.get('steps', []))} steps")
        elif decision == "CONTINUE":
            print("   ➡️ Continuing with execution...")
            # 현재 계획대로 계속 진행
            continue
    
    # 3. 최종 답변 생성
    print("\n📝 Generating final answer...")
    final_answer = agent.generate_final_answer(original_query, executed_steps)
    
    # 모든 컨텍스트 통합
    all_context_docs = []
    combined_context = []
    
    for step in executed_steps:
        if step.get('context_docs'):
            all_context_docs.extend(step['context_docs'])
        if step.get('context'):
            combined_context.append(f"Step {step['step_number']}: {step['context']}")
    
    combined_context_str = "\n\n".join(combined_context)
    
    print("✅ Plan and Execute completed successfully")
    
    return {
        "original_query": original_query,
        "plan": plan,
        "executed_steps": executed_steps,
        "final_answer": final_answer,
        "combined_context": combined_context_str,
        "all_context_docs": all_context_docs,
        "total_steps": len(executed_steps)
    }
