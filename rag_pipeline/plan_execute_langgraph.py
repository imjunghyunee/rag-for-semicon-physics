from __future__ import annotations
from typing import List, Dict, Any, Annotated, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
import json

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_pipeline import utils, retrievers, config
from langchain.schema import Document


class PlanExecuteState(TypedDict):
    """State for Plan and Execute agent"""
    input: str
    plan: List[str] 
    past_steps: Annotated[List[tuple], operator.add]
    response: str
    current_step_index: int
    retrieval_type: str
    hybrid_weights: List[float]
    all_context_docs: List[Document]
    combined_context: str


class Plan(BaseModel):
    """Plan to follow for solving semiconductor physics problems"""
    steps: List[str] = Field(
        description="Different steps to follow for solving the problem, should be in sorted order"
    )


class Response(BaseModel):
    """Final response to user"""
    response: str


class Act(BaseModel):
    """Action to perform - either respond or continue with new plan"""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use retrieval tools to get more information, use Plan."
    )


class PlanExecuteLangGraph:
    """LangGraph-based Plan and Execute Agent for semiconductor physics questions"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.2)
        self._setup_prompts()
        self.graph = self._build_graph()
    
    def _setup_prompts(self):
        """Setup prompts for planner and replanner"""
        self.planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in semiconductor physics who excels at creating step-by-step plans to solve complex problems.

For the given objective, come up with a simple step by step plan that involves individual tasks.
Each step should be specific and focused on retrieving or analyzing particular information needed for semiconductor physics problems.

Your plan should:
1. Break down the problem into logical components
2. Identify what specific information needs to be retrieved (concepts, formulas, data)
3. Determine the sequence of steps needed to solve the problem
4. Each step should be actionable and focused

Available retrieval actions:
- Search for fundamental concepts or definitions
- Search for specific equations or formulas  
- Search for numerical values, parameters, or experimental data
- Analyze gathered information to draw conclusions

The result of the final step should lead to the final answer. Make sure that each step has all the information needed - do not skip steps."""
            ),
            ("user", "{input}")
        ])
        
        self.replanner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
        )
        
        self.planner = self.planner_prompt | self.llm.with_structured_output(Plan)
        self.replanner = self.replanner_prompt | self.llm.with_structured_output(Act)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(PlanExecuteState)
        
        # Add nodes
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("agent", self.execute_step)
        workflow.add_node("replan", self.replan_step)
        
        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        
        # Conditional edge from replan
        workflow.add_conditional_edges(
            "replan",
            self.should_end,
            ["agent", END],
        )
        
        return workflow.compile()
    
    async def plan_step(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Create initial plan for the complex question"""
        print(f"📋 Creating plan for: {state['input']}")
        
        try:
            plan_result = await self.planner.ainvoke({"input": state["input"]})
            steps = plan_result.steps
            
            print(f"   Plan created with {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")
                
            return {
                "plan": steps,
                "current_step_index": 0,
                "all_context_docs": [],
                "combined_context": ""
            }
            
        except Exception as e:
            print(f"Error in planning: {e}")
            # Fallback plan
            fallback_steps = [
                f"Search for fundamental concepts related to: {state['input']}",
                f"Find relevant formulas and equations for: {state['input']}",
                f"Analyze the gathered information to solve: {state['input']}"
            ]
            return {
                "plan": fallback_steps,
                "current_step_index": 0,
                "all_context_docs": [],
                "combined_context": ""
            }
    
    async def execute_step(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Execute a single step of the plan"""
        plan = state["plan"]
        current_index = state.get("current_step_index", 0)
        
        if current_index >= len(plan):
            print("⚠️ No more steps to execute")
            return {"past_steps": []}
        
        current_step = plan[current_index]
        print(f"🔄 Executing step {current_index + 1}/{len(plan)}: {current_step}")
        
        try:
            # Perform retrieval based on the step
            retrieval_type = state.get("retrieval_type")
            hybrid_weights = state.get("hybrid_weights")
            
            context_docs = await self._perform_retrieval_async(
                current_step, retrieval_type, hybrid_weights
            )
            
            # Extract context
            context_contents = [
                doc.page_content for doc in context_docs if isinstance(doc, Document)
            ]
            context_str = "\n\n---\n\n".join(context_contents)
            
            # Generate summary of retrieved information
            if context_str.strip():
                summary = await self._summarize_step_result(current_step, context_str)
                step_result = f"Retrieved information: {summary}"
            else:
                step_result = f"No relevant information found for: {current_step}"
            
            print(f"   ✅ Step completed: {step_result[:100]}...")
            
            # Update state
            all_context_docs = state.get("all_context_docs", [])
            all_context_docs.extend(context_docs)
            
            combined_context = state.get("combined_context", "")
            if combined_context:
                combined_context += f"\n\n=== Step {current_index + 1} ===\n{context_str}"
            else:
                combined_context = f"=== Step {current_index + 1} ===\n{context_str}"
            
            return {
                "past_steps": [(current_step, step_result)],
                "current_step_index": current_index + 1,
                "all_context_docs": all_context_docs,
                "combined_context": combined_context
            }
            
        except Exception as e:
            print(f"❌ Error executing step: {e}")
            error_result = f"Error executing step: {str(e)}"
            return {
                "past_steps": [(current_step, error_result)],
                "current_step_index": current_index + 1
            }
    
    def should_end(self, state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Determine if we should continue or end"""
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"
    
    async def _perform_retrieval_async(
        self, 
        query: str, 
        retrieval_type: str = None, 
        hybrid_weights: List[float] = None
    ) -> List[Document]:
        """Async wrapper for retrieval operations"""
        print(f"🔍 Performing retrieval for: {query[:50]}...")
        print(f"   Retrieval type: {retrieval_type}")
        print(f"   Hybrid weights: {hybrid_weights}")
        
        try:
            # Since our retrieval functions are sync, we'll call them directly
            # In a production system, you might want to make these truly async
            
            if retrieval_type == "hyde" and hybrid_weights:
                print("   Using HyDE + Hybrid retrieval")
                context_docs, _ = retrievers.hyde_hybrid_retrieve(query, weights=hybrid_weights)
            elif retrieval_type == "hyde":
                print("   Using HyDE retrieval")
                context_docs, _ = retrievers.hyde_retrieve(query)
            elif retrieval_type == "summary" and hybrid_weights:
                print("   Using Summary + Hybrid retrieval")
                context_docs, _ = retrievers.summary_hybrid_retrieve(query, weights=hybrid_weights)
            elif retrieval_type == "summary":
                print("   Using Summary retrieval")
                context_docs, _ = retrievers.summary_retrieve(query)
            elif hybrid_weights:
                print("   Using Hybrid retrieval")
                context_docs = retrievers.vectordb_hybrid_retrieve(query, weights=hybrid_weights)
            else:
                print("   Using basic vector retrieval")
                context_docs = retrievers.vectordb_retrieve(query)
            
            # Handle tuple return values
            if isinstance(context_docs, tuple):
                context_docs = context_docs[0]
            
            print(f"   ✅ Retrieved {len(context_docs)} documents")
            return context_docs
            
        except Exception as e:
            print(f"   ❌ Error in retrieval: {e}")
            print(f"   Error type: {type(e).__name__}")
            return []
    
    async def _summarize_step_result(self, step: str, context: str) -> str:
        """Summarize the result of a step"""
        try:
            # 🔥 async/await 제거하고 sync 호출로 변경
            response = utils.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in semiconductor physics. Summarize the retrieved information in a clear, concise way that's relevant to the step being executed.

Keep the summary focused and technically accurate for semiconductor physics domain."""
                    },
                    {
                        "role": "user",
                        "content": f"""Step: {step}

Retrieved Information:
{context}

Provide a concise summary of the key information that's relevant to this step."""
                    }
                ],
                max_tokens=300,
                temperature=0.2,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error summarizing step result: {e}")
            return f"Retrieved information for step: {step}"
    
    async def replan_step(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Evaluate progress and decide next action - 무한 루프 방지 로직 추가"""
        print("🤔 Evaluating progress and replanning...")
        
        try:
            # 🔥 무한 루프 방지: 이미 충분한 단계를 실행했다면 종료
            past_steps = state.get("past_steps", [])
            if len(past_steps) >= 3:  # 3단계 이상 실행했으면 종료
                print("✅ Sufficient steps completed, generating final response")
                final_response = await self._generate_final_response(state)
                return {"response": final_response}
            
            # Format past steps for the replanner
            past_steps_str = "\n".join([
                f"Step: {step}\nResult: {result}" 
                for step, result in past_steps
            ])
            
            # 🔥 빈 past_steps에 대한 처리 추가
            if not past_steps_str.strip():
                print("⚠️ No past steps to evaluate, generating response with current information")
                final_response = await self._generate_final_response(state)
                return {"response": final_response}
            
            # Create replanner input
            replan_input = {
                "input": state["input"],
                "plan": "\n".join([f"{i+1}. {step}" for i, step in enumerate(state["plan"])]),
                "past_steps": past_steps_str
            }
            
            output = await self.replanner.ainvoke(replan_input)
            
            if isinstance(output.action, Response):
                print("✅ Ready to provide final response")
                return {"response": output.action.response}
            else:
                # 🔥 재계획 시에도 제한 조건 추가
                new_steps = output.action.steps
                if len(new_steps) > 3:  # 너무 많은 단계 방지
                    new_steps = new_steps[:3]
                    
                print(f"🔄 Replanning with {len(new_steps)} new steps")
                return {
                    "plan": new_steps,
                    "current_step_index": 0
                }
                
        except Exception as e:
            print(f"❌ Error in replanning: {e}")
            # If replanning fails, try to generate response with current information
            try:
                final_response = await self._generate_final_response(state)
                return {"response": final_response}
            except Exception as final_error:
                print(f"❌ Error generating final response: {final_error}")
                return {"response": f"Error processing question: {str(e)}"}
    
    async def _generate_final_response(self, state: PlanExecuteState) -> str:
        """Generate final response using all gathered information"""
        try:
            # Combine all past steps
            step_summaries = []
            for step, result in state.get("past_steps", []):
                step_summaries.append(f"Step: {step}\nResult: {result}")
            
            combined_steps = "\n\n".join(step_summaries)
            combined_context = state.get("combined_context", "")
            
            # 🔥 async/await 제거하고 sync 호출로 변경
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
                        "content": f"""Original Question: {state['input']}

Investigation Summary:
{combined_steps}

Detailed Context:
{combined_context}

Provide a comprehensive final answer to the original question."""
                    }
                ],
                max_tokens=2000,
                temperature=0.2,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating final response: {e}")
            return f"Based on the investigation steps, unable to generate complete answer due to error: {str(e)}"

    async def process_query(
        self,
        query: str,
        retrieval_type: str = None,
        hybrid_weights: List[float] = None,
        config_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process a complex query using plan-and-execute approach"""
        
        if config_dict is None:
            config_dict = {"recursion_limit": 8}  # 🔥 recursion_limit 증가
        
        initial_state = {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": "",
            "current_step_index": 0,
            "retrieval_type": retrieval_type,
            "hybrid_weights": hybrid_weights or [0.5, 0.5],
            "all_context_docs": [],
            "combined_context": ""
        }
        
        print(f"🚀 Starting LangGraph Plan and Execute for: {query}")
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config=config_dict)
            
            # 성공적으로 완료된 경우에만 response 사용
            if final_state.get("response"):
                final_answer = final_state["response"]
            else:
                # response가 없으면 past_steps 기반으로 답변 생성
                final_answer = await self._generate_final_response(final_state)
            
            return {
                "original_query": query,
                "plan": final_state.get("plan", []),
                "executed_steps": final_state.get("past_steps", []),
                "final_answer": final_answer,
                "combined_context": final_state.get("combined_context", ""),
                "all_context_docs": final_state.get("all_context_docs", []),
                "total_steps": len(final_state.get("past_steps", []))
            }
        
        except Exception as e:
            print(f"❌ Error in plan and execute: {e}")
            # 폴백: 단순 검색으로 답변 시도
            try:
                fallback_docs = retrievers.vectordb_retrieve(query)
                if fallback_docs:
                    fallback_context = "\n\n".join([doc.page_content for doc in fallback_docs])
                    # 🔥 async 제거
                    fallback_answer = utils.generate_llm_answer(query, fallback_context)
                else:
                    fallback_answer = f"Unable to process complex query due to error: {str(e)}"
                    fallback_context = ""
                    
                return {
                    "original_query": query,
                    "plan": [f"Fallback: Simple retrieval for '{query}'"],
                    "executed_steps": [("Fallback retrieval", "Used simple search due to plan-execute error")],
                    "final_answer": fallback_answer,
                    "combined_context": fallback_context,
                    "all_context_docs": fallback_docs if 'fallback_docs' in locals() else [],
                    "total_steps": 1
                }
            except Exception as fallback_error:
                return {
                    "original_query": query,
                    "plan": [],
                    "executed_steps": [],
                    "final_answer": f"Critical error: {str(e)}. Fallback also failed: {str(fallback_error)}",
                    "combined_context": "",
                    "all_context_docs": [],
                    "total_steps": 0
                }

# plan_execute_langgraph.py 수정 - 동기식 래퍼 개선
def process_complex_query_with_langgraph_plan_execute(
    original_query: str,
    retrieval_type: str = None,
    hybrid_weights: List[float] = None,
    max_steps: int = 5
) -> Dict[str, Any]:
    """동기식 래퍼 - 이벤트 루프 처리 개선"""
    import asyncio
    
    async def _async_process():
        agent = PlanExecuteLangGraph()
        config_dict = {"recursion_limit": max_steps}
        
        return await agent.process_query(
            original_query, 
            retrieval_type, 
            hybrid_weights, 
            config_dict
        )
    
    # 이벤트 루프 처리 개선
    try:
        # 현재 실행 중인 루프가 있는지 확인
        current_loop = asyncio.get_running_loop()
        # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
        import concurrent.futures
        import threading
        
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_async_process())
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
            
    except RuntimeError:
        # 실행 중인 루프가 없으면 직접 실행
        return asyncio.run(_async_process())