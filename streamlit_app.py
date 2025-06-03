import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
from typing import Dict, Any, List, Tuple
import asyncio
import traceback

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from rag_pipeline.graph_builder import build_graph
from rag_pipeline.graph_state import GraphState
from rag_pipeline import config, utils
from rag_pipeline.plan_execute_langgraph import PlanExecuteLangGraph

# 페이지 설정
st.set_page_config(
    page_title="Semiconductor Physics RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .step-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .plan-step {
        background-color: #e3f2fd;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 4px solid #2196f3;
        border-radius: 4px;
    }
    .context-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        max-height: 300px;
        overflow-y: auto;
    }
    .result-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .complexity-simple {
        color: #28a745;
        font-weight: bold;
    }
    .complexity-complex {
        color: #dc3545;
        font-weight: bold;
    }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "complexity" not in st.session_state:
        st.session_state.complexity = None
    if "plan_results" not in st.session_state:
        st.session_state.plan_results = []
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = None


def display_complexity_result(complexity: str):
    """복잡도 판정 결과 표시"""
    if complexity == "simple":
        st.markdown(
            f'<div class="complexity-simple">📝 Question Complexity: SIMPLE</div>',
            unsafe_allow_html=True
        )
        st.info("This question will be processed using direct retrieval.")
    else:
        st.markdown(
            f'<div class="complexity-complex">🧠 Question Complexity: COMPLEX</div>',
            unsafe_allow_html=True
        )
        st.info("This question requires multi-step reasoning and will be processed using Plan-and-Execute.")


def display_simple_processing(query: str):
    """Simple 질문 처리 과정 표시"""
    st.markdown("### 🔍 Processing Simple Query")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 그래프 구성
        status_text.text("Building retrieval pipeline...")
        progress_bar.progress(20)
        
        graph = build_graph(
            pdf_path=None,
            img_path=None,
            retrieval_type=config.RETRIEVAL_TYPE,
            hybrid_weights=[config.HYBRID_WEIGHT, 1 - config.HYBRID_WEIGHT]
        )
        
        # 질문 처리
        status_text.text("Processing query...")
        progress_bar.progress(50)
        
        init_state: GraphState = {"question": [query], "messages": [("user", query)]}
        final_state = graph.invoke(init_state)
        
        progress_bar.progress(80)
        status_text.text("Generating final answer...")
        
        # 결과 표시
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(0.5)
        
        # 프로그레스 바 정리
        progress_bar.empty()
        status_text.empty()
        
        # 컨텍스트 표시
        if "context" in final_state and final_state["context"]:
            with st.expander("📚 Retrieved Context", expanded=False):
                contexts = final_state["context"]
                for i, doc in enumerate(contexts[:3], 1):  # 처음 3개만 표시
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    st.markdown(f"**Context {i}:**")
                    st.markdown(f'<div class="context-box">{content[:500]}...</div>', 
                              unsafe_allow_html=True)
        
        # 최종 답변 표시
        st.markdown("### 💡 Final Answer")
        answer = final_state.get("answer", "No answer generated")
        st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
        
        return final_state
        
    except Exception as e:
        st.error(f"Error processing simple query: {str(e)}")
        st.error(traceback.format_exc())
        return None


async def process_complex_query_step_by_step(query: str):
    """Complex 질문을 단계별로 처리하고 실시간 업데이트"""
    
    # Plan-and-Execute 에이전트 생성
    agent = PlanExecuteLangGraph()
    
    # 초기 상태 설정
    initial_state = {
        "input": query,
        "plan": [],
        "past_steps": [],
        "response": "",
        "current_step_index": 0,
        "retrieval_type": config.RETRIEVAL_TYPE,
        "hybrid_weights": [config.HYBRID_WEIGHT, 1 - config.HYBRID_WEIGHT],
        "all_context_docs": [],
        "combined_context": ""
    }
    
    # 프로그레스 추적을 위한 컨테이너들
    step_container = st.empty()
    
    try:
        # Step 1: Planning
        with step_container.container():
            st.markdown("### 📋 Step 1: Creating Execution Plan")
            with st.spinner("Planning execution steps..."):
                plan_result = await agent.plan_step(initial_state)
                initial_state.update(plan_result)
            
            # 계획 표시
            if initial_state["plan"]:
                st.markdown("**Generated Plan:**")
                for i, step in enumerate(initial_state["plan"], 1):
                    st.markdown(f'<div class="plan-step">{i}. {step}</div>', 
                              unsafe_allow_html=True)
            
            time.sleep(2)  # 사용자가 계획을 읽을 시간
        
        # Step 2+: Execution Loop
        step_num = 2
        max_iterations = config.MAX_PLAN_STEPS + 2  # 안전 장치
        
        while step_num <= max_iterations:
            current_step_index = initial_state.get("current_step_index", 0)
            plan = initial_state.get("plan", [])
            
            # 실행할 단계가 있는지 확인
            if current_step_index >= len(plan):
                break
                
            # 현재 단계 표시
            with step_container.container():
                current_step = plan[current_step_index]
                st.markdown(f"### 🔄 Step {step_num}: Executing Plan")
                st.markdown(f"**Current Step:** {current_step}")
                
                with st.spinner(f"Executing: {current_step[:50]}..."):
                    # 단계 실행
                    execute_result = await agent.execute_step(initial_state)
                    initial_state.update(execute_result)
                
                # 실행 결과 표시
                past_steps = initial_state.get("past_steps", [])
                if past_steps:
                    latest_step, latest_result = past_steps[-1]
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Retrieved Context:**")
                        combined_context = initial_state.get("combined_context", "")
                        if combined_context:
                            # 가장 최근 단계의 컨텍스트만 표시
                            recent_context = combined_context.split("=== Step")[-1]
                            st.markdown(f'<div class="context-box">{recent_context[:400]}...</div>', 
                                      unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Step Result:**")
                        st.markdown(f'<div class="result-box">{latest_result[:300]}...</div>', 
                                  unsafe_allow_html=True)
                
                time.sleep(1.5)  # 결과를 읽을 시간
            
            # Replan 단계
            step_num += 1
            with step_container.container():
                st.markdown(f"### 🤔 Step {step_num}: Evaluating Progress")
                
                with st.spinner("Evaluating progress and replanning..."):
                    replan_result = await agent.replan_step(initial_state)
                    initial_state.update(replan_result)
                
                # 종료 조건 확인
                if "response" in replan_result and replan_result["response"]:
                    # 최종 답변 생성됨
                    break
                elif "plan" in replan_result:
                    # 새로운 계획 생성됨
                    st.markdown("**Updated Plan:**")
                    new_plan = replan_result["plan"]
                    for i, step in enumerate(new_plan, 1):
                        st.markdown(f'<div class="plan-step">{i}. {step}</div>', 
                                  unsafe_allow_html=True)
                
                time.sleep(1)
            
            step_num += 1
            
            # 무한 루프 방지
            if step_num > max_iterations:
                st.warning("Maximum iteration limit reached.")
                break
        
        # 최종 답변 표시
        step_container.empty()  # 이전 단계들 정리
        
        # 과정 요약을 expander로 표시
        with st.expander("📊 Process Summary", expanded=False):
            st.markdown("**Original Plan:**")
            for i, step in enumerate(initial_state.get("plan", []), 1):
                st.markdown(f"{i}. {step}")
            
            st.markdown("**Executed Steps:**")
            for i, (step, result) in enumerate(initial_state.get("past_steps", []), 1):
                st.markdown(f"**Step {i}:** {step}")
                st.markdown(f"*Result:* {result[:200]}...")
        
        # 최종 답변
        st.markdown("### 💡 Final Answer")
        final_response = initial_state.get("response", "")
        if not final_response:
            # response가 없으면 수동으로 생성
            final_response = await agent._generate_final_response(initial_state)
        
        st.markdown(f'<div class="result-box">{final_response}</div>', 
                  unsafe_allow_html=True)
        
        return initial_state
        
    except Exception as e:
        st.error(f"Error in complex query processing: {str(e)}")
        st.error(traceback.format_exc())
        return None


async def process_query_with_langgraph_streaming(query: str):
    """LangGraph의 실행 흐름을 실시간으로 추적하여 표시"""
    
    # LangGraph 빌드
    graph = build_graph(
        pdf_path=None,
        img_path=None,
        retrieval_type=config.RETRIEVAL_TYPE,
        hybrid_weights=[config.HYBRID_WEIGHT, 1 - config.HYBRID_WEIGHT]
    )
    
    # 초기 상태
    init_state: GraphState = {"question": [query], "messages": [("user", query)]}
    
    # 실시간 업데이트를 위한 컨테이너들
    main_container = st.empty()
    progress_container = st.empty()
    
    try:
        # LangGraph 스트리밍 실행
        async for event in graph.astream(init_state):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue
                    
                # 각 노드의 실행 결과를 실시간으로 표시
                with main_container.container():
                    await display_node_execution(node_name, node_output, query)
                    
                # 잠시 대기 (사용자가 읽을 수 있도록)
                await asyncio.sleep(1)
        
        # 최종 상태 가져오기
        final_state = graph.invoke(init_state)
        
        # 최종 결과 표시
        with main_container.container():
            display_final_results(final_state, query)
            
        return final_state
        
    except Exception as e:
        st.error(f"Error in LangGraph execution: {str(e)}")
        st.error(traceback.format_exc())
        return None


async def display_node_execution(node_name: str, node_output: Dict[str, Any], query: str):
    """각 노드의 실행 결과를 표시"""
    
    if node_name == "complexity_check":
        # 복잡도 체크 결과 표시
        complexity = node_output.get("next", "unknown")
        st.markdown("### 🔍 Step 1: Analyzing Question Complexity")
        display_complexity_result(complexity)
        
        if complexity == "simple":
            st.info("🔄 Proceeding with simple retrieval pipeline...")
        else:
            st.info("🧠 Proceeding with complex Plan-and-Execute pipeline...")
            
    elif node_name in ["retrieve_simple"]:
        # Simple 검색 단계
        st.markdown("### 📚 Step 2: Information Retrieval")
        with st.spinner("Searching relevant documents..."):
            await asyncio.sleep(0.5)  # 시각적 효과
        
        context = node_output.get("context", [])
        st.success(f"✅ Retrieved {len(context)} relevant documents")
        
        # 검색된 문서 미리보기
        if context:
            with st.expander("📖 Preview Retrieved Documents", expanded=False):
                for i, doc in enumerate(context[:2], 1):  # 처음 2개만
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    st.markdown(f"**Document {i}:**")
                    st.markdown(f'<div class="context-box">{content[:300]}...</div>', 
                              unsafe_allow_html=True)
    
    elif node_name == "relevance_check":
        # 관련성 체크 단계
        st.markdown("### 🎯 Step 3: Relevance Filtering")
        with st.spinner("Filtering relevant content..."):
            await asyncio.sleep(0.5)
        
        filtered_context = node_output.get("filtered_context", [])
        scores = node_output.get("scores", [])
        
        if scores:
            avg_score = sum(scores) / len(scores)
            st.success(f"✅ Filtered content (avg similarity: {avg_score:.3f})")
        else:
            st.success("✅ Content filtering completed")
    
    elif node_name == "llm_answer_simple":
        # Simple 답변 생성
        st.markdown("### 💡 Step 4: Generating Answer")
        with st.spinner("Generating final answer..."):
            await asyncio.sleep(1)
        
        answer = node_output.get("answer", "")
        if answer:
            st.success("✅ Answer generated successfully")
        
    elif node_name == "plan_and_execute":
        # Complex 처리 결과
        st.markdown("### 🧠 Complex Query Processing Results")
        
        # Plan 표시
        plan = node_output.get("plan", [])
        if plan:
            st.markdown("**📋 Execution Plan:**")
            for i, step in enumerate(plan, 1):
                st.markdown(f'<div class="plan-step">{i}. {step}</div>', 
                          unsafe_allow_html=True)
        
        # 실행된 단계들 표시
        executed_steps = node_output.get("executed_steps", [])
        if executed_steps:
            with st.expander(f"📊 Executed Steps ({len(executed_steps)} steps)", expanded=False):
                for i, (step, result) in enumerate(executed_steps, 1):
                    st.markdown(f"**Step {i}:** {step}")
                    st.markdown(f"*Result:* {result[:200]}...")
                    st.markdown("---")
        
        # 컨텍스트 표시
        combined_context = node_output.get("combined_context", "")
        if combined_context:
            with st.expander("📚 Retrieved Context", expanded=False):
                st.markdown(f'<div class="context-box">{combined_context[:800]}...</div>', 
                          unsafe_allow_html=True)


def display_final_results(final_state: Dict[str, Any], query: str):
    """최종 결과 표시"""
    st.markdown("---")
    st.markdown("### 🎉 Final Results")
    
    # 최종 답변
    answer = final_state.get("answer", "No answer generated")
    st.markdown("#### 💡 Answer:")
    st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
    
    # 처리 경로 요약
    explanation = final_state.get("explanation", "")
    if explanation:
        st.markdown("#### 📝 Processing Summary:")
        st.info(explanation)
    
    # 통계 정보
    col1, col2, col3 = st.columns(3)
    
    with col1:
        context_count = len(final_state.get("context", []))
        st.metric("📚 Documents Retrieved", context_count)
    
    with col2:
        # Plan-Execute 정보
        executed_steps = final_state.get("executed_steps", [])
        if executed_steps:
            st.metric("🔄 Steps Executed", len(executed_steps))
        else:
            st.metric("🔍 Processing Type", "Simple Retrieval")
    
    with col3:
        scores = final_state.get("scores", [])
        if scores and len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            st.metric("🎯 Avg Similarity", f"{avg_score:.3f}")


def main():
    """메인 Streamlit 앱"""
    initialize_session_state()
    
    # 헤더
    st.markdown('<div class="main-header">🔬 Semiconductor Physics RAG System</div>', 
              unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown(f"**Model:** {config.OPENAI_MODEL}")
        st.markdown(f"**Retrieval Type:** {config.RETRIEVAL_TYPE}")
        st.markdown(f"**TOP_K:** {config.TOP_K}")
        st.markdown(f"**Max Plan Steps:** {config.MAX_PLAN_STEPS}")
        
        st.markdown("### 📊 Graph Flow")
        st.markdown("""
        **LangGraph Execution:**
        1. 🔍 Complexity Check
        2a. 📚 Simple: Retrieve → Filter → Answer
        2b. 🧠 Complex: Plan-and-Execute
        3. 💡 Final Answer
        """)
        
        if st.button("🔄 Reset Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # 메인 채팅 인터페이스
    st.markdown("### 💬 Ask a Question")
    
    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("Enter your semiconductor physics question..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 어시스턴트 응답 시작
        with st.chat_message("assistant"):
            st.markdown("### 🚀 Processing Your Question")
            st.markdown(f"**Query:** {prompt}")
            
            # LangGraph 스트리밍 실행
            try:
                # 비동기 처리를 위한 래퍼
                try:
                    current_loop = asyncio.get_running_loop()
                    # 이미 루프가 실행 중이면 새 스레드에서 실행
                    import concurrent.futures
                    
                    def run_async_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                process_query_with_langgraph_streaming(prompt)
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        result = future.result()
                        
                except RuntimeError:
                    # 실행 중인 루프가 없으면 직접 실행
                    result = asyncio.run(process_query_with_langgraph_streaming(prompt))
                
                if result:
                    answer = result.get("answer", "No answer generated")
                    complexity = "Complex" if result.get("executed_steps") else "Simple"
                    response_content = f"**Question Complexity:** {complexity}\n\n**Answer:** {answer}"
                else:
                    response_content = "**Error:** Failed to process question"
                    
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                response_content = f"**Error:** {str(e)}"
        
        # 어시스턴트 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": response_content})


if __name__ == "__main__":
    main()
