import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
from typing import Dict, Any, List, Tuple
import asyncio
import traceback
import tempfile
import shutil
import uuid
from datetime import datetime
import concurrent.futures

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
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {"pdf": None, "images": []}
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None


def save_uploaded_files(uploaded_files, file_type="image"):
    """업로드된 파일들을 임시 디렉토리에 저장"""
    if not uploaded_files:
        return None
    
    # 임시 디렉토리 생성
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="streamlit_rag_")
    
    temp_dir = Path(st.session_state.temp_dir)
    
    if file_type == "pdf":
        # PDF 파일 하나만 처리
        pdf_file = uploaded_files
        pdf_path = temp_dir / pdf_file.name
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        return pdf_path
    
    elif file_type == "image":
        # 여러 이미지 파일 처리
        if len(uploaded_files) == 1:
            # 단일 이미지 파일
            image_file = uploaded_files[0]
            image_path = temp_dir / image_file.name
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
            return image_path
        else:
            # 여러 이미지 파일들을 하나의 디렉토리에 저장
            images_dir = temp_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            for i, image_file in enumerate(uploaded_files):
                image_path = images_dir / f"{i:03d}_{image_file.name}"
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
            
            return images_dir
    
    return None


def cleanup_temp_files():
    """임시 파일들 정리"""
    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
        try:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
            print("Temporary files cleaned up")
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")


def display_file_upload_section():
    """파일 업로드 섹션 표시"""
    st.markdown("### 📁 File Upload (Optional)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 PDF Document**")
        uploaded_pdf = st.file_uploader(
            "Upload a PDF file for analysis",
            type=['pdf'],
            key="pdf_uploader",
            help="Upload a PDF document to extract text and analyze with your question"
        )
        
        if uploaded_pdf:
            st.success(f"✅ PDF uploaded: {uploaded_pdf.name}")
            st.session_state.uploaded_files["pdf"] = uploaded_pdf
        else:
            st.session_state.uploaded_files["pdf"] = None
    
    with col2:
        st.markdown("**🖼️ Images**")
        uploaded_images = st.file_uploader(
            "Upload image(s) for analysis",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="image_uploader",
            help="Upload one or more images to extract text and analyze with your question"
        )
        
        if uploaded_images:
            st.success(f"✅ {len(uploaded_images)} image(s) uploaded")
            for img in uploaded_images:
                st.caption(f"📸 {img.name}")
            st.session_state.uploaded_files["images"] = uploaded_images
        else:
            st.session_state.uploaded_files["images"] = []
    
    # 파일 처리 상태 표시
    if st.session_state.uploaded_files["pdf"] or st.session_state.uploaded_files["images"]:
        st.info("💡 Files will be processed together with your question when submitted.")
        
        # 파일 미리보기 옵션
        with st.expander("🔍 File Preview", expanded=False):
            if st.session_state.uploaded_files["pdf"]:
                st.markdown("**PDF File:**")
                st.text(f"📄 {st.session_state.uploaded_files['pdf'].name}")
                st.text(f"📏 Size: {st.session_state.uploaded_files['pdf'].size:,} bytes")
            
            if st.session_state.uploaded_files["images"]:
                st.markdown("**Image Files:**")
                for i, img in enumerate(st.session_state.uploaded_files["images"], 1):
                    col_img1, col_img2 = st.columns([3, 1])
                    with col_img1:
                        st.image(img, caption=f"{i}. {img.name}", width=200)
                    with col_img2:
                        st.text(f"📏 Size: {img.size:,} bytes")


async def process_query_with_files(query: str, pdf_file=None, image_files=None):
    """파일과 함께 쿼리 처리 - PDF 에러 처리 강화"""
    
    # 파일 저장 (백그라운드에서 처리)
    pdf_path = None
    img_path = None
    
    try:
        if pdf_file:
            print(f"📄 Processing PDF: {pdf_file.name}")
            pdf_path = save_uploaded_files(pdf_file, "pdf")
            print(f"📁 PDF saved to: {pdf_path}")
            
            # PDF 처리 가능성 사전 체크
            try:
                # Poppler 설치 확인
                import subprocess
                result = subprocess.run(['pdftoppm', '-h'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    raise FileNotFoundError("Poppler not accessible")
                    
                print("✅ Poppler is available for PDF processing")
                
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as poppler_error:
                error_msg = """
❌ **PDF Processing Error: Poppler Not Found**

PDF processing requires Poppler to be installed. Please follow these steps:

**Quick Fix:**
1. Download the installer: `install_poppler.py` (in project root)
2. Run: `python install_poppler.py`
3. Restart this application

**Manual Installation:**
1. Visit: https://github.com/oschwartz10612/poppler-windows/releases
2. Download the latest release ZIP
3. Extract to a folder (e.g., `C:\\poppler`)
4. Add `poppler/bin` to your system PATH
5. Restart this application

**Test Installation:**
Open command prompt and run: `pdftoppm -h`

**Alternative (if using conda):**
```bash
conda install -c conda-forge poppler
```
                """
                st.error(error_msg)
                return None
        
        if image_files:
            img_path = save_uploaded_files(image_files, "image")
            print(f"🖼️ Images saved to: {img_path}")
        
        # 🔥 단일 LangGraph 실행으로 통합
        return await execute_langgraph_once_with_streaming(
            query, pdf_path, img_path
        )
                
    except Exception as e:
        # PDF 관련 에러는 더 자세한 정보 제공
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            st.error("❌ **PDF Processing Error**")
            st.error("Poppler is required for PDF processing but not found.")
            
            with st.expander("🔧 **How to Fix This Error**", expanded=True):
                st.markdown("""
                **Method 1: Quick Install**
                1. Download `install_poppler.py` from the project root
                2. Run in terminal: `python install_poppler.py`
                3. Restart this application
                
                **Method 2: Manual Install**
                1. Go to: https://github.com/oschwartz10612/poppler-windows/releases
                2. Download the latest ZIP file
                3. Extract to `C:\\poppler`
                4. Add `C:\\poppler\\bin` to system PATH
                5. Restart this application
                
                **Method 3: Using Conda**
                ```bash
                conda install -c conda-forge poppler
                ```
                
                **Test Installation:**
                Open command prompt and run: `pdftoppm -h`
                """)
        else:
            st.error(f"❌ Error processing query with files: {str(e)}")
        
        print(f"❌ Full error traceback:")
        print(traceback.format_exc())
        return None


def debug_graph_execution(graph, init_state: GraphState):
    """그래프 실행 디버깅을 위한 함수"""
    print("🔍 Debug: Testing graph structure...")
    
    try:
        # 그래프 구조 확인
        graph_dict = graph.get_graph()
        print(f"   Graph nodes: {list(graph_dict.nodes.keys())}")
        print(f"   Graph edges: {[(e.source, e.target) for e in graph_dict.edges]}")
        
        # 단순 invoke 테스트
        print("🔍 Debug: Testing simple invoke...")
        test_result = graph.invoke(init_state)
        print(f"   Invoke result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'non-dict'}")
        print(f"   Invoke answer: {test_result.get('answer', 'NO ANSWER')[:100]}...")
        
        return test_result
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        return None


async def execute_langgraph_once_with_streaming(
    query: str, 
    pdf_path=None, 
    img_path=None
) -> Dict[str, Any]:
    """
    🔥 핵심 수정: LangGraph를 한 번만 실행하여 중간 결과와 최종 결과를 모두 수집
    """
    print("🚀 Starting unified LangGraph execution...")
    
    try:
        # 그래프 빌드
        graph = build_graph(
            pdf_path=pdf_path,
            img_path=img_path,
            retrieval_type=config.RETRIEVAL_TYPE,
            hybrid_weights=[config.HYBRID_WEIGHT, 1 - config.HYBRID_WEIGHT]
        )
        
        # 초기 상태 설정
        init_state: GraphState = {"question": [query], "messages": [("user", query)]}
        
        # 🔥 디버깅용 그래프 테스트 (개발 중에만 사용)
        if st.session_state.get("debug_mode", False):
            debug_result = debug_graph_execution(graph, init_state)
        
        # 🔥 중간 결과 수집을 위한 변수들
        intermediate_results = []
        final_state = None
        accumulated_state = {}  # 🔥 상태 누적용
        
        # 🔥 스트리밍 실행하면서 중간 결과 수집
        print("📡 Starting streaming execution...")
        node_count = 0
        
        async for event in graph.astream(init_state):
            print(f"🔍 Event {node_count + 1}: {list(event.keys())}")
            
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    # 최종 상태는 누적된 상태를 사용
                    final_state = accumulated_state.copy()
                    print(f"✅ Final state captured with keys: {list(final_state.keys())}")
                    continue
                
                # 🔥 상태 누적 (각 노드의 출력을 누적)
                if isinstance(node_output, dict):
                    accumulated_state.update(node_output)
                    print(f"   Node '{node_name}' added keys: {list(node_output.keys())}")
                    print(f"   Accumulated keys: {list(accumulated_state.keys())}")
                
                # 중간 결과 저장
                intermediate_results.append({
                    "node_name": node_name,
                    "node_output": node_output,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # 실시간 UI 업데이트
                await display_node_execution_unified(node_name, node_output, query)
                
                # 사용자가 읽을 수 있도록 잠시 대기
                await asyncio.sleep(1)
                node_count += 1
        
        # 🔥 최종 상태 확인 및 처리
        if final_state is None or not final_state:
            print("⚠️ No proper final state, using accumulated state...")
            final_state = accumulated_state
        
        # 🔥 필수 키가 없으면 기본값 설정
        essential_keys = ["answer", "context", "executed_steps", "plan", "combined_context", "explanation"]
        for key in essential_keys:
            if key not in final_state:
                if key == "answer":
                    final_state[key] = "Processing completed but no final answer generated"
                elif key in ["context", "executed_steps", "plan"]:
                    final_state[key] = []
                else:
                    final_state[key] = ""
        
        print(f"🔍 Final state summary:")
        print(f"   Answer: {final_state.get('answer', 'None')[:100]}...")
        print(f"   Context docs: {len(final_state.get('context', []))}")
        print(f"   Executed steps: {len(final_state.get('executed_steps', []))}")
        print(f"   Plan: {len(final_state.get('plan', []))}")
        
        # 🔥 수집된 결과를 정리하여 반환
        processed_result = {
            "answer": final_state.get("answer", "No answer generated"),
            "context": final_state.get("context", []),
            "executed_steps": final_state.get("executed_steps", []),
            "plan": final_state.get("plan", []),
            "combined_context": final_state.get("combined_context", ""),
            "explanation": final_state.get("explanation", ""),
            "intermediate_results": intermediate_results,
            "node_count": len(intermediate_results),
            "complexity": "Complex" if final_state.get("executed_steps") else "Simple"
        }
        
        print(f"✅ Unified execution completed: {len(intermediate_results)} intermediate steps")
        return processed_result
        
    except Exception as e:
        print(f"❌ Error in unified LangGraph execution: {e}")
        print(traceback.format_exc())
        return {
            "answer": f"Error in processing: {str(e)}",
            "context": [],
            "executed_steps": [],
            "plan": [],
            "combined_context": "",
            "explanation": f"Error: {str(e)}",
            "intermediate_results": [],
            "node_count": 0,
            "complexity": "Error"
        }


async def process_query_with_langgraph_streaming(query: str):
    """🔥 통합된 LangGraph 실행 - 파일 없는 경우"""
    return await execute_langgraph_once_with_streaming(query, None, None)


async def display_node_execution_unified(node_name: str, node_output: Dict[str, Any], query: str):
    """🔥 수정된 노드 실행 표시 - 최종 결과 표시 없이 중간 단계만"""
    
    if node_name == "complexity_check":
        # 복잡도 체크 결과 표시
        complexity = node_output.get("next", "unknown")
        st.markdown("### 🔍 Step 1: Analyzing Question Complexity")
        display_complexity_result(complexity)
        
        if complexity == "simple":
            st.info("🔄 Proceeding with simple retrieval pipeline...")
        else:
            st.info("🧠 Proceeding with complex Plan-and-Execute pipeline...")
            
    elif node_name in ["retrieve_simple", "extract_file_content"]:
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
        
        # 컨텍스트는 최종 결과에서만 표시하도록 생략
        st.success("✅ Complex processing completed")


def display_final_results_unified(processed_result: Dict[str, Any], query: str):
    """🔥 통합된 최종 결과 표시 - 이미 수집된 결과 사용"""
    st.markdown("---")
    st.markdown("### 🎉 Final Results")
    
    # 최종 답변
    answer = processed_result.get("answer", "No answer generated")
    st.markdown("#### 💡 Answer:")
    st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
    
    # 처리 경로 요약
    explanation = processed_result.get("explanation", "")
    if explanation:
        st.markdown("#### 📝 Processing Summary:")
        st.info(explanation)
    
    # 통계 정보
    col1, col2, col3 = st.columns(3)
    
    with col1:
        context_count = len(processed_result.get("context", []))
        st.metric("📚 Documents Retrieved", context_count)
    
    with col2:
        # Plan-Execute 정보
        complexity = processed_result.get("complexity", "Unknown")
        if complexity == "Complex":
            executed_steps = processed_result.get("executed_steps", [])
            st.metric("🔄 Steps Executed", len(executed_steps))
        else:
            st.metric("🔍 Processing Type", "Simple Retrieval")
    
    with col3:
        node_count = processed_result.get("node_count", 0)
        st.metric("⚙️ Graph Nodes", node_count)
    
    # 🔥 백그라운드 자동 저장 (기존 함수 재사용)
    try:
        # processed_result를 final_state 형식으로 변환
        final_state_for_save = {
            "answer": processed_result.get("answer", ""),
            "context": processed_result.get("context", []),
            "executed_steps": processed_result.get("executed_steps", []),
            "plan": processed_result.get("plan", []),
            "combined_context": processed_result.get("combined_context", ""),
            "explanation": processed_result.get("explanation", ""),
        }
        
        # Graph State 저장
        saved_state_file = save_graph_state_to_output(final_state_for_save, query)
        if saved_state_file:
            print(f"✅ Graph state auto-saved to: {saved_state_file}")
        
        # Graph 시각화 저장
        saved_graph_files = save_graph_visualizations(final_state_for_save, query)
        if saved_graph_files:
            print(f"✅ Graph visualizations auto-saved:")
            for graph_file in saved_graph_files:
                print(f"   📈 {Path(graph_file).name}")
        
    except Exception as e:
        print(f"❌ Error in background auto-save: {e}")


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


def save_graph_state_to_output(final_state: Dict[str, Any], query: str) -> str:
    """최종 GraphState를 output 디렉토리에 저장"""
    try:
        # 출력 디렉토리 확인
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # GraphState 형식에 맞춰 데이터 구성
        graph_state_dict = {
            "question": [query],
            "explanation": final_state.get("explanation", ""),
            "context": final_state.get("context", []),
            "filtered_context": final_state.get("filtered_context", []),
            "examples": final_state.get("examples", ""),
            "answer": final_state.get("answer", ""),
            "messages": final_state.get("messages", []),
            "scores": final_state.get("scores", []),
            "filtered_scores": final_state.get("filtered_scores", []),
            "subquestions": final_state.get("subquestions", []),
            "subquestion_results": final_state.get("subquestion_results", []),
            "combined_context": final_state.get("combined_context", ""),
            "plan": final_state.get("plan", []),
            "executed_steps": final_state.get("executed_steps", []),
            "retrieval_type": config.RETRIEVAL_TYPE,
            "hybrid_weights": [config.HYBRID_WEIGHT, 1 - config.HYBRID_WEIGHT],
            "next": final_state.get("next", "")
        }
        
        # 파일명 생성 (타임스탬프 + UUID)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"graph_state_{timestamp}_{unique_id}.json"
        file_path = output_dir / filename
        
        # JSON으로 저장 (직렬화 가능한 형태로 변환)
        serializable_state = convert_to_serializable(graph_state_dict)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Graph state saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        print(f"❌ Error saving graph state: {e}")
        return ""


def convert_to_serializable(obj):
    """객체를 JSON 직렬화 가능한 형태로 변환"""
    if hasattr(obj, 'page_content'):  # Document 객체
        return {
            "page_content": obj.page_content,
            "metadata": getattr(obj, 'metadata', {})
        }
    elif hasattr(obj, 'content'):  # Message 객체
        return obj.content
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_graph_visualizations(final_state: Dict[str, Any], query: str) -> List[str]:
    """두 가지 그래프를 시각화하여 저장"""
    saved_files = []
    
    try:
        # 출력 디렉토리 확인
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Main Graph (graph_builder.py에서 생성된 그래프) 시각화
        try:
            print("📊 Generating main graph visualization...")
            
            # 동일한 설정으로 그래프 재생성
            pdf_file = st.session_state.uploaded_files.get("pdf")
            image_files = st.session_state.uploaded_files.get("images", [])
            
            pdf_path = None
            img_path = None
            if pdf_file:
                pdf_path = save_uploaded_files(pdf_file, "pdf")
            if image_files:
                img_path = save_uploaded_files(image_files, "image")
            
            main_graph = build_graph(
                pdf_path=pdf_path,
                img_path=img_path,
                retrieval_type=config.RETRIEVAL_TYPE,
                hybrid_weights=[config.HYBRID_WEIGHT, 1 - config.HYBRID_WEIGHT]
            )
            
            main_graph_path = output_dir / f"main_graph_{timestamp}.png"
            main_graph.get_graph().draw_mermaid_png(output_file_path=str(main_graph_path))
            
            saved_files.append(str(main_graph_path))
            print(f"   ✅ Main graph saved to: {main_graph_path}")
            
        except Exception as e:
            print(f"   ❌ Error saving main graph: {e}")
        
        # 2. Plan-Execute Graph 시각화 (복잡한 질문인 경우만)
        if final_state.get("executed_steps") or final_state.get("plan"):
            try:
                print("📊 Generating plan-execute graph visualization...")
                
                from rag_pipeline.plan_execute_langgraph import PlanExecuteLangGraph
                plan_execute_agent = PlanExecuteLangGraph()
                
                plan_execute_graph_path = output_dir / f"plan_execute_graph_{timestamp}.png"
                plan_execute_agent.visualize_graph(output_path=plan_execute_graph_path)
                
                saved_files.append(str(plan_execute_graph_path))
                print(f"   ✅ Plan-execute graph saved to: {plan_execute_graph_path}")
                
            except Exception as e:
                print(f"   ❌ Error saving plan-execute graph: {e}")
        
        return saved_files
        
    except Exception as e:
        print(f"❌ Error in graph visualization: {e}")
        return []


def display_final_results(final_state: Dict[str, Any], query: str):
    """기존의 최종 결과 표시 함수 - 백업용"""
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


# 🔥 main 함수에서 response_content 정의 문제 수정
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
        
        # 🔥 디버그 모드 토글 추가
        debug_mode = st.checkbox("🐛 Debug Mode", value=False, help="Enable detailed debugging logs")
        st.session_state["debug_mode"] = debug_mode
        
        st.markdown("### 📊 Graph Flow")
        st.markdown("""
        **LangGraph Execution:**
        1. 🔍 Complexity Check
        2a. 📚 Simple: Retrieve → Filter → Answer
        2b. 🧠 Complex: Plan-and-Execute
        3. 💡 Final Answer
        """)
        
        st.markdown("### 📁 File Processing")
        st.markdown("""
        **Supported Files:**
        - 📄 PDF documents
        - 🖼️ Images (PNG, JPG, etc.)
        
        **Processing Flow:**
        1. File Upload
        2. Text Extraction (OCR/PDF parsing)
        3. Content Integration
        4. Query Processing
        """)
        
        # 🔥 자동 저장 정보 추가
        st.markdown("### 💾 Auto-Save")
        st.markdown("""
        **Automatic Background Saving:**
        - 📄 Graph State (JSON)
        - 📊 Graph Visualizations (PNG/TXT)
        - 📁 Output Directory: `./output/`
        """)
        
        if st.button("🔄 Reset Session"):
            cleanup_temp_files()
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # 파일 업로드 섹션
    display_file_upload_section()
    
    st.markdown("---")
    
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
            # 파일 처리 상태 확인 (백그라운드에서)
            pdf_file = st.session_state.uploaded_files.get("pdf")
            image_files = st.session_state.uploaded_files.get("images", [])
            
            # 🔥 response_content 초기화
            response_content = "**Error:** Initialization failed"
            
            # 🔥 통합된 처리 실행
            try:
                # 비동기 처리를 위한 래퍼
                try:
                    current_loop = asyncio.get_running_loop()
                    # 이미 루프가 실행 중이면 새 스레드에서 실행
                    
                    def run_async_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            if pdf_file or image_files:
                                return new_loop.run_until_complete(
                                    process_query_with_files(prompt, pdf_file, image_files)
                                )
                            else:
                                return new_loop.run_until_complete(
                                    process_query_with_langgraph_streaming(prompt)
                                )
                        except Exception as thread_error:
                            print(f"❌ Error in thread execution: {thread_error}")
                            print(traceback.format_exc())
                            raise thread_error
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        result = future.result()
                        
                except RuntimeError:
                    # 실행 중인 루프가 없으면 직접 실행
                    try:
                        if pdf_file or image_files:
                            result = asyncio.run(process_query_with_files(prompt, pdf_file, image_files))
                        else:
                            result = asyncio.run(process_query_with_langgraph_streaming(prompt))
                    except Exception as direct_error:
                        print(f"❌ Error in direct execution: {direct_error}")
                        print(traceback.format_exc())
                        raise direct_error
                
                # 🔥 결과 처리 및 표시 - 통합된 방식 사용
                if result:
                    answer = result.get("answer", "No answer generated")
                    complexity = result.get("complexity", "Unknown")
                    
                    # 파일 처리 정보는 간단하게만 표시 (선택적)
                    file_info = ""
                    if pdf_file or image_files:
                        file_count = (1 if pdf_file else 0) + len(image_files)
                        file_info = f" (with {file_count} uploaded file{'s' if file_count > 1 else ''})"
                    
                    response_content = f"**Question Complexity:** {complexity}{file_info}\n\n**Answer:** {answer}"
                    
                    # 🔥 통합된 최종 결과 표시
                    display_final_results_unified(result, prompt)
                    
                    # 파일 업로드 상태 자동 정리
                    if pdf_file or image_files:
                        st.session_state.uploaded_files = {"pdf": None, "images": []}
                        
                else:
                    response_content = "**Error:** Failed to process question"
                    
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                st.error(error_msg)
                print(f"❌ {error_msg}")
                print(f"❌ Full traceback:")
                print(traceback.format_exc())
                response_content = f"**Error:** {str(e)}"
            
            # 🔥 저장 관련 UI 완전 제거 - 대신 간단한 상태 표시만
            st.markdown("---")
            st.markdown("#### ℹ️ Session Info")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                if st.button("🔄 Refresh", help="Refresh the interface"):
                    st.rerun()
            
            with col_info2:
                output_dir = Path(config.OUTPUT_DIR)
                if output_dir.exists():
                    file_count = len(list(output_dir.glob("*")))
                    st.metric("📁 Output Files", file_count, help=f"Files in {output_dir}")
                else:
                    st.metric("📁 Output Files", 0)
            
            with col_info3:
                if st.button("🗑️ Clear Files", help="Clear uploaded files"):
                    cleanup_temp_files()
                    st.session_state.uploaded_files = {"pdf": None, "images": []}
                    st.rerun()
        
        # 어시스턴트 메시지 저장
        st.session_state.messages.append({"role": "assistant", "content": response_content})


# 🔥 메인 함수 호출 추가 - 이 부분이 누락되어 있었음!
if __name__ == "__main__":
    main()
