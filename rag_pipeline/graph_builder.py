from __future__ import annotations
from pathlib import Path
from langgraph.graph import StateGraph
from rag_pipeline.graph_state import GraphState
from rag_pipeline import nodes
from typing import List
import nest_asyncio


def build_graph(
    pdf_path: Path | None = None,
    img_path: Path | None = None,
    retrieval_type: str | None = None,
    hybrid_weights: List[float] | None = None,
):
    g = StateGraph(GraphState)

    # Simple/Complex routing function
    def route_complexity(state: GraphState) -> str:
        """Route based on query complexity"""
        decision = state.get("next", "simple")
        return decision

    # Nodes
    # Add file content extraction node if file paths are provided
    if pdf_path or img_path:
        if pdf_path:
            g.add_node(
                "extract_file_content", 
                lambda s: nodes.node_retrieve_file_embedding(s, pdf_path)
            )
        elif img_path:
            g.add_node(
                "extract_file_content", 
                lambda s: nodes.node_retrieve_img_embedding(s, img_path)
            )
        
        g.add_node("complexity_check", nodes.node_simple_or_not)
    else:
        g.add_node("complexity_check", nodes.node_simple_or_not)

    # Simple query processing nodes
    if pdf_path or img_path:
        # For file-based queries, we already have the content from extract_file_content
        g.add_node("retrieve_simple", nodes.node_retrieve)
    elif retrieval_type == "hyde" and hybrid_weights:
        g.add_node("retrieve_simple", nodes.node_retrieve_hyde_hybrid)
    elif retrieval_type == "hyde":
        g.add_node("retrieve_simple", nodes.node_retrieve_hyde)
    elif retrieval_type == "summary" and hybrid_weights:
        g.add_node("retrieve_simple", nodes.node_retrieve_summary_hybrid)
    elif retrieval_type == "summary":
        g.add_node("retrieve_simple", nodes.node_retrieve_summary)
    elif hybrid_weights:
        g.add_node("retrieve_simple", nodes.node_retrieve_hybrid)
    else:
        g.add_node("retrieve_simple", nodes.node_retrieve)

    g.add_node("relevance_check", nodes.node_relevance_check)
    g.add_node("llm_answer_simple", nodes.node_llm_answer)

    # Complex query processing nodes - 단일 노드로 단순화
    g.add_node("plan_and_execute", nodes.node_plan_and_execute)

    # Entry point
    if pdf_path or img_path:
        g.set_entry_point("extract_file_content")
        # Connect file extraction to complexity check
        g.add_edge("extract_file_content", "complexity_check")
    else:
        g.set_entry_point("complexity_check")

    # Conditional edges based on complexity
    g.add_conditional_edges(
        "complexity_check",
        route_complexity,
        {
            "simple": "retrieve_simple",
            "complex": "plan_and_execute",
        },
    )

    # Simple query path
    g.add_edge("retrieve_simple", "relevance_check")
    g.add_edge("relevance_check", "llm_answer_simple")
    g.add_edge("llm_answer_simple", "__end__")

    # Complex query path - 🔥 바로 종료
    g.add_edge("plan_and_execute", "__end__")

    return g.compile()


def visualize_graph(graph: StateGraph, output_path: Path = Path("./graph.png"), return_image: bool = False):
    """Visualize the graph and save to output_path or return image bytes"""
    try:
        nest_asyncio.apply()
        
        # 🔥 콘솔 로그만 유지 (간소화)
        print(f"🔄 Background: Generating main graph -> {output_path.name}")
        
        if return_image:
            # 이미지 바이트 반환
            image_bytes = graph.get_graph().draw_mermaid_png()
            print(f"✅ Main graph generated (bytes)")
            return image_bytes
        else:
            # 파일로 저장
            graph.get_graph().draw_mermaid_png(output_file_path=str(output_path))
            print(f"✅ Main graph saved: {output_path.name}")
            return str(output_path)
            
    except Exception as e:
        print(f"❌ Main graph generation failed: {type(e).__name__}")
        
        # 🔥 에러 시에도 파일 경로 반환 (텍스트 설명)
        if output_path and not return_image:
            try:
                with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write("Main Graph Structure:\n")
                    f.write("START -> [complexity_check] -> [simple|complex]\n")
                    f.write("Simple path: retrieve_simple -> relevance_check -> llm_answer_simple -> END\n")
                    f.write("Complex path: plan_and_execute -> END\n\n")
                    f.write("Note: Visual graph generation failed, showing text description instead.\n")
                print(f"✅ Main graph description saved: {output_path.with_suffix('.txt').name}")
                return str(output_path.with_suffix('.txt'))
            except:
                pass
        
        return None
