from __future__ import annotations
import argparse, json, sys, ast
from pathlib import Path
from rag_pipeline.graph_builder import build_graph, visualize_graph
from rag_pipeline import config
from rag_pipeline.graph_state import GraphState
from langchain.schema import Document
from typing import Any, List, Dict
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# final_state에서 HumanMessage / AIMessage 객체를 문자열로 변환
def convert_to_string(value):
    if isinstance(value, (HumanMessage, AIMessage)):
        return value.content
    elif isinstance(value, Document):
        return value.page_content
    elif isinstance(value, list):
        return [convert_to_string(item) for item in value]
    elif isinstance(value, dict):
        return {key: convert_to_string(val) for key, val in value.items()}
    return value


def run(
    query: str,
    pdf_path: str | None = None,
    img_path: str | None = None,
):
    # 경로 검증 추가
    pdf_path_obj = None
    img_path_obj = None
    
    if pdf_path:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            print(f"Error: PDF file not found at {pdf_path_obj}")
            return
        if not pdf_path_obj.is_file():
            print(f"Error: PDF path is not a file: {pdf_path_obj}")
            return
    
    if img_path:
        img_path_obj = Path(img_path)
        if not img_path_obj.exists():
            print(f"Error: Image path not found at {img_path_obj}")
            return
        # 이미지는 파일 또는 디렉토리 모두 허용
        if not (img_path_obj.is_file() or img_path_obj.is_dir()):
            print(f"Error: Image path is neither a file nor directory: {img_path_obj}")
            return
        
        print(f"📁 Image input type: {'File' if img_path_obj.is_file() else 'Directory'}")
        print(f"📁 Image path: {img_path_obj}")

    graph = build_graph(
        pdf_path_obj,
        img_path_obj,
        config.RETRIEVAL_TYPE,
        config.HYBRID_WEIGHT,
    )
    init_state: GraphState = {"question": [query], "messages": [("user", query)]}
    final_state = graph.invoke(init_state)

    visualize_graph(graph)

    final_state_converted = convert_to_string(final_state)

    # for debugging
    print("\n===== 최종 답변 =====\n")
    print(final_state["answer"])
    print("\n===== 내부 상태 (디버그) =====\n")
    print(json.dumps(final_state_converted, indent=2, ensure_ascii=False))

    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save json file
    output_data = {
        "answer": final_state["answer"],
        "debug_state": final_state_converted,
    }

    output_filename = f"output_{uuid.uuid4()}.json"
    output_path = output_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved {output_path}! \n")

    # for eval - 평가용 직렬화된 상태 반환
    serializable = {
        "question": [
            m.content if hasattr(m, "content") else str(m)
            for m in final_state.get("question", [])
        ],
        "explanation": final_state.get("explanation", ""),
        "context": final_state.get("context", []),
        "answer": final_state.get("answer", ""),
        "score": final_state.get("scores", []),
        # Legacy Query decomposition 관련 정보
        "subquestions": final_state.get("subquestions", []),
        "subquestion_results": final_state.get("subquestion_results", []),
        "combined_context": final_state.get("combined_context", ""),
        # Plan and Execute 관련 정보 추가
        "plan": final_state.get("plan", {}),
        "executed_steps": final_state.get("executed_steps", []),
    }
    return serializable


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="question")
    p.add_argument("--pdf", help="pdf file path", default=None)
    p.add_argument("--img", help="image file path", default=None)
    # p.add_argument("--type", help="query type (hyde, summary)", default=None)
    # p.add_argument(
    #     "--hybrid", help="hybrid retriever weights [float1,float2]", default=None
    # )
    args = p.parse_args()

    # hybrid_weights = None
    # if args.hybrid:
    #     try:
    #         hybrid_weights = ast.literal_eval(args.hybrid)
    #         if not isinstance(hybrid_weights, list) or len(hybrid_weights) != 2:
    #             print(
    #                 "Warning: hybrid weights should be a list of two floats [float1,float2]"
    #             )
    #             hybrid_weights = None
    #     except (SyntaxError, ValueError):
    #         print(
    #             "Warning: Could not parse hybrid weights. Format should be [float1,float2]"
    #         )

    print(f"Query received: {args.query} (Type: {type(args.query)})")
    # if hybrid_weights:
    #     print(f"Using hybrid retrieval with weights: {hybrid_weights}")

    run(args.query, args.pdf, args.img)
