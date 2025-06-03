# rag-for-semicon-physics

1. pip install -r requirements.txt 실행

    -   만약 streamlit에서 의존성 문제가 발생한다면 requirements_streamlit.txt로도 다운
    -   그래도 streamlit이 안된다면 python fix_dependencies.py 실행
    -   만약 위 두 단계 없이 바로 streamlit과 python main.py가 실행된다면 위 두 파일 requirements_streamlit.txt과 fix_dependencies.py은 legacy파일로 옮겨도 무방.

2. 터미널에서 실행하는 명령어 : python main.py --query "인공지능 기술의 발전이 노동 시장에 미치는 영향을 분석하고, 이에 따른 윤리적 문제들과 정책적 대응 방안을 비교하여 설명하시오."

    -   쿼리는 예시

3. streamlit 실행 명령어:

    -   python run_streamlit.py
    -   streamlit run streamlit_app.py
        둘 다 가능

    -   streamlit 실행하면 화면이 뜰 때까지 시간이 꽤 소모. 터미널 창에서 어떤 RuntimeError가 두 개 뜰 떄까지 대기.
    -   RuntimeError가 뜨면 Streamlit 페이지 로딩 완료, RuntimeError는 무시해도 됨.(있어도 실행은 되더라구)

4. streamlit에서 쿼리넣고 실행시 총 두 번 langgraph 실행....ㅜ

    4-1) 첫 번째 모든 실행 마치면 executed step과 retrieved context를 출력 -> 바로 캡쳐, 두 번쨰 출력 나오면 지워짐.
   
    4-2) 두 번째 실행이 끝나면 최종 결과가 화면에 출력.

6. streamlit 실행 예시.

![Image](https://github.com/user-attachments/assets/12186f89-56ce-4d52-94d5-1f5a973a9b90)

![Image](https://github.com/user-attachments/assets/49749f8c-d467-44d4-8f3b-966b6aa526f5)

![Image](https://github.com/user-attachments/assets/9aaa7e53-8805-4d2f-8866-01a8c22d1ac7)

    -   📊 Executed Steps (5 steps) : 전체 step이 어떤 plan을 따라갔는지 확인 가능. 하지만 그 plan에 대한 step 결과가 어떤 것인지까지는 출력이 안 나옴.
    -   지금 확인해보니 📊 Executed Steps (5 steps), 📚 Retrieved Context 이 두 부분 toggle하면 더 자세한 내용 볼 수 있도록 하는게 의도였는데 모든 출력이 반영이 안되었음...
    -   그래서 저 두 부분은 굳이 열지 말고 plan과 마지막 출력 결과물만 보여주는거에 신경쓰면 될 듯.

6. Option

-   plan_execute_langgraph.py 파일 안에 서브 그래프 시각화 함수 구현은 되어 있음.
-   만약 시각화가 필요하다면 main.py 파일을 아래와 같이 변경 후

```python
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
  visualize: bool = False,  # 🔥 그래프 시각화 옵션 추가
):
  graph = build_graph(
      Path(pdf_path) if pdf_path else None,
      Path(img_path) if img_path else None,
      config.RETRIEVAL_TYPE,
      config.HYBRID_WEIGHT,
  )
  init_state: GraphState = {"question": [query], "messages": [("user", query)]}
  final_state = graph.invoke(init_state)

  if visualize:
      print("\n===== 그래프 시각화 =====")
      visualize_graph(graph)

      # Plan-Execute 서브그래프도 시각화 (복잡한 쿼리인 경우)
      if "plan" in final_state and final_state["plan"]:
          print("\n===== Plan-Execute 서브그래프 시각화 =====")
          try:
              from rag_pipeline.plan_execute_langgraph import PlanExecuteLangGraph
              plan_execute_agent = PlanExecuteLangGraph()

              output_path = Path(config.OUTPUT_DIR) / "plan_execute_subgraph.png"
              plan_execute_agent.visualize_graph(output_path=output_path)
              print(f"Plan-Execute 서브그래프 저장됨: {output_path}")

              # 그래프 정보 출력
              graph_info = plan_execute_agent.get_graph_info()
              print(f"서브그래프 노드 수: {graph_info['total_nodes']}")
              print(f"서브그래프 엣지 수: {graph_info['total_edges']}")

          except Exception as e:
              print(f"Plan-Execute 서브그래프 시각화 실패: {e}")

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
  p.add_argument("--visualize", action="store_true", help="generate graph visualizations")  # 🔥 시각화 옵션
  args = p.parse_args()

  print(f"Query received: {args.query} (Type: {type(args.query)})")
  if args.visualize:
      print("Graph visualization enabled")

  run(args.query, args.pdf, args.img, args.visualize)

```

-   python main.py --query "..." --visualize 이런 식으로 할 수 있었던 것 같다... (안해보긴 해서 꼭 필요한 경우에만 시도)
