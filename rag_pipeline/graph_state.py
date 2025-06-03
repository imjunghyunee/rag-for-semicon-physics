from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    question: Annotated[List[str], add_messages]
    explanation: Annotated[str, "Explanation"]
    context: Annotated[str, "Context"]
    filtered_context: Annotated[str, "Filtered_Context"]
    examples: Annotated[str, "Examples"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    scores: Annotated[List[float], "Scores"]
    filtered_scores: Annotated[List[float], "Filtered Scores"]
    # Query decomposition related states (legacy)
    subquestions: Annotated[List[str], "Sub-questions for complex queries"]
    subquestion_results: Annotated[List[Dict[str, Any]], "Results from sub-question processing"]
    combined_context: Annotated[str, "Combined context from all sub-questions"]
    # Plan and Execute related states
    plan: Annotated[List[str], "Execution plan steps for complex queries"]  # 🔥 수정
    executed_steps: Annotated[List[tuple], "Results from executed steps as (step, result) tuples"]  # 🔥 수정
    retrieval_type: Annotated[str, "Type of retrieval (hyde, summary, etc.)"]
    hybrid_weights: Annotated[List[float], "Hybrid retrieval weights"]
    next: Annotated[str, "Next node to execute"]
