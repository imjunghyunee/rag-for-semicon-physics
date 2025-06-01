from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    question: Annotated[List[str], add_messages]
    explanation: Annotated[str, "Explanation"]
    context: Annotated[str, "Context"]
    filtered_context: Annotated[str, "Filtered_Context"]
    examples: Annotated[str, "Examples"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    # relevance: Annotated[str, "Relevance"]
    scores: Annotated[List[float], "Scores"]
    filtered_scores: Annotated[List[float], "Filtered Scores"]
