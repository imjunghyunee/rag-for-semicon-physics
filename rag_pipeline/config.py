from pathlib import Path
import os


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val == "True"


# ----- 벡터 DB 및 임베딩 설정 -----
EMBED_MODEL_NAME: str = "jinaai/jina-embeddings-v3"
RERANKER_NAME: str = "BAAI/bge-reranker-v2-m3"
VECTOR_DB_PATH: Path = Path("./vectordb/jina/neamen_content")

REMOTE_LLM_URL: str = "http://localhost:8000/v1/chat/completions"
REMOTE_LLM_MODEL: str = "agent:llama-4-scout-17B-16E-instruct"

# ----- 검색 파라미터 -----
TOP_K: int = int(os.getenv("TOP_K", 1))
SIM_THRESHOLD: float = float(os.getenv("SIM_THRESHOLD", 0.01))
RERANK: bool = _get_bool("RERANK", True)

OUTPUT_DIR: str = "./output"
SCORE_PATH: str = "./output/similarity_score.json"
