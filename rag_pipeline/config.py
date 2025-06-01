from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _validate_path(path: Path, description: str) -> Path:
    if not path.exists():
        print(f"Warning: {description} not found at {path}")
        if not path.suffix:  # 확장자가 없으면 디렉토리로 간주
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")
    return path


# ----- 벡터 DB 및 임베딩 설정 -----
EMBED_MODEL_NAME: str = "jinaai/jina-embeddings-v3"
RERANKER_NAME: str = "BAAI/bge-reranker-v2-m3"
CONTENT_DB_PATH: Path = Path(
    "C:/Users/juk27/OneDrive/Desktop/JH/rag-for-semicon-physics/vectordb/faiss"
)

# ----- OpenAI API 설정 -----
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your OpenAI API key. "
        "See .env.example for reference."
    )

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ----- 검색 파라미터 -----
TOP_K: int = int(os.getenv("TOP_K", 3))
SIM_THRESHOLD: float = float(os.getenv("SIM_THRESHOLD", 0.70))
RERANK: bool = _get_bool("RERANK", False)
HYBRID_WEIGHT: float = float(
    os.getenv("HYBRID_WEIGHT", 0.5)
)  # hybrid retrieval에서 vector embedding similarity의 가중치 (BM25 가중치: 1-config.HYBRID_WEIGHT)
RETRIEVAL_TYPE: str = str(
    os.getenv("RETRIEVAL_TYPE", "original_query")
)  # original_query, hyde, summary 중에 선택

# ----- 출력 디렉토리 설정 -----
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
OUTPUT_PATH = _validate_path(Path(OUTPUT_DIR), "Output directory")

SCORE_PATH: str = str(OUTPUT_PATH / "similarity_score.json")
SAVE_PATH: str = str(OUTPUT_PATH / "similarity_score.json")


# 설정 검증
print(f"Configuration loaded:")
print(f"  - Model: {OPENAI_MODEL}")
print(f"  - TOP_K: {TOP_K}")
print(f"  - SIM_THRESHOLD: {SIM_THRESHOLD}")
print(f"  - RERANK: {RERANK}")
print(f"  - Output directory: {OUTPUT_PATH}")
