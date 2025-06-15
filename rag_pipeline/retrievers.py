from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# from langchain.embeddings import (
#     HuggingFaceEmbeddings,
#     HuggingFaceCrossEncoder,
# )
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.schema.messages import HumanMessage

from rag_pipeline import config, utils
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# 기본 임베딩 모델 로드
model = SentenceTransformer(config.EMBED_MODEL_NAME, trust_remote_code=True)
model.to(device)

# LangChain용 임베딩 래퍼
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBED_MODEL_NAME,
    model_kwargs={"device": device, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

# Cross-Encoder Reranker
reranker = HuggingFaceCrossEncoder(model_name=config.RERANKER_NAME)


def load_parent_store(jsonl_path: Path) -> InMemoryStore:
    """JSONL 파일을 읽어 InMemoryStore에 적재"""
    with jsonl_path.open("r", encoding="utf-8") as f:
        docs = [
            Document(
                id=rec["id"],
                page_content=rec["page_content"],
                metadata=rec["metadata"],
            )
            for rec in map(json.loads, f)
        ]
    store = InMemoryStore()
    store.mset([(d.id, d) for d in docs])
    return store


def _rerank(query: str, docs: List[Document]) -> Tuple[List[Document], List[float]]:
    """Cross-Encoder 점수로 재정렬 - 디버깅 로그 추가"""
    print(f"🔄 Starting reranking with {len(docs)} documents...")

    try:
        passages = [d.page_content for d in docs]
        print(f"   Extracted {len(passages)} passages")

        pairs = [[query, passage] for passage in passages]
        print(f"   Created {len(pairs)} query-passage pairs")

        print(f"   Using reranker: {config.RERANKER_NAME}")
        scores = reranker.score(pairs)
        print(f"   ✅ Reranker scores computed: {scores}")

        ranked = sorted(zip(docs, scores), key=lambda t: t[1], reverse=True)
        docs_sorted, scores_sorted = zip(*ranked)

        print(f"   ✅ Reranking completed successfully")
        return list(docs_sorted), list(scores_sorted)

    except Exception as e:
        print(f"   ❌ Error in _rerank: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise e


def retrieve_from_file_embedding(
    query: HumanMessage | str, pdf_path: Path, top_k: int = config.TOP_K
) -> List[Document]:
    docs = utils.pdf_to_docs(pdf_path)

    if not docs:
        return "Failed to extract text from PDF!"

    query_text = query.content if hasattr(query, "content") else query
    texts = [d.page_content for d in docs]

    # Determine retrieval approach based on config
    hybrid_weight_check = config.HYBRID_WEIGHT < 1.0

    if config.RETRIEVAL_TYPE == "original_query":
        search_query = query_text
    elif config.RETRIEVAL_TYPE == "hyde":
        # Generate HyDE documents and use their average embedding
        hypo_docs = []
        hydes = []
        for _ in range(5):
            try:
                hypo_doc = utils.generate_hyde_document(query_text)
                embedding = model.encode(hypo_doc, normalize_embeddings=True)
                hypo_docs.append(hypo_doc)  # 텍스트 저장
                hydes.append(embedding)  # 벡터 저장
            except Exception as e:
                print(f"Error generating HyDE document: {e}")
                continue

        if hydes:
            mean_hyde = np.mean(np.stack(hydes, axis=0), axis=0)
            norm = np.linalg.norm(mean_hyde)
            if norm > 0:
                mean_hyde /= norm
            q_vecs = mean_hyde
        else:
            q_vecs = model.encode(
                [query_text], convert_to_tensor=True, normalize_embeddings=True
            )[0]
        search_query = None
    elif config.RETRIEVAL_TYPE == "summary":
        search_query = utils.generate_summary(query_text)
    elif config.RETRIEVAL_TYPE == "summary_mean":
        # Generate multiple summaries and use their average embedding
        summaries = []
        summary_texts = []
        for _ in range(5):
            try:
                summary_text = utils.generate_summary(query_text)
                embedding = model.encode(summary_text, normalize_embeddings=True)
                summary_texts.append(summary_text)
                summaries.append(embedding)
            except Exception as e:
                print(f"Error generating summary: {e}")
                continue

        if summaries:
            mean_summary = np.mean(np.stack(summaries, axis=0), axis=0)
            norm = np.linalg.norm(mean_summary)
            if norm > 0:
                mean_summary /= norm
            q_vecs = mean_summary
        else:
            q_vecs = model.encode(
                [query_text], convert_to_tensor=True, normalize_embeddings=True
            )[0]
        search_query = None
    else:
        search_query = query_text

    # Calculate similarity scores
    doc_vecs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    if search_query is not None:
        q_vecs = model.encode(
            [search_query], convert_to_tensor=True, normalize_embeddings=True
        )[0]

    # Vector similarity scores
    cos_sim = util.cos_sim(q_vecs, doc_vecs)[0].float().cpu().numpy()

    if hybrid_weight_check:
        # Hybrid retrieval: combine vector similarity with BM25

        # Tokenize documents for BM25
        tokenized_docs = [doc.split() for doc in texts]
        bm25 = BM25Okapi(tokenized_docs)

        if config.RETRIEVAL_TYPE == "hyde":
            search_query = hypo_docs[0] if hypo_docs else query_text
        elif config.RETRIEVAL_TYPE == "summary_mean":
            search_query = summary_texts[0] if summary_texts else query_text

        bm25_query = search_query.split()
        bm25_scores = np.array(bm25.get_scores(bm25_query))

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )

        # Combine scores using hybrid weights
        hybrid_weight_embedding = config.HYBRID_WEIGHT
        hybrid_weight_bm25 = 1.0 - hybrid_weight_embedding

        combined_scores = (
            hybrid_weight_embedding * cos_sim + hybrid_weight_bm25 * bm25_scores
        )

        best_idx = combined_scores.argsort()[-top_k:][::-1]
        best_scores = combined_scores[best_idx]
    else:
        # Pure vector similarity
        best_idx = cos_sim.argsort()[-top_k:][::-1]
        best_scores = cos_sim[best_idx]

    best_docs = [docs[i] for i in best_idx]

    with open(config.SCORE_PATH, "w") as f:
        json.dump(best_scores.tolist(), f)

    # Apply reranking if enabled
    if config.RERANK:
        print("🔄 Applying reranking...")
        try:
            reranked_docs, scores = _rerank(query_text, best_docs)
            print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
            print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
            return reranked_docs
        except Exception as rerank_error:
            print(f"   ❌ Reranking failed: {rerank_error}")
            print(f"   Falling back to original results")
            return best_docs
    else:
        print("⏭️ Skipping reranking (disabled)")

    return best_docs


def retrieve_from_img_embedding(
    query: HumanMessage | str, img_path: Path, top_k: int = config.TOP_K
) -> List[Document]:
    docs = utils.img_to_docs(img_path)

    if not docs:
        return "Failed to extract text from image!"

    query_text = query.content if hasattr(query, "content") else query
    texts = [d.page_content for d in docs]

    # Determine retrieval approach based on config
    hybrid_weight_check = config.HYBRID_WEIGHT < 1.0

    if config.RETRIEVAL_TYPE == "original_query":
        search_query = query_text
    elif config.RETRIEVAL_TYPE == "hyde":
        # Generate HyDE documents and use their average embedding
        hydes = []
        hypo_docs = []
        for _ in range(5):
            try:
                hypo_doc = utils.generate_hyde_document(query_text)
                embedding = model.encode(hypo_doc, normalize_embeddings=True)
                hydes.append(embedding)
                hypo_docs.append(hypo_doc)
            except Exception as e:
                print(f"Error generating HyDE document: {e}")
                continue

        if hydes:
            mean_hyde = np.mean(np.stack(hydes, axis=0), axis=0)
            norm = np.linalg.norm(mean_hyde)
            if norm > 0:
                mean_hyde /= norm
            q_vecs = mean_hyde
        else:
            q_vecs = model.encode(
                [query_text], convert_to_tensor=True, normalize_embeddings=True
            )[0]
        search_query = None
    elif config.RETRIEVAL_TYPE == "summary":
        search_query = utils.generate_summary(query_text)
    elif config.RETRIEVAL_TYPE == "summary_mean":
        # Generate multiple summaries and use their average embedding
        summaries = []
        summary_texts = []
        for _ in range(5):
            try:
                summary_text = utils.generate_summary(query_text)
                embedding = model.encode(summary_text, normalize_embeddings=True)
                summary_texts.append(summary_text)
                summaries.append(embedding)
            except Exception as e:
                print(f"Error generating summary: {e}")
                continue

        if summaries:
            mean_summary = np.mean(np.stack(summaries, axis=0), axis=0)
            norm = np.linalg.norm(mean_summary)
            if norm > 0:
                mean_summary /= norm
            q_vecs = mean_summary
        else:
            q_vecs = model.encode(
                [query_text], convert_to_tensor=True, normalize_embeddings=True
            )[0]
        search_query = None
    else:
        search_query = query_text

    # Calculate similarity scores
    doc_vecs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    if search_query is not None:
        q_vecs = model.encode(
            [search_query], convert_to_tensor=True, normalize_embeddings=True
        )[0]

    # Vector similarity scores
    cos_sim = util.cos_sim(q_vecs, doc_vecs)[0].float().cpu().numpy()

    if hybrid_weight_check:
        # Hybrid retrieval: combine vector similarity with BM25

        # Tokenize documents for BM25
        tokenized_docs = [doc.split() for doc in texts]
        bm25 = BM25Okapi(tokenized_docs)

        if config.RETRIEVAL_TYPE == "hyde":
            search_query = hypo_docs[0] if hypo_docs else query_text
        elif config.RETRIEVAL_TYPE == "summary_mean":
            search_query = summary_texts[0] if summary_texts else query_text

        bm25_query = search_query.split()
        bm25_scores = np.array(bm25.get_scores(bm25_query))

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )

        # Combine scores using hybrid weights
        hybrid_weight_embedding = config.HYBRID_WEIGHT
        hybrid_weight_bm25 = 1.0 - hybrid_weight_embedding

        combined_scores = (
            hybrid_weight_embedding * cos_sim + hybrid_weight_bm25 * bm25_scores
        )

        best_idx = combined_scores.argsort()[-top_k:][::-1]
        best_scores = combined_scores[best_idx]
    else:
        # Pure vector similarity
        best_idx = cos_sim.argsort()[-top_k:][::-1]
        best_scores = cos_sim[best_idx]

    best_docs = [docs[i] for i in best_idx]

    with open(config.SCORE_PATH, "w") as f:
        json.dump(best_scores.tolist(), f)

    # Apply reranking if enabled
    if config.RERANK:
        print("🔄 Applying reranking...")
        try:
            reranked_docs, scores = _rerank(query_text, best_docs)
            print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
            print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
            return reranked_docs
        except Exception as rerank_error:
            print(f"   ❌ Reranking failed: {rerank_error}")
            print(f"   Falling back to original results")
            return best_docs
    else:
        print("⏭️ Skipping reranking (disabled)")

    return best_docs


def vectordb_retrieve(query: HumanMessage | str) -> List[Document]:
    """기본 벡터 DB 검색 - 상세한 디버깅 로그 추가"""
    print(f"🔍 Starting vectordb_retrieve with query: {query}")

    try:
        # Step 1: Query 변환
        print("📝 Step 1: Converting query to text...")
        query_text = query.content if hasattr(query, "content") else query
        print(f"   Query text: '{query_text}' (type: {type(query_text)})")

        # Step 2: Vector DB 로딩
        print("📂 Step 2: Loading vector database...")
        print(f"   DB path: {config.CONTENT_DB_PATH}")
        print(f"   Path exists: {config.CONTENT_DB_PATH.exists()}")

        if not config.CONTENT_DB_PATH.exists():
            raise FileNotFoundError(
                f"Vector database not found at {config.CONTENT_DB_PATH}"
            )

        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"   ✅ Vector DB loaded successfully")
        print(f"   DB info: {len(vectordb.docstore._dict)} documents in store")

        # Step 3: Query 임베딩 생성
        print("🔢 Step 3: Generating query embedding...")
        query_emb = model.encode(
            query_text,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        print(f"   ✅ Query embedding shape: {query_emb.shape}")
        print(f"   Embedding dtype: {query_emb.dtype}")

        # Step 4: 유사도 검색
        print(f"🔍 Step 4: Performing similarity search (TOP_K={config.TOP_K})...")
        sem = vectordb.similarity_search_by_vector(query_emb, k=config.TOP_K)
        print(f"   ✅ Found {len(sem)} documents")

        if not sem:
            print("   ⚠️ Warning: No documents found in similarity search")
            return []

        # 검색 결과 미리보기
        for i, doc in enumerate(sem[:2]):  # 첫 2개 문서만 미리보기
            preview = (
                doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            )
            print(f"   Doc {i+1} preview: {preview}")

        # Step 5: 문서 임베딩 생성
        print("🔢 Step 5: Generating document embeddings...")
        doc_contents = [d.page_content for d in sem]
        print(f"   Processing {len(doc_contents)} document contents")

        doc_vecs = model.encode(
            doc_contents,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        print(f"   ✅ Document embeddings shape: {doc_vecs.shape}")

        # Step 6: 코사인 유사도 계산
        print("📊 Step 6: Computing cosine similarity...")
        cos_sim = util.cos_sim(query_emb, doc_vecs)[0].float().cpu().numpy()
        print(f"   ✅ Similarity scores: {cos_sim}")
        print(f"   Max score: {cos_sim.max():.4f}, Min score: {cos_sim.min():.4f}")

        # Step 7: 출력 디렉토리 확인 및 점수 저장
        print("💾 Step 7: Saving similarity scores...")
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {output_dir}")

        with open(config.SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(cos_sim.tolist(), f, ensure_ascii=False)
        print(f"   ✅ Scores saved to: {config.SAVE_PATH}")

        # Step 8: Reranking (선택적)
        if config.RERANK:
            print("🔄 Step 8: Applying reranking...")
            try:
                reranked_docs, scores = _rerank(query_text, sem)
                print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
                print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
                return reranked_docs
            except Exception as rerank_error:
                print(f"   ❌ Reranking failed: {rerank_error}")
                print(f"   Falling back to original results")
                return sem
        else:
            print("⏭️ Step 8: Skipping reranking (disabled)")

        print("✅ vectordb_retrieve completed successfully")
        return sem

    except FileNotFoundError as e:
        print(f"❌ FileNotFoundError in vectordb_retrieve: {e}")
        print(f"   Check if vector database exists at: {config.CONTENT_DB_PATH}")
        return []

    except Exception as e:
        print(f"❌ Unexpected error in vectordb_retrieve: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error details: {str(e)}")

        # 스택 트레이스 출력
        import traceback

        print("📋 Full traceback:")
        traceback.print_exc()

        return []


def vectordb_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float]
) -> List[Document]:
    """FAISS + BM25 하이브리드 검색 - 반환값 일관성 수정"""
    try:
        query_text = query.content if hasattr(query, "content") else query

        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # Use weighted sum hybrid approach instead of ensemble retriever
        all_docs: List[Document] = list(vectordb.docstore._dict.values())
        texts: List[str] = [doc.page_content for doc in all_docs]

        # Vector similarity scores
        query_emb = model.encode(
            query_text, convert_to_tensor=False, normalize_embeddings=True
        )
        doc_vecs = model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        cos_sim_scores = util.cos_sim(query_emb, doc_vecs)[0].cpu().numpy()

        # BM25 scores
        tokenized_texts = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        bm25_query_tokens = query_text.split()
        bm25_scores = np.array(bm25.get_scores(bm25_query_tokens))

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # Combine scores using weighted sum
        w1, w2 = weights
        hybrid_scores = w1 * cos_sim_scores + w2 * bm25_scores

        # Get top-k documents
        top_indices = hybrid_scores.argsort()[-config.TOP_K :][::-1]
        sem = [all_docs[i] for i in top_indices]
        top_scores = hybrid_scores[top_indices]

        # 출력 디렉토리 생성 확인
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump(top_scores.tolist(), f, ensure_ascii=False)

        # Apply reranking if enabled
        if config.RERANK:
            print("🔄 Applying reranking...")
            try:
                reranked_docs, scores = _rerank(query_text, sem)
                print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
                print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
                return reranked_docs
            except Exception as rerank_error:
                print(f"   ❌ Reranking failed: {rerank_error}")
                print(f"   Falling back to original results")
                return sem
        else:
            print("⏭️ Skipping reranking (disabled)")

        return sem

    except Exception as e:
        print(f"Error in vectordb_hybrid_retrieve: {e}")
        return []


def summary_retrieve(query: HumanMessage | str) -> Tuple[List[Document], str]:
    """FAISS + LLM 설명 + 임베딩 검색"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    query_explanation = utils.generate_summary(query_text)

    query_emb = model.encode(
        query_explanation,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    sem = vectordb.similarity_search_by_vector(query_emb, k=config.TOP_K)

    doc_vecs = model.encode(
        [d.page_content for d in sem],
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    cos_sim = util.cos_sim(query_emb, doc_vecs)[0].float().cpu().numpy()
    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump(cos_sim.tolist(), f, ensure_ascii=False)

    # Apply reranking if enabled
    if config.RERANK:
        print("🔄 Applying reranking...")
        try:
            reranked_docs, scores = _rerank(query_text, sem)
            print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
            print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
            return reranked_docs, query_explanation
        except Exception as rerank_error:
            print(f"   ❌ Reranking failed: {rerank_error}")
            print(f"   Falling back to original results")
            return sem, query_explanation
    else:
        print("⏭️ Skipping reranking (disabled)")

    return sem, query_explanation


def summary_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float] = [0.5, 0.5]
) -> Tuple[List[Document], str]:
    """FAISS + BM25 하이브리드 검색 + LLM 설명"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # LLM으로 질문 설명 생성
    query_explanation = utils.generate_summary(query_text)

    all_docs: List[Document] = list(vectordb.docstore._dict.values())
    texts: List[str] = [doc.page_content for doc in all_docs]

    # Step 5: FAISS 기반 코사인 유사도 계산
    doc_vecs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    query_vec = model.encode(
        query_explanation, convert_to_tensor=True, normalize_embeddings=True
    )

    cos_sim_scores = util.cos_sim(query_vec, doc_vecs)[0].cpu().numpy()

    # Step 6: BM25 점수 계산
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    bm25_query = query_explanation.split()
    bm25_scores = np.array(bm25.get_scores(bm25_query))

    # Step 7: BM25 점수 정규화
    if bm25_scores.max() > bm25_scores.min():
        bm25_scores = (bm25_scores - bm25_scores.min()) / (
            bm25_scores.max() - bm25_scores.min()
        )

    # Step 8: 가중합 스코어 계산
    w1, w2 = weights
    hybrid_scores = w1 * cos_sim_scores + w2 * bm25_scores

    # Step 9: 정렬 및 결과 문서 추출
    top_k = config.TOP_K
    top_indices = hybrid_scores.argsort()[-top_k:][::-1]
    top_docs = [all_docs[i] for i in top_indices]
    top_scores = hybrid_scores[top_indices]

    # Step 10: 저장 및 반환
    Path(config.SCORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump(top_scores.tolist(), f, ensure_ascii=False)

    # Apply reranking if enabled
    if config.RERANK:
        print("🔄 Applying reranking...")
        try:
            reranked_docs, scores = _rerank(query_text, top_docs)
            print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
            print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
            return reranked_docs, query_explanation
        except Exception as rerank_error:
            print(f"   ❌ Reranking failed: {rerank_error}")
            print(f"   Falling back to original results")
            return top_docs, query_explanation
    else:
        print("⏭️ Skipping reranking (disabled)")

    return top_docs, query_explanation


def hyde_retrieve(query: str) -> Tuple[List[Document], List[str]]:
    """HyDE 검색 - 에러 처리 및 반환값 일관성 개선"""
    try:
        query_text = query.content if hasattr(query, "content") else query

        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        hydes: List[np.ndarray] = []
        hypo_docs: List[str] = []

        for i in range(5):
            try:
                hypo_doc = utils.generate_hyde_document(query_text)
                hypo_docs.append(hypo_doc)

                embedding = model.encode(
                    hypo_doc,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                ).cpu()
                hydes.append(embedding)

            except Exception as e:
                print(f"Error generating HyDE document {i+1}: {e}")
                continue

        if not hydes:
            print("Warning: No HyDE documents generated, falling back to direct search")
            return vectordb_retrieve(query), []

        mean_hyde = torch.stack(hydes).mean(dim=0)
        mean_hyde = F.normalize(mean_hyde, p=2, dim=0)

        mean_hyde_np = mean_hyde.float().numpy()

        sem = vectordb.similarity_search_by_vector(mean_hyde_np, k=config.TOP_K)

        sem_vecs = model.encode(
            [d.page_content for d in sem],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()

        dtype = torch.float32
        mean_hyde = mean_hyde.to(dtype)
        sem_vecs = sem_vecs.to(dtype)
        sem_hyde_cos_sim = util.cos_sim(mean_hyde.unsqueeze(0), sem_vecs)[0].numpy()

        # 출력 디렉토리 생성 확인
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump(sem_hyde_cos_sim.tolist(), f, ensure_ascii=False)

        # Apply reranking if enabled
        if config.RERANK:
            print("🔄 Applying reranking...")
            try:
                reranked_docs, scores = _rerank(query_text, sem)
                print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
                print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
                return reranked_docs, hypo_docs
            except Exception as rerank_error:
                print(f"   ❌ Reranking failed: {rerank_error}")
                print(f"   Falling back to original results")
                return sem, hypo_docs
        else:
            print("⏭️ Skipping reranking (disabled)")

        return sem, hypo_docs

    except Exception as e:
        print(f"Error in hyde_retrieve: {e}")
        return [], []


def hyde_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float]
) -> Tuple[List[Document], str]:
    """HyDE + 하이브리드 검색"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    all_docs: List[Document] = list(vectordb.docstore._dict.values())
    texts: List[str] = [doc.page_content for doc in all_docs]

    # HyDE 문서 생성
    hydes: List[np.ndarray] = []
    hypo_docs: List[str] = []

    for _ in range(5):
        hypo_doc = utils.generate_hyde_document(query_text)
        hypo_docs.append(hypo_doc)

        embedding = model.encode(
            hypo_doc,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        hydes.append(embedding)

    # 가설 문서 임베딩의 평균 계산
    mean_hyde = np.mean(np.stack(hydes, axis=0), axis=0)
    norm = np.linalg.norm(mean_hyde)
    if norm > 0:
        mean_hyde /= norm

    doc_vecs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    cos_sim_scores = util.cos_sim(mean_hyde, doc_vecs)[0].cpu().numpy()  # (N,)

    # BM25
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    bm25_query = hypo_docs[0] if hypo_docs else query_text
    bm25_query_tokens = bm25_query.split()
    bm25_raw_scores = np.array(bm25.get_scores(bm25_query_tokens))

    if bm25_raw_scores.max() > bm25_raw_scores.min():
        bm25_scores = (bm25_raw_scores - bm25_raw_scores.min()) / (
            bm25_raw_scores.max() - bm25_raw_scores.min()
        )
    else:
        bm25_scores = np.zeros_like(bm25_raw_scores)

    w1, w2 = weights
    hybrid_scores = w1 * cos_sim_scores + w2 * bm25_scores

    top_k = config.TOP_K
    top_indices = hybrid_scores.argsort()[-top_k:][::-1]
    top_docs = [all_docs[i] for i in top_indices]
    top_scores = hybrid_scores[top_indices]

    Path(config.SCORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump([float(s) for s in top_scores], f, ensure_ascii=False)

    # Apply reranking if enabled
    if config.RERANK:
        print("🔄 Applying reranking...")
        try:
            reranked_docs, scores = _rerank(query_text, top_docs)
            print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
            print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
            return reranked_docs, hypo_docs
        except Exception as rerank_error:
            print(f"   ❌ Reranking failed: {rerank_error}")
            print(f"   Falling back to original results")
            return top_docs, hypo_docs
    else:
        print("⏭️ Skipping reranking (disabled)")

    return top_docs, hypo_docs


def query_expansion_retrieve(
    query: HumanMessage | str,
) -> Tuple[List[Document], List[dict]]:
    print(f"🔍 Starting query_expansion_retrieve with query: {query}")

    try:
        # Step 1: Query text extraction
        query_text = query.content if hasattr(query, "content") else query
        print(f"Query text: '{query_text}'")

        # Step 2: Load vector databases
        print("📂 Loading vector databases...")
        content_vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        summary_vectordb = FAISS.load_local(
            config.SUMMARY_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("   ✅ Vector databases loaded successfully")

        # Step 3: Generate query embedding based on retrieval type
        print(f"🔢 Generating query embedding (type: {config.RETRIEVAL_TYPE})...")

        if config.RETRIEVAL_TYPE == "summary":
            query_explanation = utils.generate_summary(query_text)
            query_emb = model.encode(
                query_explanation,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
        elif config.RETRIEVAL_TYPE == "hyde":
            # Generate 5 HyDE documents and use their average embedding
            hydes = []
            for _ in range(5):
                try:
                    hypo_doc = utils.generate_hyde_document(query_text)
                    embedding = model.encode(hypo_doc, normalize_embeddings=True)
                    hydes.append(embedding)
                except Exception as e:
                    print(f"   Warning: Error generating HyDE document: {e}")
                    continue

            if hydes:
                query_emb = np.mean(np.stack(hydes, axis=0), axis=0)
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
            else:
                print("   Warning: No HyDE documents generated, using original query")
                query_emb = model.encode(query_text, normalize_embeddings=True)
        elif config.RETRIEVAL_TYPE == "summary_mean":
            # Generate 5 summaries and use their average embedding
            summaries = []
            for _ in range(5):
                try:
                    summary_text = utils.generate_summary(query_text)
                    embedding = model.encode(summary_text, normalize_embeddings=True)
                    summaries.append(embedding)
                except Exception as e:
                    print(f"   Warning: Error generating summary: {e}")
                    continue

            if summaries:
                query_emb = np.mean(np.stack(summaries, axis=0), axis=0)
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
            else:
                print("   Warning: No summaries generated, using original query")
                query_emb = model.encode(query_text, normalize_embeddings=True)
        else:
            query_emb = model.encode(query_text, normalize_embeddings=True)

        print(f"   ✅ Query embedding generated, shape: {query_emb.shape}")

        # Step 4: Retrieve from content database
        print("🔍 Retrieving from content database...")
        sem = content_vectordb.similarity_search_by_vector(query_emb, k=config.TOP_K)
        print(f"   ✅ Retrieved {len(sem)} content documents")

        # Step 5: Create expanded query with content
        print("📝 Creating expanded query with content...")
        content_texts = [doc.page_content for doc in sem]
        query_with_content = query_text + "\n" + "\n".join(content_texts)

        query_with_content_embed = model.encode(
            query_with_content,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        print(f"   ✅ Expanded query embedding generated")

        # Step 6: Retrieve from summary/examples database
        print("🔍 Retrieving from summary/examples database...")
        summary_sem = summary_vectordb.similarity_search_by_vector(
            query_with_content_embed, k=config.TOP_K
        )
        print(f"   ✅ Retrieved {len(summary_sem)} summary documents")

        # Step 7: Load parent documents
        print("📂 Loading parent documents...")
        parent_child_matching_dir = Path(
            "./vectordb/jina_processed/examples_original.jsonl"
        )

        if not parent_child_matching_dir.exists():
            print(
                f"   ⚠️ Warning: Parent-child mapping file not found at {parent_child_matching_dir}"
            )
            parent_docs = []
        else:
            parent_ids = [
                d.metadata.get("parent_id")
                for d in summary_sem
                if d.metadata.get("parent_id")
            ]
            parent_doc_map = {}

            with open(parent_child_matching_dir, "r", encoding="utf-8") as f:
                for line in f:
                    parent_doc = json.loads(line)
                    parent_doc_map[parent_doc["id"].replace("parent-", "")] = parent_doc

            parent_docs = []
            for parent_id in parent_ids:
                if parent_id in parent_doc_map:
                    parent_docs.append(parent_doc_map[parent_id])

            print(f"   ✅ Loaded {len(parent_docs)} parent documents")

        # Step 8: Calculate similarity scores
        print("📊 Calculating similarity scores...")

        # Content document embeddings
        content_doc_vecs = model.encode(
            [d.page_content for d in sem],
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Summary document embeddings
        summary_doc_vecs = model.encode(
            [d.page_content for d in summary_sem],
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Calculate cosine similarities
        content_query_cos_sim = (
            util.cos_sim(query_emb, content_doc_vecs)[0].float().cpu().numpy()
        )
        summary_query_cos_sim = (
            util.cos_sim(query_emb, summary_doc_vecs)[0].float().cpu().numpy()
        )
        summary_expanded_query_cos_sim = (
            util.cos_sim(query_with_content_embed, summary_doc_vecs)[0]
            .float()
            .cpu()
            .numpy()
        )

        print(f"   ✅ Similarity scores calculated")

        # Step 9: Save similarity scores
        print("💾 Saving similarity scores...")
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(
            output_dir / "content_query_similarity_score.json", "w", encoding="utf-8"
        ) as f:
            json.dump(content_query_cos_sim.tolist(), f, ensure_ascii=False)

        with open(
            output_dir / "summary_query_similarity_score.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary_query_cos_sim.tolist(), f, ensure_ascii=False)

        with open(
            output_dir / "content_expanded_query_similarity_score.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary_expanded_query_cos_sim.tolist(), f, ensure_ascii=False)

        print("   ✅ All similarity scores saved")

        # Step 10: Apply reranking if enabled
        if config.RERANK:
            print("🔄 Applying reranking...")
            try:
                reranked_docs, scores = _rerank(query_text, sem)
                print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
                print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
                return reranked_docs, parent_docs
            except Exception as rerank_error:
                print(f"   ❌ Reranking failed: {rerank_error}")
                print(f"   Falling back to original results")
                return sem, parent_docs
        else:
            print("⏭️ Skipping reranking (disabled)")

        print("✅ query_expansion_retrieve completed successfully")
        return sem, parent_docs

    except Exception as e:
        print(f"❌ Error in query_expansion_retrieve: {e}")
        import traceback

        traceback.print_exc()
        return [], []


def query_expansion_retrieve_hybrid(
    query: HumanMessage | str,
    weights: List[float] = [0.5, 0.5],
    weights_examples: List[float] = [0.5, 0.5],
) -> Tuple[List[Document], List[dict]]:
    """Query expansion hybrid retrieval with weighted sum approach"""
    print(f"🔍 Starting query_expansion_retrieve_hybrid with query: {query}")

    try:
        # Step 1: Query text extraction
        query_text = query.content if hasattr(query, "content") else query
        print(f"   Query text: '{query_text}'")

        # Step 2: Load vector databases
        print("📂 Loading vector databases...")
        content_vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        summary_vectordb = FAISS.load_local(
            config.SUMMARY_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("   ✅ Vector databases loaded successfully")

        # Step 3: Generate query embedding based on retrieval type
        print(f"🔢 Generating query embedding (type: {config.RETRIEVAL_TYPE})...")

        if config.RETRIEVAL_TYPE == "summary":
            query_explanation = utils.generate_summary(query_text)
            query_emb = model.encode(query_explanation, normalize_embeddings=True)
            search_query = query_explanation
        elif config.RETRIEVAL_TYPE == "hyde":
            hydes = []
            for _ in range(5):
                try:
                    hypo_doc = utils.generate_hyde_document(query_text)
                    embedding = model.encode(hypo_doc, normalize_embeddings=True)
                    hydes.append(embedding)
                except Exception as e:
                    print(f"   Warning: Error generating HyDE document: {e}")
                    continue

            if hydes:
                query_emb = np.mean(np.stack(hydes, axis=0), axis=0)
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
            else:
                print("   Warning: No HyDE documents generated, using original query")
                query_emb = model.encode(query_text, normalize_embeddings=True)
            search_query = query_text
        elif config.RETRIEVAL_TYPE == "summary_mean":
            summaries = []
            for _ in range(5):
                try:
                    summary_text = utils.generate_summary(query_text)
                    embedding = model.encode(summary_text, normalize_embeddings=True)
                    summaries.append(embedding)
                except Exception as e:
                    print(f"   Warning: Error generating summary: {e}")
                    continue

            if summaries:
                query_emb = np.mean(np.stack(summaries, axis=0), axis=0)
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
            else:
                print("   Warning: No summaries generated, using original query")
                query_emb = model.encode(query_text, normalize_embeddings=True)
            search_query = query_text
        else:
            query_emb = model.encode(query_text, normalize_embeddings=True)
            search_query = query_text

        print(f"   ✅ Query embedding generated")

        # Step 4: Weighted sum hybrid retrieval from content database
        print("🔍 Performing weighted sum hybrid retrieval from content database...")
        all_content_docs = list(content_vectordb.docstore._dict.values())
        content_texts = [doc.page_content for doc in all_content_docs]

        # Vector similarity scores
        content_doc_vecs = model.encode(
            content_texts, convert_to_tensor=True, normalize_embeddings=True
        )
        content_cos_sim_scores = (
            util.cos_sim(query_emb, content_doc_vecs)[0].cpu().numpy()
        )

        # BM25 scores
        content_tokenized_texts = [text.split() for text in content_texts]
        content_bm25 = BM25Okapi(content_tokenized_texts)
        content_bm25_scores = np.array(content_bm25.get_scores(search_query.split()))

        # Normalize BM25 scores
        if content_bm25_scores.max() > content_bm25_scores.min():
            content_bm25_scores = (content_bm25_scores - content_bm25_scores.min()) / (
                content_bm25_scores.max() - content_bm25_scores.min()
            )
        else:
            content_bm25_scores = np.zeros_like(content_bm25_scores)

        # Combine scores
        w1, w2 = weights
        content_hybrid_scores = w1 * content_cos_sim_scores + w2 * content_bm25_scores

        # Get top-k content documents
        content_top_indices = content_hybrid_scores.argsort()[-config.TOP_K :][::-1]
        sem = [all_content_docs[i] for i in content_top_indices]
        print(
            f"   ✅ Retrieved {len(sem)} content documents via weighted sum hybrid search"
        )

        # Step 5: Create expanded query with content
        print("📝 Creating expanded query with content...")
        content_texts = [doc.page_content for doc in sem]
        query_with_content = query_text + "\n" + "\n".join(content_texts)

        query_with_content_embed = model.encode(
            query_with_content,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        print(f"   ✅ Expanded query embedding generated")

        # Step 6: Weighted sum hybrid retrieval from summary/examples database
        print(
            "🔍 Performing weighted sum hybrid retrieval from summary/examples database..."
        )
        all_summary_docs = list(summary_vectordb.docstore._dict.values())
        summary_texts = [doc.page_content for doc in all_summary_docs]

        # Vector similarity scores
        summary_doc_vecs = model.encode(
            summary_texts, convert_to_tensor=True, normalize_embeddings=True
        )
        summary_cos_sim_scores = (
            util.cos_sim(query_with_content_embed, summary_doc_vecs)[0].cpu().numpy()
        )

        # BM25 scores
        summary_tokenized_texts = [text.split() for text in summary_texts]
        summary_bm25 = BM25Okapi(summary_tokenized_texts)
        summary_bm25_scores = np.array(
            summary_bm25.get_scores(query_with_content.split())
        )

        # Normalize BM25 scores
        if summary_bm25_scores.max() > summary_bm25_scores.min():
            summary_bm25_scores = (summary_bm25_scores - summary_bm25_scores.min()) / (
                summary_bm25_scores.max() - summary_bm25_scores.min()
            )
        else:
            summary_bm25_scores = np.zeros_like(summary_bm25_scores)

        # Combine scores
        w3, w4 = weights_examples
        summary_hybrid_scores = w3 * summary_cos_sim_scores + w4 * summary_bm25_scores

        # Get top-k summary documents
        summary_top_indices = summary_hybrid_scores.argsort()[-config.TOP_K :][::-1]
        summary_sem = [all_summary_docs[i] for i in summary_top_indices]
        print(
            f"   ✅ Retrieved {len(summary_sem)} summary documents via weighted sum hybrid search"
        )

        # Step 7: Load parent documents
        print("📂 Loading parent documents...")
        parent_child_matching_dir = Path(
            "./vectordb/jina_processed/examples_original.jsonl"
        )

        if not parent_child_matching_dir.exists():
            print(
                f"   ⚠️ Warning: Parent-child mapping file not found at {parent_child_matching_dir}"
            )
            parent_docs = []
        else:
            parent_ids = [
                d.metadata.get("parent_id")
                for d in summary_sem
                if d.metadata.get("parent_id")
            ]
            parent_doc_map = {}

            with open(parent_child_matching_dir, "r", encoding="utf-8") as f:
                for line in f:
                    parent_doc = json.loads(line)
                    parent_doc_map[parent_doc["id"].replace("parent-", "")] = parent_doc

            parent_docs = []
            for parent_id in parent_ids:
                if parent_id in parent_doc_map:
                    parent_docs.append(parent_doc_map[parent_id])

            print(f"   ✅ Loaded {len(parent_docs)} parent documents")

        # Step 8: Save similarity scores
        print("💾 Saving similarity scores...")
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        content_selected_scores = content_hybrid_scores[content_top_indices]
        summary_selected_scores = summary_hybrid_scores[summary_top_indices]

        with open(
            output_dir / "content_query_similarity_score.json", "w", encoding="utf-8"
        ) as f:
            json.dump(content_selected_scores.tolist(), f, ensure_ascii=False)

        with open(
            output_dir / "summary_query_similarity_score.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary_selected_scores.tolist(), f, ensure_ascii=False)

        print("   ✅ All similarity scores saved")

        # Step 9: Apply reranking if enabled
        if config.RERANK:
            print("🔄 Applying reranking...")
            try:
                reranked_docs, scores = _rerank(query_text, sem)
                print(f"   ✅ Reranking completed: {len(reranked_docs)} documents")
                print(f"   Rerank scores: {scores[:3] if len(scores) >= 3 else scores}")
                return reranked_docs, parent_docs
            except Exception as rerank_error:
                print(f"   ❌ Reranking failed: {rerank_error}")
                print(f"   Falling back to original results")
                return sem, parent_docs
        else:
            print("⏭️ Skipping reranking (disabled)")

        print("✅ query_expansion_retrieve_hybrid completed successfully")
        return sem, parent_docs

    except Exception as e:
        print(f"❌ Error in query_expansion_retrieve_hybrid: {e}")
        import traceback

        traceback.print_exc()
        return [], []
        # Step 8: Get top-k documents
        print(f"🔝 Selecting top-{config.TOP_K} documents...")
        top_indices = hybrid_scores.argsort()[-config.TOP_K :][::-1]
        top_docs = [all_docs[i] for i in top_indices]
        top_scores = hybrid_scores[top_indices]
        print(f"   ✅ Selected {len(top_docs)} documents")

        # Step 9: Save scores
        print("💾 Saving similarity scores...")
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump(top_scores.tolist(), f, ensure_ascii=False)
        print(f"   ✅ Scores saved to: {config.SCORE_PATH}")

        # Step 10: Apply reranking if enabled
        if config.RERANK:
            print("🔄 Applying reranking...")
            try:
                reranked_docs, rerank_scores = _rerank(query_text, top_docs)
                print(f"   ✅ Reranking completed")
                return reranked_docs, summary_texts
            except Exception as rerank_error:
                print(f"   ❌ Reranking failed: {rerank_error}")
                return top_docs, summary_texts
        else:
            print("⏭️ Skipping reranking (disabled)")

        print("✅ summary_mean_retrieve_hybrid completed successfully")
        return top_docs, summary_texts

    except Exception as e:
        print(f"❌ Error in summary_mean_retrieve_hybrid: {e}")
        import traceback

        traceback.print_exc()
        return [], []
