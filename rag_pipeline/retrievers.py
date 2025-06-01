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
    """Cross-Encoder 점수로 재정렬"""
    passages = [d.page_content for d in docs]
    pairs = [[query, passage] for passage in passages]
    scores = reranker.score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda t: t[1], reverse=True)
    docs_sorted, scores_sorted = zip(*ranked)
    return list(docs_sorted), list(scores_sorted)


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
        hydes = []
        for _ in range(5):
            try:
                hypo_doc = utils.generate_hyde_document(query_text)
                embedding = model.encode(hypo_doc, normalize_embeddings=True)
                hydes.append(embedding)
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
            q_vecs = model.encode([query_text], normalize_embeddings=True)[0]
        search_query = None  # We already have the query vector
    elif config.RETRIEVAL_TYPE == "summary":
        search_query = utils.generate_summary(query_text)
    else:
        search_query = query_text

    # Calculate similarity scores
    doc_vecs = model.encode(texts, normalize_embeddings=True)

    if search_query is not None:
        q_vecs = model.encode([search_query], normalize_embeddings=True)[0]

    # Vector similarity scores
    cos_sim = util.cos_sim(q_vecs, doc_vecs)[0].float().cpu().numpy()

    if hybrid_weight_check:
        # Hybrid retrieval: combine vector similarity with BM25

        # Tokenize documents for BM25
        tokenized_docs = [doc.split() for doc in texts]
        bm25 = BM25Okapi(tokenized_docs)

        # Get BM25 scores
        if config.RETRIEVAL_TYPE == "summary":
            bm25_query = utils.generate_summary(query_text).split()
        elif config.RETRIEVAL_TYPE == "hyde":
            # Generate HyDE documents for BM25 as well
            hyde_docs_for_bm25 = []
            for _ in range(5):
                try:
                    hypo_doc = utils.generate_hyde_document(query_text)
                    hyde_docs_for_bm25.append(hypo_doc)
                except Exception as e:
                    print(f"Error generating HyDE document for BM25: {e}")
                    continue

            if hyde_docs_for_bm25:
                # Combine all HyDE documents for BM25 query
                combined_hyde_text = " ".join(hyde_docs_for_bm25)
                bm25_query = combined_hyde_text.split()
            else:
                # Fallback to original query if HyDE generation fails
                bm25_query = query_text.split()
        else:
            bm25_query = query_text.split()

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
        for _ in range(5):
            try:
                hypo_doc = utils.generate_hyde_document(query_text)
                embedding = model.encode(hypo_doc, normalize_embeddings=True)
                hydes.append(embedding)
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
            q_vecs = model.encode([query_text], normalize_embeddings=True)[0]
        search_query = None  # We already have the query vector
    elif config.RETRIEVAL_TYPE == "summary":
        search_query = utils.generate_summary(query_text)
    else:
        search_query = query_text

    # Calculate similarity scores
    doc_vecs = model.encode(texts, normalize_embeddings=True)

    if search_query is not None:
        q_vecs = model.encode([search_query], normalize_embeddings=True)[0]

    # Vector similarity scores
    cos_sim = util.cos_sim(q_vecs, doc_vecs)[0].float().cpu().numpy()

    if hybrid_weight_check:
        # Hybrid retrieval: combine vector similarity with BM25

        # Tokenize documents for BM25
        tokenized_docs = [doc.split() for doc in texts]
        bm25 = BM25Okapi(tokenized_docs)

        # Get BM25 scores
        if config.RETRIEVAL_TYPE == "summary":
            bm25_query = utils.generate_summary(query_text).split()
        elif config.RETRIEVAL_TYPE == "hyde":
            # Generate HyDE documents for BM25 as well
            hyde_docs_for_bm25 = []
            for _ in range(5):
                try:
                    hypo_doc = utils.generate_hyde_document(query_text)
                    hyde_docs_for_bm25.append(hypo_doc)
                except Exception as e:
                    print(f"Error generating HyDE document for BM25: {e}")
                    continue

            if hyde_docs_for_bm25:
                # Combine all HyDE documents for BM25 query
                combined_hyde_text = " ".join(hyde_docs_for_bm25)
                bm25_query = combined_hyde_text.split()
            else:
                # Fallback to original query if HyDE generation fails
                bm25_query = query_text.split()
        else:
            bm25_query = query_text.split()

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

    return best_docs


def vectordb_retrieve(query: HumanMessage | str) -> List[Document]:
    """기본 벡터 DB 검색 - 반환값 일관성 수정"""
    try:
        query_text = query.content if hasattr(query, "content") else query

        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        query_emb = model.encode(
            query_text,
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

        # 출력 디렉토리 생성 확인
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(config.SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(cos_sim.tolist(), f, ensure_ascii=False)

        # 조건부로 reranking 적용
        if config.RERANK:
            reranked_docs, scores = _rerank(query_text, sem)
            return reranked_docs

        return sem

    except Exception as e:
        print(f"Error in vectordb_retrieve: {e}")
        return []


def vectordb_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float] = [0.5, 0.5]
) -> List[Document]:
    """FAISS + BM25 하이브리드 검색 - 반환값 일관성 수정"""
    try:
        query_text = query.content if hasattr(query, "content") else query

        vectordb = FAISS.load_local(
            config.CONTENT_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": config.TOP_K})

        all_docs = list(vectordb.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = config.TOP_K

        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=weights,
        )

        sem = ensemble_retriever.get_relevant_documents(query_text)

        reranked_docs, scores = _rerank(query_text, sem)

        # 출력 디렉토리 생성 확인
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump([float(s) for s in scores], f, ensure_ascii=False)
        return reranked_docs

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

    # 조건부로 reranking 적용
    if config.RERANK:
        reranked_docs, scores = _rerank(query_text, sem)
        return reranked_docs, query_explanation

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

    # 하이브리드 검색 설정
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": config.TOP_K})

    all_docs = list(vectordb.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.TOP_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights,
    )

    sem = ensemble_retriever.get_relevant_documents(query_explanation)

    reranked_docs, scores = _rerank(query_explanation, sem)
    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump([float(s) for s in scores], f, ensure_ascii=False)
    return reranked_docs, query_explanation


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

        if config.RERANK:
            reranked_docs, scores = _rerank(query_text, sem)
            return reranked_docs, hypo_docs

        return sem, hypo_docs

    except Exception as e:
        print(f"Error in hyde_retrieve: {e}")
        return [], []


def hyde_hybrid_retrieve(
    query: HumanMessage | str, weights: List[float] = [0.5, 0.5]
) -> Tuple[List[Document], str]:
    """HyDE + 하이브리드 검색"""
    query_text = query.content if hasattr(query, "content") else query

    vectordb = FAISS.load_local(
        config.CONTENT_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    hydes: List[np.ndarray] = []
    hypo_docs: List[str] = []

    # HyDE 문서 생성
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

    # 하이브리드 검색 설정
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": config.TOP_K})

    all_docs = list(vectordb.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.TOP_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights,
    )

    # 가설 문서로 검색
    combined_hypo_doc = " ".join(hypo_docs)
    sem = ensemble_retriever.get_relevant_documents(combined_hypo_doc)

    # reranking
    reranked_docs, scores = _rerank(query_text, sem)
    with open(config.SCORE_PATH, "w", encoding="utf-8") as f:
        json.dump([float(s) for s in scores], f, ensure_ascii=False)

    return reranked_docs, hypo_docs
