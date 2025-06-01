#!/usr/bin/env python3
"""
기존 벡터 데이터베이스 확인 및 테스트
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def check_existing_vectordb():
    """기존 벡터 DB 상태 확인"""
    
    # 가능한 경로들 확인
    possible_paths = [
        "./vectordb/faiss",
        "./vectordb/jina/neamen_content", 
        "./vectordb"
    ]
    
    print("🔍 Checking existing vector database...")
    
    for path in possible_paths:
        db_path = Path(path)
        print(f"\nChecking: {db_path}")
        
        if db_path.exists():
            print(f"✅ Directory exists: {db_path}")
            
            # 내부 파일 확인
            files = list(db_path.glob("*"))
            print(f"📁 Contents: {[f.name for f in files]}")
            
            # FAISS 필수 파일 확인
            faiss_files = ["index.faiss", "index.pkl"]
            missing_files = []
            
            for required_file in faiss_files:
                file_path = db_path / required_file
                if file_path.exists():
                    print(f"✅ {required_file}: {file_path.stat().st_size} bytes")
                else:
                    print(f"❌ {required_file}: missing")
                    missing_files.append(required_file)
            
            # 벡터 DB 로드 테스트
            if not missing_files:
                try:
                    print(f"🧪 Testing vector DB load from {db_path}...")
                    
                    # 간단한 임베딩 모델로 테스트
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    vectordb = FAISS.load_local(
                        str(db_path),
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # 간단한 검색 테스트
                    test_results = vectordb.similarity_search("인공지능", k=1)
                    
                    print(f"✅ Vector DB loaded successfully!")
                    print(f"📊 Total documents: {vectordb.index.ntotal}")
                    if test_results:
                        print(f"📝 Sample content: {test_results[0].page_content[:100]}...")
                    
                    return str(db_path)
                    
                except Exception as e:
                    print(f"❌ Failed to load vector DB: {e}")
            else:
                print(f"❌ Missing required files: {missing_files}")
        else:
            print(f"❌ Directory not found: {db_path}")
    
    print("\n❌ No valid vector database found!")
    return None

def check_token_info():
    """token_info.json 파일 확인"""
    token_file = Path("./vectordb/token_info.json")
    
    if token_file.exists():
        print(f"\n📋 Found token info: {token_file}")
        
        import json
        with open(token_file, 'r', encoding='utf-8') as f:
            token_data = json.load(f)
        
        print(f"📊 Token info contains {len(token_data)} chunks")
        
        # 샘플 출력
        if token_data:
            sample = token_data[0]
            print(f"📝 Sample chunk: {sample.get('chunk_index', 'N/A')}")
            print(f"🔢 Token count: {sample.get('token_count', 'N/A')}")
            print(f"📂 Metadata: {sample.get('metadata', {})}")
    else:
        print(f"\n❌ Token info not found: {token_file}")

if __name__ == "__main__":
    print("🚀 Checking existing vector database setup...")
    
    # 벡터 DB 확인
    valid_path = check_existing_vectordb()
    
    # 토큰 정보 확인  
    check_token_info()
    
    if valid_path:
        print(f"\n✅ Recommendation: Update config.py CONTENT_DB_PATH to '{valid_path}'")
        print(f"💡 Then test with: python main.py --query '인공지능의 정의는 무엇인가?'")
    else:
        print(f"\n❌ Need to create new vector database")
        print(f"💡 Run: python create_vectordb.py")