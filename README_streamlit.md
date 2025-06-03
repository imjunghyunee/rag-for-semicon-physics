# Semiconductor Physics RAG System - Streamlit Demo

이 Streamlit 앱은 Semiconductor Physics RAG 시스템의 완전한 시연을 제공합니다.

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements_streamlit.txt
```

### 2. 앱 실행

```bash
# 방법 1: 직접 실행
streamlit run streamlit_app.py

# 방법 2: 래퍼 스크립트 사용
python run_streamlit.py
```

### 3. 브라우저에서 접속

앱이 실행되면 자동으로 브라우저가 열리며, `http://localhost:8501`에서 접속할 수 있습니다.

## 📋 주요 기능

### 1. 질문 복잡도 자동 판정

-   **Simple**: 직접 검색으로 답변 가능한 질문
-   **Complex**: 다단계 추론이 필요한 질문

### 2. Simple 질문 처리

-   직접 벡터 검색
-   관련성 체크
-   최종 답변 생성

### 3. Complex 질문 처리 (Plan-and-Execute)

-   **실시간 진행 상황 표시**
-   **단계별 계획 수립**
-   **중간 결과 확인**
-   **동적 재계획**
-   **최종 답변 통합**

## 🎯 화면 구성

### 메인 화면

1. **헤더**: 시스템 제목
2. **채팅 인터페이스**: 질문 입력 및 대화 이력
3. **사이드바**: 시스템 설정 정보

### Simple 질문 처리 화면

1. 복잡도 판정 결과
2. 검색 진행 상황
3. 검색된 컨텍스트 (접을 수 있음)
4. 최종 답변

### Complex 질문 처리 화면

1. 복잡도 판정 결과
2. **Plan-and-Execute 단계별 진행**:
    - 초기 계획 수립
    - 각 단계 실행 (검색 + 중간 결과)
    - 진행 상황 평가 및 재계획
    - 최종 답변 생성
3. **과정 요약** (접을 수 있음)
4. 최종 답변

## 🔧 설정

시스템 설정은 `rag_pipeline/config.py`에서 관리됩니다:

-   `MAX_PLAN_STEPS`: 최대 계획 단계 수
-   `RETRIEVAL_TYPE`: 검색 방식 (original_query, hyde, summary)
-   `TOP_K`: 검색할 문서 수
-   `OPENAI_MODEL`: 사용할 LLM 모델

## 🎨 UI 특징

### Perplexity 스타일 인터페이스

-   **실시간 단계별 진행**: 각 단계가 순차적으로 표시
-   **동적 콘텐츠 업데이트**: 이전 단계는 사라지고 현재 단계가 강조
-   **접을 수 있는 상세 정보**: 과정 요약 및 중간 결과
-   **깔끔한 디자인**: 단계별 구분과 색상 코딩

### 반응형 레이아웃

-   **2열 구성**: 컨텍스트와 결과를 나란히 표시
-   **프로그레스 바**: 처리 진행 상황 시각화
-   **스피너**: 실시간 로딩 상태 표시

## 📝 사용 예시

### Simple 질문 예시

-   "What is the bandgap of silicon?"
-   "Define electron mobility"
-   "What is the formula for drift velocity?"

### Complex 질문 예시

-   "How does temperature affect both carrier concentration and mobility in silicon devices?"
-   "Compare the effects of different doping strategies on transistor performance"
-   "Analyze the trade-offs between speed and power consumption in CMOS design"

## 🚨 문제 해결

### 일반적인 문제

1. **포트 충돌**: 8501 포트가 사용 중인 경우 다른 포트 사용
2. **의존성 오류**: requirements_streamlit.txt 재설치
3. **모델 로딩 실패**: OpenAI API 키 확인

### 디버깅

-   터미널에서 실행하여 로그 확인
-   브라우저 개발자 도구에서 에러 확인
-   사이드바에서 "Reset Session" 버튼 사용

## 🎯 발표 팁

1. **Simple vs Complex 비교**: 같은 주제의 간단한 질문과 복잡한 질문을 연속으로 시연
2. **실시간 처리 강조**: Plan-and-Execute의 단계별 진행 과정 설명
3. **중간 결과 활용**: 각 단계에서 어떤 정보가 수집되고 활용되는지 설명
4. **시스템 유연성**: 다양한 유형의 질문에 대한 적응력 시연
