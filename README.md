# STM 보드 & 마이크로컨트롤러 학습 챗봇

임베디드 시스템을 처음 배우는 학생과 초보자를 위한 AI 기반 학습 도우미입니다.

## 프로젝트 소개

### 개요
이 프로젝트는 STM 보드와 마이크로컨트롤러를 학습하는 과정에서 발생하는 다양한 질문들에 대해 즉각적이고 정확한 답변을 제공하는 AI 챗봇 서비스입니다. LLM(Large Language Model)과 RAG(Retrieval-Augmented Generation) 기술을 활용하여 사용자의 질문을 정확하게 이해하고, 관련된 학습 자료를 검색하여 맞춤형 답변을 생성합니다.

### 주요 기능
- 실시간 질의응답: STM 보드 및 마이크로컨트롤러 관련 질문에 대한 즉각적인 답변 제공
- 맞춤형 예제 코드: 사용자의 이해 수준에 맞는 구체적인 코드 예시 제공
- 개념 설명: 복잡한 임베디드 시스템 개념을 쉽게 설명
- 오류 해결 가이드: 일반적인 오류 상황에 대한 해결 방법 제시

### 대상 사용자
- 마이크로컨트롤러 입문자
- 임베디드 프로그래밍을 시작하는 취미 개발자
- STM 보드를 활용하는 교육 기관
- 임베디드 분야 신입 엔지니어

### 기술 스택
- LLM: Upstage Solar Pro (또는 유사 성능 모델)
- 벡터 데이터베이스: Pinecone
- 임베딩 모델: Upstage Solar Embedding
- RAG 파이프라인: LangChain
- 데이터 소스: STM 공식 문서

### 기대효과
- 학습 시간 단축: 필요한 정보를 신속하게 찾아 실습에 집중
- 높은 학습 효율: 예제를 통한 이해도 향상
- 즉각적인 문제 해결: 실습 중 발생하는 오류나 의문점을 빠르게 해소



# 사용법

### 1. 사전 준비

-   사용하고 있는 STM 보드의 모델명과 스펙 확인
-   구현하고자 하는 기능 정리
-   보드의 데이터시트 준비

### 2. 챗봇 이용하기

1.  배포된 사이트에 접속
2.  사용할 모델 선택:
    -   **모델1**: 간단한 질문과 빠른 답변이 필요할 때
    -   **모델2**: 일반적인 설명과 기본적인 사용법 안내가 필요할 때
    -   **모델3**: 상세한 기술 설명과 구체적인 구현 방법이 필요할 때
3.  질문 입력:
    -   보드의 정보
    -   구현하고자 하는 기능
    -   특정 오류나 문제 상황

### 3. 효과적인 사용 팁

-   구체적인 보드 모델명과 사용하고자 하는 기능을 명시하면 더 정확한 답변을 받을 수 있습니다
-   오류 해결이 필요한 경우, 발생한 에러 메시지나 현상을 자세히 설명해주세요

# 로컬 실행 방법

### Backend 설정
1. 레포지토리 클론
   ```bash
   git clone [https://github.com/nOOne-is-hier/STM-Genie.git]
   cd [project-name]/src/backend
   ```

2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. 환경변수 설정
   - `.env` 파일을 backend 디렉토리에 생성
   ```env
   UPSTAGE_API_KEY=[your_upstage_api_key]
   PINECONE_API_KEY=[your_pinecone_api_key] # 개인적인 임베딩 세팅 없이 진행 시 팀장에게 연락하여 받기(pinecone db 저장 결과 받기 위함)
   PINECONE_ENV=us-east1-gcp
   ```

4. 의존성 패키지 설치 및 서버 실행
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
   서버는 기본적으로 http://127.0.0.1:8000 에서 실행됩니다.

### Frontend 설정
1. 새 터미널에서 frontend 디렉토리로 이동
   ```bash
   cd [project-name]/src/frontend
   ```

2. 환경변수 설정
   - `.env.development` 파일을 frontend 디렉토리에 생성
   ```env
   API_ENDPOINT=http://127.0.0.1:8000
   ```

3. 의존성 패키지 설치 및 개발 서버 실행
   ```bash
   npm install
   npm start
   ```

### 웹사이트 접속
- 브라우저에서 http://localhost:1234 접속
- 정상적으로 실행되면 챗봇 서비스 화면을 확인할 수 있습니다

### 문제해결
- Backend 서버 실행 오류 시:
  - Python 버전 확인
  - 가상환경 활성화 상태 확인
  - 필요한 API 키가 올바르게 설정되었는지 확인
- Frontend 실행 오류 시:
  - Node.js 버전 확인
  - npm install 명령어 재실행
  - API_ENDPOINT 환경변수 설정 확인

# 배포 링크
  - https://stm-genie.vercel.app/ (프론트 페이지)
  - 백엔드는 용량 문제로 인해 배포 X
