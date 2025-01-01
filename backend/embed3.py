'''
## **모델 3: 도메인 특화 및 심층적 질문 대응**

### 설계
1. **Text Splitting**:
   - Chunk Size: `4500 letters`
   - Overlap: `500 letters`

2. **Embedding**:
   - 모델: **SciBERT (allenai/scibert_scivocab_uncased)** (도메인 특화 임베딩).
   - 벡터 차원: `768`.

3. **Retriever**:
   - 반환할 문서 수(`k`): `1`.
   - 유사도 임계값: `0.8`.

4. **Reranker**:
   - 사용하지 않음.

### 목표
- 도메인 특화된 SciBERT를 활용하여 복잡한 질문에 대한 심층적이고 정확한 응답을 보장.
- 긴 청크 설정으로 문서 내 맥락과 세부사항을 최대한 활용.
'''


import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone  # LangChain의 Pinecone 래퍼
from langchain_huggingface import HuggingFaceEmbeddings  # langchain-huggingface 패키지에서 가져옴
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone as PineconeClient, ServerlessSpec  # 공식 Pinecone SDK
import torch
from PyPDF2 import PdfReader

# 에러 메시지 무시
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 문서 로드 함수 (PDF 지원)
def load_document(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return [Document(page_content=text, metadata={})]

# SciBERT 임베딩 생성 함수
def embed_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

if __name__ == '__main__':
    # Pinecone API Key 및 환경 변수 가져오기
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")  # 예: "us-east1-gcp"

    logger.info(f"PINECONE_API_KEY: {'설정됨' if pinecone_api_key else '설정되지 않음'}")
    logger.info(f"PINECONE_ENVIRONMENT: {'설정됨' if pinecone_env else '설정되지 않음'}")

    if not pinecone_api_key or not pinecone_env:
        logger.error("PINECONE_API_KEY 또는 PINECONE_ENVIRONMENT 환경 변수가 설정되어 있지 않습니다.")
        exit(1)

    # Pinecone 클라이언트 초기화
    try:
        pc = PineconeClient(api_key=pinecone_api_key, environment=pinecone_env)
        logger.info("Pinecone 클라이언트 초기화 성공")
    except Exception as e:
        logger.error(f"Pinecone 클라이언트 초기화 중 오류 발생: {e}")
        exit(1)

    # 인덱스 이름과 PDF 경로 설정
    index_name = "model3-index"
    pdf_path = "usart.pdf"

    # Pinecone 인덱스 생성 및 연결
    try:
        if index_name not in pc.list_indexes():
            logger.info(f"Pinecone 인덱스 생성: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,  # SciBERT 기본 차원
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            logger.info(f"Pinecone 인덱스 '{index_name}' 이미 존재합니다.")

        # Pinecone Index 객체 가져오기
        index = pc.Index(index_name)
        logger.info(f"Pinecone Index 객체 연결 성공: {index_name}")
    except Exception as e:
        logger.error(f"Pinecone 인덱스 확인 또는 생성 중 오류 발생: {e}")
        exit(1)

    # Hugging Face SciBERT 설정
    model_name = "allenai/scibert_scivocab_uncased"
    try:
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loading model for {model_name}...")
        scibert_model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"모델 로딩 중 오류 발생: {e}")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scibert_model.to(device)

    logger.info(f"Processing {index_name}...")

    try:
        docs = load_document(pdf_path)
        logger.info(f"문서 로드 완료. 페이지 수: {len(docs)}")
    except Exception as e:
        logger.error(f"문서 로드 중 오류 발생: {e}")
        exit(1)

    try:
        # 문서를 청크로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4500,
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"문서 분할 완료. 분할된 청크 수: {len(splits)}")
    except Exception as e:
        logger.error(f"문서 분할 중 오류 발생: {e}")
        exit(1)

    try:
        # LangChain HuggingFace 임베딩 설정
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        # LangChain의 Pinecone VectorStore 초기화 및 텍스트 추가
        pinecone_vectorstore = Pinecone.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name  # Pinecone Index 이름 전달
        )
        logger.info(f"Pinecone에 벡터 저장 완료: {index_name}")
    except Exception as e:
        logger.error(f"Pinecone에 벡터 저장 중 오류 발생: {e}")
        exit(1)

    logger.info(f"Finished processing {index_name}")
