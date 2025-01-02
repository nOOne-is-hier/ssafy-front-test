'''
## **모델 1: Baseline (기본 설정)**

### 설계
1. **Text Splitting**:
   - Chunk Size: `1200 letters`
   - Overlap: `100 letters`

2. **Embedding**:
   - 모델: **embedding-query(Upstage Solar Embedding)** (범용 임베딩).
   - 벡터 차원: `4096(수치 수정 불가)`.

3. **Retriever**:
   - 반환할 문서 수(`k`): `2`.
   - 유사도 임계값: `0.6`.

4. **Reranker**:
   - 사용하지 않음.

### 목표
- 단순한 RAG 파이프라인의 기준 성능 측정.
- Chunking 및 기본 설정의 효율성 확인.
'''


import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Load Upstage API Key
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# Pinecone API Key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Index and PDF Path
# index_name = "model1-index"
# pdf_path = "usart.pdf"

# for real service
index_name = "model1"
pdf_path = "raw_data.pdf"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,  # Upstage Solar Embedding 차원
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print(f"Processing {index_name}...")

# Parse document
document_parse_loader = UpstageDocumentParseLoader(
    pdf_path,
    output_format='html',  # 결과물 형태 : HTML
    coordinates=False)  # 이미지 OCR 좌표계 가지고 오지 않기

docs = document_parse_loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # 모델 1에 따른 청크 크기
    chunk_overlap=100  # 모델 1에 따른 오버랩 크기
)

splits = text_splitter.split_documents(docs)

# Store in Pinecone
PineconeVectorStore.from_documents(
    splits, embedding_upstage, index_name=index_name
)

print(f"Finished processing {index_name}")