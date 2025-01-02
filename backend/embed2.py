'''
## **모델 2: Chunking 최적화 및 Reranker 도입**

### 설계
1. **Text Splitting**:
   - Chunk Size: `800 letters`
   - Overlap: `200 letters`

2. **Embedding**:
   - 모델: **embedding-query(Upstage Solar Embedding)** (범용 임베딩).
   - 벡터 차원: `4096(수치 수정 불가)`.

3. **Retriever**:
   - 반환할 문서 수(`k`): `3 of 10`.
   - 유사도 임계값: `없음(mmr)`.

4. **Reranker**:
   - **CrossEncoder 기반 Reranker**를 사용.
   - 검색된 상위 `3`개의 청크를 재정렬하여 최적의 청크를 선택.

### 목표
- Chunking 설정을 최적화하여 맥락 유지와 검색 효율성 간 균형 확인.
- Reranker 도입에 따른 검색 품질 및 응답 정확도 향상 평가.
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
embedding_upstage = UpstageEmbeddings(model="embedding-query")  # Upstage Solar Embedding

# Pinecone API Key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Index and PDF Path
# index_name = "model2-index"  # 변경된 모델 2에 맞는 인덱스 이름
# pdf_path = "usart.pdf"

# for real service
index_name = "model2"  # 변경된 모델 2에 맞는 인덱스 이름
pdf_path = "raw_data.pdf"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,  # 모델 2의 벡터 차원 설정
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print(f"Processing {index_name}...")

# Parse document
document_parse_loader = UpstageDocumentParseLoader(
    pdf_path,
    output_format='html',  # 결과물 형태 : HTML
    coordinates=False  # 이미지 OCR 좌표계 가지고 오지 않기
)

# Load documents
docs = document_parse_loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # 모델 2의 Chunk Size
    chunk_overlap=200  # 모델 2의 Overlap
)

# Split documents
splits = text_splitter.split_documents(docs)

# Store in Pinecone
PineconeVectorStore.from_documents(
    splits, embedding_upstage, index_name=index_name
)

print(f"Finished processing {index_name}")
