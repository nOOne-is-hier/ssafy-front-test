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
import getpass
import torch

from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset
from sentence_transformers import CrossEncoder

class SciBertEmbeddings(Embeddings):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 모델을 평가 모드로 설정
        print(f"SciBERT model hidden size: {self.model.config.hidden_size}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                print(f"Output last_hidden_state shape for text {i}: {outputs.last_hidden_state.shape}")
                # 평균 풀링: [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                print(f"SciBERT Embedding {i} dimension: {len(embedding)}")
                embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.embed_documents([text])[0]
        print(f"SciBERT Embedding Query dimension: {len(embedding)}")
        return embedding

try:
    if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
        os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter your Upstage API key: ")

    print("API key has been set successfully.")
except:
    print("Something wrong with your API KEY. Check your API Console again.")

try:
    if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
        os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter your Upstage API key: ")

    print("API key has been set successfully.")
except:
    print("Something wrong with your API KEY. Check your API Console again.")

scibert_embeddings = SciBertEmbeddings()
chat_upstage = ChatUpstage(api_key=os.environ.get("UPSTAGE_API_KEY"), model="solar-pro")

# Reranker 모델 초기화 (모델 2용)
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 적절한 CrossEncoder 모델 선택
reranker = CrossEncoder(reranker_model_name)

prompt_template = PromptTemplate.from_template(
    '''{context}

    당신은 STM 보드와 관련된 전문적인 도움을 제공하는 AI 전문가(STM Genie)입니다.

    사용자의 질문: "{input}"

    질문에 대해 심층적이고 전문적인 답변을 작성하세요. 답변은 다음을 포함해야 합니다:
    1. 질문의 배경을 간략히 요약하여 컨텍스트를 제공.
    2. 질문에 대한 정확하고 구체적인 응답.
    3. 관련된 사례 연구, 권장 사항, 또는 주의 사항.

    사용자가 더 깊은 이해를 할 수 있도록 세부적으로 작성하되, 명확한 논리적 흐름을 유지하세요.
    '''
)

def ragas_evalate(dataset):
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=chat_upstage,
        embeddings=scibert_embeddings,
    )
    return result

def fill_data(data, question, retr):
    print(f"Using single query text for model_id=3: {question}")
    results = retr.invoke(question)
    context = [doc.page_content for doc in results]

    chain = prompt_template | chat_upstage | StrOutputParser()
    answer = chain.invoke({"context": context, "input": question})

    data["question"].append(question)
    data["answer"].append(answer)
    data["contexts"].append(context)
    data["ground_truth"].append("")

load_dotenv()

# Load Upstage API Key
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# Pinecone API Key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

pinecone_env = os.environ.get("PINECONE_ENV")

pinecone_client = Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

# Index and PDF Path
index_name = "model3"
pdf_path = "raw_data.pdf"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Upstage Solar Embedding 차원
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print(f"Processing {index_name}...")

# Parse document
# document_parse_loader = UpstageDocumentParseLoader(
#     pdf_path,
#     output_format='html',  # 결과물 형태 : HTML
#     coordinates=False)  # 이미지 OCR 좌표계 가지고 오지 않기

# docs = document_parse_loader.load()

# Split the document into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1200,  # 모델 1에 따른 청크 크기
#     chunk_overlap=100  # 모델 1에 따른 오버랩 크기
# )

# splits = text_splitter.split_documents(docs)

# Store in Pinecone
pinecone_vectorstore = LangChainPinecone.from_existing_index(
    index_name=index_name,
    embedding=scibert_embeddings
)

retriever = pinecone_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 3}
)

questions = [
    "STM32F1 시리즈에서 클록(CLOCK) 소스는 어떤 종류가 있나요?",
    "NVIC(중첩 벡터 인터럽트 컨트롤러)의 역할은 무엇인가요?",
    "STM32F1의 클록 트리에서 PLL(Phase-Locked Loop)을 설정하는 방법은 무엇인가요?",
    "USART 통신에서 바이트를 송신하기 위해 설정해야 하는 주요 레지스터는 무엇인가요?",
    "STM32F1에서 DMA와 ADC(Analog to Digital Converter)를 결합하여 데이터를 처리하는 방법은 무엇인가요?",
    "STM32F1의 클록 트리에서 USB PLL 설정이 필요한 이유와 관련 레지스터는 무엇인가요?",
    "STM32F1의 인터럽트 우선순위를 변경하려면 어떤 레지스터를 수정해야 하나요?"
]

ex1_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for question in questions:
    fill_data(ex1_data, question, retriever)
    
ex1_dataset = Dataset.from_dict(ex1_data)
print(ragas_evalate(ex1_dataset))

print(f"Finished processing {index_name}")