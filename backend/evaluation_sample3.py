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


# 모델 및 프롬프트 구성
scibert_embeddings = SciBertEmbeddings()
chat_upstage = ChatUpstage(api_key=os.environ.get("UPSTAGE_API_KEY"), model="solar-pro")
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

# Reranker 모델 초기화 (모델 2용)
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 적절한 CrossEncoder 모델 선택
reranker = CrossEncoder(reranker_model_name)

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

# Store in Pinecone
# 이미 Pinecone의 index_name에 chank 구성대로 저장되어 있기 때문에 chunk 관련 구성 X
pinecone_vectorstore = LangChainPinecone.from_existing_index(
    index_name=index_name,
    embedding=scibert_embeddings
)

retriever = pinecone_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 3}
)

# 평가 질문 리스트
questions = [
    "STM32F1 마이크로컨트롤러에서 타이머(Timer)를 설정하려면 어떤 레지스터를 주로 사용하나요?",
    "STM32F1의 GPIO에서 외부 인터럽트를 활성화하려면 어떤 설정이 필요한가요?",
    "RTC(Real-Time Clock) 기능을 활성화하기 위한 기본 설정 과정은 무엇인가요?",
    "저전력 모드(Low Power Mode)로 전환하는 과정과 관련된 레지스터는 무엇인가요?"
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