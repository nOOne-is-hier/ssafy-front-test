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
   - 반환할 문서 수(`k`): `3`.
   - 유사도 임계값: `없음(mmr)`.

4. **Reranker**:
   - **CrossEncoder 기반 Reranker**를 사용.
   - 검색된 상위 `3`개의 청크를 재정렬하여 최적의 청크를 선택.

### 목표
- Chunking 설정을 최적화하여 맥락 유지와 검색 효율성 간 균형 확인.
- Reranker 도입에 따른 검색 품질 및 응답 정확도 향상 평가.
'''


import os
import getpass

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
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset
from sentence_transformers import CrossEncoder


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
embedding_upstage = UpstageEmbeddings(model="embedding-query")
chat_upstage = ChatUpstage(api_key=os.environ.get("UPSTAGE_API_KEY"), model="solar-pro")
prompt_template = PromptTemplate.from_template(
    '''{context}

    당신은 STM 보드와 관련된 모든 질문에 대해 깊이 있는 정보를 제공하는 AI 전문가(STM Genie)입니다.

    사용자의 질문: "{input}"

    질문과 관련된 모든 중요한 정보를 포괄적으로 답변하세요. 답변은 다음을 포함해야 합니다:
    1. 질문의 핵심에 대한 직접적인 응답.
    2. 질문과 관련된 추가적인 정보와 배경 설명.
    3. 필요 시, 유용한 예시 또는 권장되는 방법.

    답변은 논리적이고 명확한 구조로 작성하세요.
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
        embeddings=embedding_upstage,
    )
    return result

def fill_data(data, question, retr):
    results = retr.invoke(question)
    context = [doc.page_content for doc in results]
    print((f"Applying reranker for model_id=2 with query: {question}"))
    
    # Reranker 입력 형식: (query, doc)
    pairs = [(question, text) for text in context]
    reranked_scores = reranker.predict(pairs)
    
    # 문서와 점수를 함께 묶기
    scored_docs = list(zip(results, reranked_scores))

    # 점수를 기준으로 내림차순 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 최종 상위 3개 문서만 선택
    result_docs = [doc for doc, score in scored_docs[:3]]
    
    final_context = "\n\n".join([doc.page_content for doc in result_docs])

    chain = prompt_template | chat_upstage | StrOutputParser()
    answer = chain.invoke({"context": final_context, "input": question})

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
index_name = "model2"
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

# Store in Pinecone
# 이미 Pinecone의 index_name에 chank 구성대로 저장되어 있기 때문에 chunk 관련 구성 X
pinecone_vectorstore = LangChainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding_upstage
)

retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={"k": 10}
)

# 평가 질문 리스트
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