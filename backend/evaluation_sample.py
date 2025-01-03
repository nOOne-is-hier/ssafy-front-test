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
   - 반환할 문서 수(`k`): `1`.
   - 유사도 임계값: `0.6`.

4. **Reranker**:
   - 사용하지 않음.

### 목표
- 단순한 RAG 파이프라인의 기준 성능 측정.
- Chunking 및 기본 설정의 효율성 확인.
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
chat_upstage = ChatUpstage(api_key=os.environ.get("UPSTAGE_API_KEY"), model="solar-pro")
prompt_template = PromptTemplate.from_template(
    '''{context}

    당신은 STM 보드 전문가(STM Genie)입니다. 사용자에게 간단하고 명확한 답변을 제공합니다.

    사용자의 질문: "{input}"

    질문의 내용을 이해하고, 간결하면서도 정확하게 답변하세요. 불필요한 추가 설명은 생략하십시오.
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
        embeddings=embedding_upstage,
    )
    return result

def fill_data(data, question, retr):
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

# Pinecone 클라이언트 인스턴스화
pinecone_client = Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

# Index and PDF Path
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

# Store in Pinecone
# 이미 Pinecone의 index_name에 chank 구성대로 저장되어 있기 때문에 chunk 관련 구성 X
pinecone_vectorstore = LangChainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding_upstage
)

retriever = pinecone_vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 2}
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