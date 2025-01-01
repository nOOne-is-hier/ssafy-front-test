import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Pinecone  # 업데이트된 임포트
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain.embeddings.base import Embeddings
from pinecone import Pinecone as PineconeClient
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel  # SciBERT 관련 라이브러리
import torch  # PyTorch
from langchain.prompts.chat import ChatPromptTemplate
import logging

load_dotenv()

# 에러 핸들링
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SciBERT 모델 초기화
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
scibert_model = AutoModel.from_pretrained(model_name)

# Pinecone 및 Upstage 설정
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")  # Pinecone 환경 변수 추가
pinecone_client = PineconeClient(api_key=pinecone_api_key, environment=pinecone_env)

# Upstage 임베딩 설정
embedding_upstage = UpstageEmbeddings(model="embedding-query")
chat_upstage = ChatUpstage()

# 모델 구성 정의
MODEL_CONFIGS = {
    1: {
        "index_name": "model1-index",
        "dimension": 4096,
        "k": 1,
        "threshold": 0.6,
        "prompt": '''{context}

당신은 STM 보드 전문가(STM Genie)입니다. 사용자에게 간단하고 명확한 답변을 제공합니다.

사용자의 질문: "{input}"

질문의 내용을 이해하고, 간결하면서도 정확하게 답변하세요. 불필요한 추가 설명은 생략하십시오.
'''
    },
    2: {
        "index_name": "model2-index",
        "dimension": 4096,
        "k": 3,
        "threshold": None,
        "prompt": '''{context}

당신은 STM 보드와 관련된 모든 질문에 대해 깊이 있는 정보를 제공하는 AI 전문가(STM Genie)입니다.

사용자의 질문: "{input}"

질문과 관련된 모든 중요한 정보를 포괄적으로 답변하세요. 답변은 다음을 포함해야 합니다:
1. 질문의 핵심에 대한 직접적인 응답.
2. 질문과 관련된 추가적인 정보와 배경 설명.
3. 필요 시, 유용한 예시 또는 권장되는 방법.

답변은 논리적이고 명확한 구조로 작성하세요.
'''
    },
    3: {
        "index_name": "model3-index",
        "dimension": 768,
        "k": 1,
        "threshold": 0.8,
        "prompt": '''{context}

당신은 STM 보드와 관련된 전문적인 도움을 제공하는 AI 전문가(STM Genie)입니다.

사용자의 질문: "{input}"

질문에 대해 심층적이고 전문적인 답변을 작성하세요. 답변은 다음을 포함해야 합니다:
1. 질문의 배경을 간략히 요약하여 컨텍스트를 제공.
2. 질문에 대한 정확하고 구체적인 응답.
3. 관련된 사례 연구, 권장 사항, 또는 주의 사항.

사용자가 더 깊은 이해를 할 수 있도록 세부적으로 작성하되, 명확한 논리적 흐름을 유지하세요.
'''
    }
}

# FastAPI 앱 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스에서는 필요한 도메인만 허용하도록 수정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str
    model_id: int
    

from langchain.embeddings.base import Embeddings

class SciBertEmbeddings(Embeddings):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 평균 풀링을 사용하여 임베딩 생성 후 리스트로 변환
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # model_id 유효성 검증
    model_config = MODEL_CONFIGS.get(req.model_id)
    if not model_config:
        raise HTTPException(status_code=400, detail="Invalid model_id")

    # Pinecone 인덱스 선택
    index_name = model_config["index_name"]

    # 임베딩 설정
    if req.model_id == 3:
        # 모델 3의 경우 SciBERT Embeddings 객체 사용
        scibert_embeddings = SciBertEmbeddings()
        pinecone_vectorstore = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=scibert_embeddings
        )
    else:
        # 다른 모델은 Upstage 임베딩 사용
        pinecone_vectorstore = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embedding_upstage
        )

    # 리트리버 설정
    retriever = pinecone_vectorstore.as_retriever(
        search_type="mmr" if req.model_id == 2 else "similarity",
        search_kwargs={"k": model_config["k"]},
    )

    # 문서 검색 (get_relevant_documents 대신 invoke 사용)
    try:
        result_docs = retriever.invoke(req.message)  # 최신 메서드 사용
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Error during retrieval")

    # 검색된 문서의 내용을 하나의 문자열로 결합하여 컨텍스트 생성
    context = "\n\n".join([doc.page_content for doc in result_docs])

    # MODEL_CONFIGS에서 해당 모델의 프롬프트 가져오기
    model_prompt = model_config["prompt"].format(context=context, input=req.message)

    # ChatPromptTemplate을 사용하여 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                model_prompt
            ),
            ("human", "{input}"),
        ]
    )

    # RunnableSequence을 사용하여 체인 구성 (prompt | llm)
    chain = prompt | chat_upstage

    # 체인 호출 시 'context'와 'input'을 모두 제공
    try:
        result = chain.invoke({"context": context, "input": req.message})
    except Exception as e:
        logger.error(f"Error during QA chain invocation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return {"reply": result}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
