import os
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

load_dotenv()

embedding_upstage = UpstageEmbeddings(model="embedding-passage")

test_vec = embedding_upstage.embed_query("Hello world!")
print("Embedding dimension:", len(test_vec))

# SciBERT 모델 초기화
embedding_scibert = UpstageEmbeddings(model="scibert")

# 모델 차원 출력
test_vec = embedding_scibert.embed_query("Hello world!")
print("Embedding dimension:", len(test_vec))