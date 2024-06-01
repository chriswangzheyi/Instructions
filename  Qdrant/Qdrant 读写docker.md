# Qdrant 读写docker



## 依赖

```python
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
```



## 存数据



```python
self.vectorstore = Qdrant.from_documents(
    documents=all_splits,  # 以分块的文档
    embedding=OpenAIEmbeddings(),  # 用OpenAI的Embedding Model做嵌入
    url="http://127.0.0.1:6333/",  # in-memory 存储
    prefer_grpc=False,
    collection_name="docs", )  # 指定collection_name
```

prefer_grpc 要设置为True，如果不是6333端口。





## 读数据



```python
client = QdrantClient(url="http://127.0.0.1:6333")

qdrant = Qdrant(
    client=client, collection_name="docs",
    # embedding_function=embeddings.embed_query,
    embeddings=OpenAIEmbeddings())
```

