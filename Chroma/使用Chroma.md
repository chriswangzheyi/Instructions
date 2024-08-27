# 使用Chroma



### 初始化 Chroma 客户端

```python
client = chromadb.PersistentClient(path="/path/to/save/to")
```



### 启动 Chroma 服务器

```
chroma run --path /db_path
```



### 连接到服务器

```python
import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```





### 使用集合（Collections）

```python
collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
```





### 修改集合的嵌入函数

```python
collection = client.create_collection(
    name="collection_name",
    metadata={"hnsw:space": "cosine"}  # l2 是默认选项
)
```





### 向集合添加数据

```python
collection.add(
    documents=["lorem ipsum...", "doc2", "doc3", ...],
    metadatas=[{"chapter": "3", "verse": "16"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```





### 查询集合

```python
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains": "search_string"}
)
```





### 指定返回数据

```python
collection.get(include=["documents"])
collection.query(include=["documents"])
```





### 使用过滤器

```python
{"metadata_field": {"$eq": "search_string"}}
{"$contains": "search_string"}
```





### 更新集合中的数据

```python
collection.update(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```





### 从集合中删除数据

```python
collection.delete(ids=["id1", "id2", "id3", ...], where={"chapter": "20"})
```





## Demo



```python
import chromadb

# 初始化 Chroma 客户端
client = chromadb.PersistentClient(path="/path/to/save/to")

# 创建集合
collection = client.create_collection(name="news_articles", embedding_function=emb_fn)

# 添加新闻文章
collection.add(
    documents=["Article content 1...", "Article content 2...", "Article content 3...", ...],
    metadatas=[{"title": "Title 1", "author": "Author 1", "date": "2021-01-01"}, ...],
    ids=["id1", "id2", "id3", ...]
)

# 查询相关的新闻文章
results = collection.query(
    query_embeddings=[[11.1, 12.1, 13.1], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains": "search_string"}
)

# 打印查询结果
for result in results:
    print(result)
```

