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







## Demo2





```
import chromadb
chroma_client  = chromadb.PersistentClient(path="/Users/zheyiwang/Documents/persist/")


collection = chroma_client.create_collection(name="Students")


student_info = """
Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.
"""

club_info = """
The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.
"""

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""


collection.add(
    documents = [student_info, club_info, university_info],
    metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
    ids = ["id1", "id2", "id3"]
)


results = collection.query(
    query_texts=["What is the student name?"],
    n_results=2
)


print(results)

```



打印结果：



```
{'ids': [['id1', 'id2']], 
'distances': [[1.2947064596381113, 1.3954305347392126]],
'metadatas': [[{'source': 'student info'}, {'source': 'club info'}]], 
'embeddings': None, 
'documents': [['\nAlexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,\nis a member of the programming and chess clubs who enjoys pizza, swimming, and hiking\nin her free time in hopes of working at a tech company after graduating from the University of Washington.\n', "\nThe university chess club provides an outlet for students to come together and enjoy playing\nthe classic strategy game of chess. Members of all skill levels are welcome, from beginners learning\nthe rules to experienced tournament players. The club typically meets a few times per week to play casual games,\nparticipate in tournaments, analyze famous chess matches, and improve members' skills.\n"]], 'uris': None, 'data': None, 'included': ['metadatas', 
'documents', 
'distances']}

```

