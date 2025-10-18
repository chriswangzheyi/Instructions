## 启动（基于 deepseek 大模型 + 阿里百炼嵌入模型）

### 启动 server 端

``` SH
cd mcp-in-action/02-mcp-rag/rag-server
uv run server-ali.py
```
执行结果如下：

``` SH
load_dotenv
```

### 启动 client 端
``` SH
cd mcp-in-action/02-mcp-rag/rag-client
uv run client-v3-deepseek-ali.py ../rag-server/server-ali.py
```
执行结果如下：

``` SH
>>> 开始初始化 RAG 系统
[06/19/25 15:34:54] INFO     Processing request of type ListToolsRequest                                                  server.py:534
可用工具： ['index_docs', 'retrieve_docs']
>>> 系统连接成功
>>> 正在索引医学文档...
                    INFO     Processing request of type CallToolRequest                                                   server.py:534
[06/19/25 15:34:55] INFO     HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings "HTTP/1.1  _client.py:1025
                             200 OK"                                                                                                   
>>> 文档索引完成

请输入您要查询的医学问题（输入'退出'结束查询）：
```