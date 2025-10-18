from typing import Any, List
import os
import faiss
import numpy as np
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
print("load_dotenv")
load_dotenv()

# 初始化 MCP Server
mcp = FastMCP("rag")

# 向量索引（内存版 FAISS）
_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(1536)  #创建了一个FAISS向量索引，IndexFlatL2 是FAISS中最基础的索引类型，1536 是向量维度
_docs: List[str] = []  #存储原始文档内容的列表

# OpenAI API（用于生成嵌入）
openai = OpenAI()

#将文本转换为数值向量
async def embed_text(texts: List[str]) -> np.ndarray: #async是异步非阻塞
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        encoding_format="float"
    )
    return np.array([d.embedding for d in resp.data], dtype='float32')

@mcp.tool()  #将这个函数注册为MCP工具
async def index_docs(docs: List[str]) -> str: 
    """将一批文档加入索引。
    Args:
        docs: 文本列表
    """
    global _index, _docs
    embeddings = await embed_text(docs)
    _index.add(embeddings.astype('float32'))
    _docs.extend(docs)
    return f"已索引 {len(docs)} 篇文档，总文档数：{len(_docs)}"

@mcp.tool()
async def retrieve_docs(query: str, top_k: int = 3) -> str:
    """检索最相关文档片段。
    Args:
        query: 用户查询
        top_k: 返回的文档数
    """
    q_emb = await embed_text([query])
    D, I = _index.search(q_emb.astype('float32'), top_k)  #D - 距离数组，表示相似度（距离越小越相似），I - 索引数组，表示最相似文档的索引位置
    results = [f"[{i}] {_docs[i]}" for i in I[0] if i < len(_docs)]
    return "\n\n".join(results) if results else "未检索到相关文档。"

if __name__ == "__main__":
    mcp.run(transport="stdio")  #transport="stdio" - 使用标准输入输出作为通信方式
    # mcp.run(transport="tcp", host="127.0.0.1", port=8000)
