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
_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(1536)
_docs: List[str] = []

# ----- 替换为阿里云百炼 ------
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

async def embed_text(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
        dimensions=1536,# 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    return np.array([d.embedding for d in resp.data], dtype='float32')

# ----- 替换为阿里云百炼 ------


@mcp.tool()
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
    D, I = _index.search(q_emb.astype('float32'), top_k)
    results = [f"[{i}] {_docs[i]}" for i in I[0] if i < len(_docs)]
    return "\n\n".join(results) if results else "未检索到相关文档。"

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # mcp.run(transport="tcp", host="127.0.0.1", port=8000)
