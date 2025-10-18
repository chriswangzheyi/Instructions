# source .venv/bin/activate
# uv run client.py ../rag-server/server.py
import sys, asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class RagClient:
    def __init__(self):
        self.session = None
        self.transport = None   # 用来保存 stdio_client 的上下文管理器
        self.anthropic = Anthropic()

    async def connect(self, server_script: str):
        # 1) 构造参数对象
        params = StdioServerParameters(
            # command="uv",
            command="/home/huangj2/Documents/mcp-in-action/02-mcp-rag/rag-server/.venv/bin/python",
            # command="/home/huangj2/Documents/mcp-in-action/02-mcp-rag/rag-server/.venv/bin/uv",
            # args=["run", server_script],
            args=["-u",server_script],
        )
        # 2) 保存上下文管理器
        self.transport = stdio_client(params)
        # 3) 进入上下文，拿到 stdio, write
        self.stdio, self.write = await self.transport.__aenter__()

        # 4) 初始化 MCP 会话
        self.session = await ClientSession(self.stdio, self.write).__aenter__()
        await self.session.initialize() # 必须要有，否则无法初始化对话
        resp = await self.session.list_tools()
        print("可用工具：", [t.name for t in resp.tools])

    async def query(self, q: str):
        result = await self.session.call_tool(
            "retrieve_docs",
            {"query": q, "top_k": 3}
        )
        print("检索结果：\n", result)

    async def close(self):
        # 先关闭 MCP 会话
        await self.session.__aexit__(None, None, None)
        # 再退出 stdio_client 上下文
        await self.transport.__aexit__(None, None, None)

async def main():
    print(">>> start main")              # 1
    if len(sys.argv) < 2:
        print("用法: python client.py <server.py 路径>")
        return

    client = RagClient()
    print(">>> before connect")          # 2
    await client.connect(sys.argv[1])
    print(">>> after connect")           # 3

    print(">>> calling index_docs")      # 4
    res = await client.session.call_tool(
        "index_docs",
        {"docs": ["今天天气很好", "机器学习和深度学习的区别"]}
    )
    print(">>> index_docs 返回：", res)   # 5

    print(">>> calling retrieve_docs")   # 6
    await client.query("深度学习 是 什么")
    print(">>> after query")             # 7

    await client.close()
    print(">>> end main")                # 8

if __name__ == "__main__":
    asyncio.run(main())
