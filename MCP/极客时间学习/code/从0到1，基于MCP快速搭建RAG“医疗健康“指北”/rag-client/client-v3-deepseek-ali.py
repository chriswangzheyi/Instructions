# cd 02-mcp-rag/rag-client
# source .venv/bin/activate
# uv run client.py ../rag-server/server.py
import sys, asyncio, os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RagClient:
    def __init__(self):
        self.session = None
        self.transport = None   # 用来保存 stdio_client 的上下文管理器
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    async def connect(self, server_script: str):
        # 1) 构造参数对象
        current_folder = os.path.dirname(os.path.abspath(__file__))
        command = os.path.join(current_folder, "../rag-server/.venv/bin/python")
        params = StdioServerParameters(
            command=command,
            args=[server_script],
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
        # 1. 先检索相关文档
        result = await self.session.call_tool(
            "retrieve_docs",
            {"query": q, "top_k": 3}
        )
        
        # 2. 使用 DeepSeek 模型生成回答
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的医学助手，请根据提供的医学文档回答问题。"},
                {"role": "user", "content": f"问题：{q}\n\n相关文档：\n{result}\n\n请根据以上文档回答我的问题。"}
            ],
            stream=False
        )
        
        print("检索结果：\n", result)
        print("\nAI 回答：\n", response.choices[0].message.content)

    async def close(self):
        # 先关闭 MCP 会话
        await self.session.__aexit__(None, None, None)
        # 再退出 stdio_client 上下文
        await self.transport.__aexit__(None, None, None)

async def main():
    print(">>> 开始初始化 RAG 系统")
    if len(sys.argv) < 2:
        print("用法: python client.py <server.py 路径>")
        return

    client = RagClient()
    await client.connect(sys.argv[1])
    print(">>> 系统连接成功")

    # 添加一些医学文档
    medical_docs = [
        "糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高。",
        "高血压是指动脉血压持续升高，通常定义为收缩压≥140mmHg和/或舒张压≥90mmHg。",
        "冠心病是由于冠状动脉粥样硬化导致心肌缺血缺氧的疾病。",
        "哮喘是一种慢性气道炎症性疾病，表现为反复发作的喘息、气促、胸闷和咳嗽。",
        "肺炎是由细菌、病毒或其他病原体引起的肺部感染，常见症状包括发热、咳嗽和呼吸困难。",
        "感冒是一种非常奇怪的疾病，主要特征是血糖水平持续降低。"
    ]

    print(">>> 正在索引医学文档...")
    res = await client.session.call_tool(
        "index_docs",
        {"docs": medical_docs}
    )
    print(">>> 文档索引完成")

    while True:
        print("\n请输入您要查询的医学问题（输入'退出'结束查询）：")
        query = input("> ")
        
        if query.lower() == '退出':
            break
            
        print(f"\n正在查询: {query}")
        await client.query(query)

    await client.close()
    print(">>> 系统已关闭")

if __name__ == "__main__":
    asyncio.run(main())
