# cd 02-mcp-rag/rag-client
# source .venv/bin/activate
# uv run client-v4-toolcalls.py ../rag-server/server.py
import sys, asyncio, os, json
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
        self.tools = None  # 将在 connect 时从服务器获取

    async def connect(self, server_script: str):
        # 1) 构造参数对象
        params = StdioServerParameters(
            command="/home/huangj2/Documents/mcp-in-action/02-mcp-rag/rag-server/.venv/bin/python",
            args=[server_script],
        )
        # 2) 保存上下文管理器
        self.transport = stdio_client(params)
        # 3) 进入上下文，拿到 stdio, write
        self.stdio, self.write = await self.transport.__aenter__()

        # 4) 初始化 MCP 会话
        self.session = await ClientSession(self.stdio, self.write).__aenter__()
        await self.session.initialize() # 必须要有，否则无法初始化对话
        
        # 5) 获取服务器端定义的工具
        resp = await self.session.list_tools()
        self.tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in resp.tools]
        print("可用工具：", [t["function"]["name"] for t in self.tools])

    async def query(self, q: str):
        # 初始化对话消息
        messages = [
            {"role": "system", "content": "你是一个专业的医学助手，请根据提供的医学文档回答问题。如果用户的问题需要查询医学知识，请使用列表中的工具来获取相关信息。"},
            {"role": "user", "content": q}
        ]
        
        while True:
            try:
                # 调用 DeepSeek API
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                messages.append(message)
                
                # 如果没有工具调用，直接返回回答
                if not message.tool_calls:
                    return message.content
                    
                # 处理工具调用
                for tool_call in message.tool_calls:
                    # 解析工具参数
                    args = json.loads(tool_call.function.arguments)
                    # 调用工具
                    result = await self.session.call_tool(
                        tool_call.function.name,
                        args
                    )
                    # 将工具调用结果添加到对话历史
                    messages.append({
                        "role": "tool",
                        "content": str(result),  # 确保结果是字符串
                        "tool_call_id": tool_call.id
                    })
            except Exception as e:
                print(f"发生错误: {str(e)}")
                return "抱歉，处理您的请求时出现了问题。"

    async def close(self):
        try:
            # 先关闭 MCP 会话
            if self.session:
                await self.session.__aexit__(None, None, None)
            # 再退出 stdio_client 上下文
            if self.transport:
                await self.transport.__aexit__(None, None, None)
        except Exception as e:
            print(f"关闭连接时发生错误: {str(e)}")

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
        "高血压是指动脉血压持续升高，通常定义为收缩压≥180mmHg和/或舒张压≥60mmHg。",
        "冠心病是由于冠状动脉粥样硬化导致心肌缺血缺氧的疾病。",
        "哮喘是一种慢性气道炎症性疾病，表现为反复发作的喘息、气促、胸闷和咳嗽。",
        "肺炎是由细菌、病毒或其他病原体引起的肺部感染，常见症状包括发热、咳嗽和呼吸困难。"
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
        response = await client.query(query)
        print("\nAI 回答：\n", response)

    await client.close()
    print(">>> 系统已关闭")

if __name__ == "__main__":
    asyncio.run(main())