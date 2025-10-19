#!/usr/bin/env python3
"""
测试MCP服务器和客户端通信
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test_mcp_connection():
    print("🔧 测试MCP连接...")
    
    # 服务器脚本路径
    server_script = project_root / "rag-server" / "server.py"
    print(f"📁 服务器脚本: {server_script}")
    print(f"📁 脚本存在: {server_script.exists()}")
    
    if not server_script.exists():
        print("❌ 服务器脚本不存在")
        return False
    
    try:
        # 创建参数
        params = StdioServerParameters(
            command="python3",
            args=[str(server_script)],
        )
        
        print("🚀 启动MCP服务器...")
        
        # 创建客户端
        async with stdio_client(params) as (stdio, write):
            print("✅ 服务器启动成功")
            
            # 创建会话
            async with ClientSession(stdio, write) as session:
                print("✅ 会话创建成功")
                
                # 初始化会话
                await session.initialize()
                print("✅ 会话初始化成功")
                
                # 获取工具列表
                tools_resp = await session.list_tools()
                print(f"✅ 获取到 {len(tools_resp.tools)} 个工具:")
                for tool in tools_resp.tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # 测试工具调用
                print("🧪 测试工具调用...")
                result = await session.call_tool("index_docs", {
                    "docs": ["这是一个测试文档"]
                })
                print(f"✅ 工具调用成功: {result}")
                
                return True
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🐍 Python环境测试")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"当前目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    
    # 检查依赖
    print("\n📦 依赖检查")
    print("-" * 30)
    try:
        import mcp
        print("✅ MCP 已安装")
    except ImportError as e:
        print(f"❌ MCP 未安装: {e}")
        return
    
    try:
        import openai
        print("✅ OpenAI 已安装")
    except ImportError as e:
        print(f"❌ OpenAI 未安装: {e}")
        return
    
    try:
        import faiss
        print("✅ FAISS 已安装")
    except ImportError as e:
        print(f"❌ FAISS 未安装: {e}")
        return
    
    # 测试MCP连接
    print("\n🔗 MCP连接测试")
    print("-" * 30)
    success = await test_mcp_connection()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败，请检查配置")

if __name__ == "__main__":
    asyncio.run(main())
