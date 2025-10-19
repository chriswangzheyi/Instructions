#!/usr/bin/env python3
"""
MCP RAG 系统启动脚本
支持启动服务器和客户端
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()

def activate_venv():
    """激活虚拟环境"""
    project_root = get_project_root()
    venv_python = project_root / ".venv" / "bin" / "python"
    
    if not venv_python.exists():
        print("❌ 虚拟环境不存在，请先运行 setup.py 配置环境")
        sys.exit(1)
    
    return str(venv_python)

def start_server():
    """启动RAG服务器"""
    print("🚀 启动 RAG 服务器...")
    python_path = activate_venv()
    server_script = get_project_root() / "rag-server" / "server.py"
    
    if not server_script.exists():
        print(f"❌ 服务器脚本不存在: {server_script}")
        sys.exit(1)
    
    try:
        subprocess.run([python_path, str(server_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 服务器启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")

def start_client():
    """启动RAG客户端"""
    print("🚀 启动 RAG 客户端...")
    python_path = activate_venv()
    client_script = get_project_root() / "rag-client" / "client-v4-toolcalls-openai.py"
    server_script = get_project_root() / "rag-server" / "server.py"
    
    if not client_script.exists():
        print(f"❌ 客户端脚本不存在: {client_script}")
        sys.exit(1)
    
    if not server_script.exists():
        print(f"❌ 服务器脚本不存在: {server_script}")
        sys.exit(1)
    
    try:
        subprocess.run([python_path, str(client_script), str(server_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 客户端启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 客户端已停止")

def main():
    parser = argparse.ArgumentParser(description="MCP RAG 系统管理工具")
    parser.add_argument("command", choices=["server", "client"], 
                       help="要执行的命令")
    
    args = parser.parse_args()
    
    if args.command == "server":
        start_server()
    elif args.command == "client":
        start_client()

if __name__ == "__main__":
    main()
