#!/usr/bin/env python3
"""
测试Python解释器和环境
"""

import sys
import os
from pathlib import Path

def test_environment():
    print("🐍 Python环境测试")
    print("=" * 50)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {Path(__file__).parent.absolute()}")
    
    # 检查虚拟环境
    venv_path = Path(__file__).parent / ".venv" / "bin" / "python"
    print(f"虚拟环境路径: {venv_path}")
    print(f"虚拟环境存在: {venv_path.exists()}")
    
    # 检查依赖包
    print("\n📦 依赖包检查")
    print("-" * 30)
    
    packages = [
        "mcp",
        "openai", 
        "faiss",
        "numpy",
        "anthropic",
        "dotenv"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
    
    # 环境变量
    print("\n🔧 环境变量")
    print("-" * 30)
    env_file = Path(__file__).parent / ".env"
    print(f".env文件存在: {env_file.exists()}")
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OPENAI_API_KEY - 已配置")
            else:
                print("❌ OPENAI_API_KEY - 未配置")
    
    print("\n🎯 建议")
    print("-" * 30)
    print("1. 如果依赖包未安装，请运行:")
    print("   source .venv/bin/activate")
    print("   pip install faiss-cpu mcp[cli] openai anthropic python-dotenv numpy")
    print("\n2. 如果API密钥未配置，请编辑 .env 文件")
    print("\n3. 在VS Code中选择Python解释器:")
    print("   Cmd+Shift+P -> Python: Select Interpreter")
    print("   选择: ./.venv/bin/python")

if __name__ == "__main__":
    test_environment()
