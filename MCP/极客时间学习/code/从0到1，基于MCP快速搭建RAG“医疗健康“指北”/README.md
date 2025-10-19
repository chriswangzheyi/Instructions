# MCP RAG 医疗健康问答系统

基于 MCP (Model Context Protocol) 快速搭建的 RAG (Retrieval-Augmented Generation) 系统，专门用于医疗健康领域的智能问答。

## 🚀 快速开始

### 1. 环境配置

项目已经配置好了虚拟环境和依赖，你只需要：

1. **配置API密钥**：编辑 `.env` 文件，添加你的API密钥：
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here  # 可选
   DEEPSEEK_API_KEY=your_deepseek_api_key_here    # 可选
   ```

2. **激活虚拟环境**：
   ```bash
   source .venv/bin/activate
   ```

### 2. 启动方式

#### 方式一：使用启动脚本（推荐）

```bash
# 启动服务器
python run.py server

# 启动客户端
python run.py client
```

#### 方式二：使用IDE直接运行

在 VS Code 中：
1. 按 `F5` 或点击"运行和调试"
2. 选择以下配置之一：
   - **启动 RAG 服务器** - 直接运行服务器
   - **启动 RAG 客户端** - 直接运行客户端

#### 方式三：手动启动

```bash
# 启动服务器
.venv/bin/python rag-server/server.py

# 启动客户端（新终端）
.venv/bin/python rag-client/client-v4-toolcalls-openai.py rag-server/server.py
```

## 📁 项目结构

```
├── rag-server/           # RAG服务器
│   ├── server.py         # 主服务器文件
│   └── pyproject.toml    # 依赖配置
├── rag-client/           # RAG客户端
│   ├── client-v4-toolcalls-openai.py  # OpenAI版本客户端
│   └── pyproject.toml    # 依赖配置
├── .venv/                # Python虚拟环境
├── .env                  # 环境变量配置
├── run.py               # 启动脚本
├── .vscode/             # VS Code配置
└── 说明.md              # 详细说明文档
```

## 🎯 使用场景

- **医疗健康问答** - 基于医学文档的智能问答
- **知识库检索** - 企业文档的智能检索
- **专业领域咨询** - 基于专业文档的咨询服务

## 🛠️ 技术栈

- **FastMCP** - MCP服务器框架
- **Faiss** - 向量索引引擎
- **OpenAI** - 文本嵌入和LLM
- **Python 3.10+** - 开发语言

## 📝 使用示例

1. **启动系统**：
   ```bash
   python run.py server  # 终端1
   python run.py client  # 终端2
   ```

2. **索引文档**：系统会自动索引预设的医学文档

3. **开始问答**：
   ```
   请输入您要查询的医学问题（输入'退出'结束查询）：
   > 什么是糖尿病？
   
   AI 回答：
   糖尿病是一种慢性代谢性疾病，主要特征是血糖水平持续升高...
   ```

## 🔍 故障排除

### 常见问题

1. **虚拟环境问题**：
   ```bash
   # 重新创建虚拟环境
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install faiss-cpu mcp[cli] openai anthropic python-dotenv numpy
   ```

2. **API密钥问题**：
   - 确保 `.env` 文件中的API密钥正确
   - 检查API密钥是否有足够的额度

## 📚 更多信息

详细的技术说明请查看 `说明.md` 文件。
