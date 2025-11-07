# **langchain**



## 定义

**LangChain** 是一个开源框架，旨在帮助开发者**将大型语言模型（LLMs）与外部数据、工具和多步骤逻辑整合起来**。

**核心价值：**

- 把 LLM 视为**推理引擎（reasoning engine）**；
- 提供一套标准化接口，让开发者专注于业务逻辑而非底层调用；
- 提供 Prompt 管理、记忆、检索、工具调用、代理（Agent）等模块。



## 核心概念



LangChain 的架构由 **Components（组件）** 与 **Chains（链）** 两层构成：

> Components 是基本单元，Chains 是它们的组合

**Components**：
 基础构件，如 LLM、Prompt、Retriever、Memory、Tool、Output Parser 等。
 每个组件有清晰的输入/输出接口。

**Chains**：
 将多个组件串联起来形成一个工作流（Workflow）。



#### Prompt Templates and Values

**PromptTemplate** 是 LangChain 的灵魂，用于动态构造输入提示词

```python
from langchain import PromptTemplate
template = "Translate the following text into {language}: {text}"
prompt = PromptTemplate(template=template, input_variables=["language", "text"])
```

- **PromptTemplate**：定义模板结构；
- **PromptValue**：传入实际的值；
- 支持变量替换、Prompt 复用、Prompt 优化。



#### Example Selectors

**Example Selector** 是 LangChain 提供的“示例管理器”。
 用于 Few-shot Learning 场景 —— 让模型通过上下文示例学习任务模式。

```
用户问题 → 自动选择相似示例 → 构造 Few-shot Prompt → 调用 LLM
```

常见选择策略：

- **Similarity Selector**（基于向量相似度）
- **Length-based Selector**（按 Prompt 长度）
- **MMLU / embedding 检索选择**



#### Output Parsers

LLM 输出往往是**非结构化文本**。
 **Output Parser** 将模型输出解析为结构化数据，如 JSON、字典或 Pydantic 对象。

```python
from langchain.output_parsers import StructuredOutputParser
```

用途：

- 将自然语言回答转化为可编程数据；
- 支撑“代码执行”“参数提取”“数据库更新”等自动化场景。



#### Indexes and Retrievers

LangChain 提供 **文档索引（Indexes）** 和 **检索模块（Retrievers）** 来支持 **RAG（Retrieval-Augmented Generation）**。

###### 🧩 Indexes

- 把文本、PDF、网页等数据切分、向量化、存储；
- 常见后端：FAISS、Chroma、Pinecone、Milvus。

###### 🔍 Retrievers

- 根据用户 Query 检索最相关文档；
- 将检索结果注入 LLM 上下文。



#### Chat Message History

**ChatMessageHistory** 模块用于管理上下文记忆。它支持多轮对话中保存与加载历史消息，使 LLM 能实现“连续记忆”。

类型：

- In-memory（默认）
- Redis / SQLite / Chroma / DynamoDB 等持久化版本



####  Agents and Toolkits

###### 🧠 Agents

Agent 是 LangChain 的智能“决策层”。
 它允许 LLM 在运行时**自主选择调用的工具**来解决问题。

###### 🧰 Toolkits

Toolkit 是 Agent 可调用的工具集合，比如：

- 搜索引擎（SerpAPI）
- Python 解释器
- Calculator
- SQL 数据库接口
- 自定义 API（如天气、日程、知识库）

> 通过 ReAct 框架（Reason + Act），模型能像人类一样“思考 + 行动 + 反思”。



## 什么是 LangChain Agent

```
用户指令 → LLM 判断需要哪些工具 → 调用工具 → 整理结果 → 生成最终回答
```

类型：

- **Zero-shot-react-description**：无示例推理型；
- **Conversational Agent**：有记忆的对话代理；
- **Plan-and-execute Agent**：分解任务再执行；
- **Structured Tools Agent**：明确定义输入输出参数的工具调用。



## 如何使用 LangChain？

1. **选择模型**

```python
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)
```

2. **定义 Prompt**

```python
from langchain import PromptTemplate
prompt = PromptTemplate(template="What is {thing}?", input_variables=["thing"])
```

3.**构建chain**

```python
from langchain.chains import LLMChain
chain = LLMChain(prompt=prompt, llm=llm)
chain.run("LangChain")
```

4. **添加 Memory / Retriever / Tool**
   - 用 Memory 保存上下文；
   - 用 Retriever 连接外部知识；
   - 用 Agent 调用外部 API。



## LangChain 支持哪些功能



| 类别        | 功能                               |
| ----------- | ---------------------------------- |
| Prompt 工程 | 模板、示例选择、动态注入           |
| 模型接口    | LLM、Chat、Embedding、Text2SQL 等  |
| 知识检索    | RAG、向量数据库集成                |
| 记忆管理    | Conversation Memory、Buffer Memory |
| 工具扩展    | Agents + Toolkits                  |
| 结果解析    | Output Parser、Structured Output   |
| 调试与评估  | LangSmith、Callback、Tracing       |



## 什么是 LangChain model

LangChain 不定义新的模型，而是封装了 LLM API（如 OpenAI、Anthropic、Cohere、HuggingFace Hub 等）。
 也支持本地模型（如 Llama 2、Qwen、ChatGLM、Vicuna）。



LangChain Model 的作用：

- 提供统一接口（统一调用逻辑）；
- 支持模型间切换（从 GPT → Llama 无缝过渡）；
- 可在多框架（OpenAI / Ollama / HuggingFace）间复用。