# Deepseek Harness 简介

DeepSeek Harness 不是一个新的 AI 模型，而是一套让 AI 模型真正执行任务的 Agent（智能体）运行框架。它的核心价值是：让 DeepSeek 不只是回答问题，而是能够调用工具、操作文件、执行代码、完成一整套工作流程。



<img src="Images/1.jpg" style="zoom:50%;" />

公式：**Agent = Model + Harness**

模型负责思考和生成，Harness 负责把这些能力接入文件系统、终端、网页、[工具链](https://zhida.zhihu.com/search?content_id=281728298&content_type=Article&match_order=1&q=工具链&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTIxMTI4NjUsInEiOiLlt6Xlhbfpk74iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyODE3MjgyOTgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.XePwS3eJLM0tnPMAm-RL88FX50ISpMB71QtmnVz3EP4&zhida_source=entity)，让 AI 真正能在现实环境里干活。

<img src="Images/2.jpg" style="zoom:50%;" />



## 它和普通 Agent 框架有什么不同？

DeepSeek Harness 的核心设计理念是 Everything is a Plugin（一切皆插件）。

也就是说，它尽可能把每项功能拆成可以自由组合、替换的插件。

| 组件       | 通俗理解                                |
| ---------- | --------------------------------------- |
| Model      | AI 大脑，负责分析和决策                 |
| Tools      | AI 的双手，负责搜索、操作文件、运行命令 |
| Skills     | 专业技能，例如代码审查、文档处理        |
| Session    | 任务执行记录和上下文                    |
| Sandbox    | 隔离执行环境，控制风险                  |
| Agent Loop | 持续执行、检查结果、决定下一步          |
| Scheduler  | 调度任务或多个 Agent                    |
| UI         | 用户交互界面                            |



## 它和 LangChain、Dify、Claude Code 有什么区别？

| 产品                  | 主要定位             | 特点                             |
| --------------------- | -------------------- | -------------------------------- |
| DeepSeek Harness      | Agent 运行框架       | 插件化程度高，强调可组合、可追踪 |
| LangGraph / LangChain | Agent 开发与编排框架 | 适合开发自定义工作流及状态管理   |
| Dify                  | AI 应用开发平台      | 可视化工作流，快速构建应用       |
| Claude Code           | 编程 Agent 产品      | 更偏向直接完成代码开发任务       |

