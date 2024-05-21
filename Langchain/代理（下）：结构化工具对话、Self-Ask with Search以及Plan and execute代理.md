# 代理（下）：结构化工具对话、Self-Ask with Search以及Plan and execute代理



## 结构化工具



结构化工具的示例包括：

1. 文件管理工具集：支持所有文件系统操作，如写入、搜索、移动、复制、列目录和查找。
2. Web 浏览器工具集：官方的 PlayWright 浏览器工具包，允许代理访问网站、点击、提交表单和查询数据。



以 PlayWright 工具包为例，来实现一个结构化工具对话代理。





## 什么是 Playwright



Playwright 是一个开源的自动化框架，它可以让你模拟真实用户操作网页，帮助开发者和测试者自动化网页交互和测试。用简单的话说，它就像一个“机器人”，可以按照你给的指令去浏览网页、点击按钮、填写表单、读取页面内容等等，就像一个真实的用户在使用浏览器一样。



可以通过 Playwright 浏览器工具来访问一个测试网页。



```python
from playwright.sync_api import sync_playwright

def run():
    # 使用Playwright上下文管理器
    with sync_playwright() as p:
        # 使用Chromium，但你也可以选择firefox或webkit
        browser = p.chromium.launch()
        
        # 创建一个新的页面
        page = browser.new_page()
        
        # 导航到指定的URL
        page.goto('https://langchain.com/')
        
        # 获取并打印页面标题
        title = page.title()
        print(f"Page title is: {title}")
        
        # 关闭浏览器
        browser.close()

if __name__ == "__main__":
    run()
```



输出如下：



```
Page title is: LangChain
```



## 使用结构化工具对话代理



Demo使用的 Agent 类型是 STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION。要使用的工具则是 PlayWrightBrowserToolkit，这是 LangChain 中基于 PlayWrightBrowser 包封装的工具箱，它继承自 BaseToolkit 类。





```python
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_async_playwright_browser

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
print(tools)

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatAnthropic, ChatOpenAI

# LLM不稳定，对于这个任务，可能要多跑几次才能得到正确结果
llm = ChatOpenAI(temperature=0.5)  

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

async def main():
    response = await agent_chain.arun("What are the headers on python.langchain.com?")
    print(response)

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

