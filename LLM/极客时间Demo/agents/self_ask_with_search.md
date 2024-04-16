# LangChain 核心模块 Agent - Self ask with search

- Google 搜索对接
    - 第三方平台：https://serpapi.com
    - LangChain API 封装：`SerpAPIWrapper`
- LangChain Agent 模块使用
    - Agent 类型：`SELF_ASK_WITH_SEARCH`
    - Agent 实例化：`initialize_agent` 方法

https://python.langchain.com/docs/modules/agents/agent_types/


```python
import os

# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "d9824b4920b4705c8cfc2929be56486c1e02b1360308c2299a4671e6b8bab1c6"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'
os.environ['OPENAI_API_KEY'] = 'key here'

```


```python
from langchain_openai import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool

llm = OpenAI(temperature=0)
```


```python
# 实例化查询工具
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]
```


```python
# 实例化 SELF_ASK_WITH_SEARCH Agent
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True,handle_parsing_errors=True
)
```


```python
# 实际运行 Agent，查询问题（正确）
self_ask_with_search.run(
    "成都举办的大运会是第几届大运会？"
)
```


​    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mCould not parse output:  No. 
    Final answer: 第28届大运会 (The 28th Universiade)[0m
    Intermediate answer: Invalid or incomplete response
    [32;1m[1;3mCould not parse output: [0m
    Intermediate answer: Invalid or incomplete response
    [32;1m[1;3mSo the final answer is: 第28届大运会 (The 28th Universiade)[0m
    
    [1m> Finished chain.[0m





    '第28届大运会 (The 28th Universiade)'




```python
# 实际运行 Agent，查询问题（有歧义）
self_ask_with_search.run(
    "2023年大运会举办地在哪里？"
)
```


​    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mCould not parse output:  No.[0m
    Intermediate answer: Invalid or incomplete response
    [32;1m[1;3mSo the final answer is: Invalid or incomplete response[0m
    
    [1m> Finished chain.[0m





    'Invalid or incomplete response'




```python
# Reason-only 答案（错误）
print(llm("成都举办的大运会是第几届大运会？"))
```

    /opt/anaconda3/envs/LLM/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(


​    
​    
    成都举办的大运会是第31届大运会。



```python
# Reason-only 答案（错误）
print(llm("2023年大运会举办地在哪里？"))
```


​    
    2023年大运会的举办地将在中国的重庆市。


#### 使用 GPT-4 作为大语言模型实现更优的 ReAct 范式


```python
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
```


```python
self_ask_with_search_chat = initialize_agent(
    tools, chat_model, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
```


```python
# GPT-4 based ReAct 答案（正确）
self_ask_with_search_chat.run(
    "2023年大运会举办地在哪里？"
)
```


​    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mYes.
    Follow up: What is the 2023年大运会 in English?[0m
    Intermediate answer: [36;1m[1;3m2023年夏季世界大学生运动会（英语：XXXII 2023 Summer World University Games）简称为叶卡捷琳堡大运会，是第32届夏季世界大学生运动会。[0m
    [32;1m[1;3mFollow up: Where is the 2023 Summer World University Games being held?[0m
    Intermediate answer: [36;1m[1;3m['... event, which was to be held from 8 to 19 August 2023 in Yekaterinburg, Russia. 2023 Summer World University Games 2023 Летняя Универсиада. Host city ...', 'FISU Summer World University Games · FISU Winter World University Games · FISU World University Championships. Football World Cup Jinjiang 2023. FISU University ...', 'With Rhine Ruhr 2025, the FISU Summer Games will be held in Germany for the second time after Duisburg in 1989. And it will be the first time ever that the ...', 'It effectively replaced the 2023 Summer World University Games, that was set to be held in Yekaterinburg, Russia. It was cancelled after the 2022 Russian ...', 'The XXXI Summer World University Games, which are set to be held from 28 July to 8 August 2023 in Chengdu, Sichuan, China, released the competition schedule for ...', "The opening ceremony of the FISU World University Games 2023 Chengdu will be held on July 28 at the Dong'an Lake Sports Park Stadium while ...", 'ABOUT THE 2023 CHENGDU GAMES\u200b\u200b The 2023 Summer World Unviersity Games will be held in Chengdu, China July 28 - August 8, 2023. The event marks the 32nd edition ...', 'The 2023 Summer Universiade announced the competition schedule for the FISU World University Games in Chengdu, China.', "See which nations won medals at the FISU World University Games 2023 in Chengdu in the People's Republic of China between 28 July and 8 August.", 'Play has begun for the U.S. athletes competing in Chengdu, China at the FISU World University Games taking place from July 28 to August 8, ...'][0m
    [32;1m[1;3mSo the final answer is: Chengdu, China[0m
    
    [1m> Finished chain.[0m





    'Chengdu, China'




```python

```
