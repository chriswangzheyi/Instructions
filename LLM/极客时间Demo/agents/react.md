# LangChain 核心模块 Agent - ReAct

ReAct 核心思想是 推理+操作，本示例以` Google Search` 和 `LLM Math` 作为可选操作集合（toolkits），实现 ReAct 功能。

以下`SERPAPI_API_KEY`仅为示例，请访问 https://serpapi.com 注册账号并替换为自己的 `API_KEY`（每月100次免费调用）


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

llm = OpenAI(temperature=0)
```


```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

#加载 LangChain 内置的 Tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```

### 使用语言模型（GPT-3）


```python
# 实例化 ZERO_SHOT_REACT Agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```

    /opt/anaconda3/envs/LLM/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `initialize_agent` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use Use new agent constructor methods like create_react_agent, create_json_agent, create_structured_chat_agent, etc. instead.
      warn_deprecated(



```python
agent.run("谁是莱昂纳多·迪卡普里奥的女朋友？她现在年龄的0.43次方是多少?")
```

    /opt/anaconda3/envs/LLM/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `run` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(


​    
​    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I should use a search engine to find information about Leonardo DiCaprio's girlfriend and then use a calculator to calculate her age.
    Action: Search
    Action Input: "Leonardo DiCaprio girlfriend"[0m
    Observation: [36;1m[1;3mModel Vittoria Ceretti, who is dating Leonardo DiCaprio, landed an Elle cover story to talk about modeling and not talk about Leo. The 49-year-old actor only warrants one mention in the piece about his 25-year-old girlfriend.[0m
    Thought:[32;1m[1;3m Now I have the name of Leonardo DiCaprio's girlfriend, I can use a calculator to calculate her age.
    Action: Calculator
    Action Input: 25^0.43[0m
    Observation: [33;1m[1;3mAnswer: 3.991298452658078[0m
    Thought:[32;1m[1;3m I now know that Leonardo DiCaprio's girlfriend is 25 years old and her age to the power of 0.43 is approximately 3.99.
    Final Answer: 3.99[0m
    
    [1m> Finished chain.[0m





    '3.99'



### 使用聊天模型（GPT-4）


```python
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("谁是莱昂纳多·迪卡普里奥的女朋友？她现在年龄的0.43次方是多少?")
```


​    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mQuestion: 谁是莱昂纳多·迪卡普里奥的女朋友？
    Thought: 首先，我需要找出莱昂纳多·迪卡普里奥当前的女朋友是谁。这是一个关于当前事件的问题，所以我将使用搜索工具来找到答案。
    Action:
    ```json
    {
      "action": "Search",
      "action_input": "莱昂纳多·迪卡普里奥 当前女朋友"
    }
    ```
    [0m
    Observation: [36;1m[1;3m19岁的模特伊甸园·波拉尼被认为是“泰坦尼克号”才子莱昂纳多·迪卡普里奥的新女友。 《每日邮报》刊登了一张演员和年轻女友的照片。 据报道，伊甸园·波拉尼是以色列人。 她拥有1米73的身高，具有发展模特工作的优势。[0m
    Thought:[32;1m[1;3m根据搜索结果，莱昂纳多·迪卡普里奥的女朋友是19岁的模特伊甸园·波拉尼。现在我需要计算她年龄的0.43次方。
    
    Action:
    ```json
    {
      "action": "Calculator",
      "action_input": {
        "expression": "19^0.43"
      }
    }
    ```
    
    [0m


```python

```
