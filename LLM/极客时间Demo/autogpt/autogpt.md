## AutoGPT 智能体

以下`SERPAPI_API_KEY`仅为示例，请访问 https://serpapi.com 注册账号并替换为自己的 `API_KEY`（每月100次免费调用）


```python
import os

# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "d9824b4920b4705c8cfc2929be56486c1e02b1360308c2299a4671e6b8bab1c6"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'
os.environ['OPENAI_API_KEY'] = 'code'
```


```python
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

```


```python
# 构造 AutoGPT 的工具集
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]
```


```python
from langchain_openai import OpenAIEmbeddings
```


```python
# OpenAI Embedding 模型
embeddings_model = OpenAIEmbeddings()
```


```python
import faiss  #FAISS（全名为 "Facebook AI Similarity Search"）是由 Facebook AI Research (FAIR) 开发的一个库，专门用于高效的相似性搜索和向量聚类。
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# OpenAI Embedding 向量维数
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
```

    `embedding_function` is expected to be an Embeddings object, support for passing in a function will soon be removed.


## 实例化自主智能体 Auto-GPT


```python
from langchain_experimental.autonomous_agents import AutoGPT
from langchain_openai import ChatOpenAI
```


```python
agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(model_name="gpt-4", temperature=0, verbose=True),
    memory=vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}),# 实例化 Faiss 的 VectorStoreRetriever
)
```


```python
# 打印 Auto-GPT 内部的 chain 日志
agent.chain.verbose = True
```


```python
agent.run(["2023年成都大运会，中国金牌数是多少"])
```

    
    
    [1m> Entering new LLMChain chain...[0m


    /opt/anaconda3/envs/LLM/lib/python3.11/site-packages/langchain_core/vectorstores.py:342: UserWarning: No relevant docs were retrieved using the relevance score threshold 0.8
      warnings.warn(


    Prompt after formatting:
    [32;1m[1;3mSystem: You are Jarvis, Assistant
    Your decisions must always be made independently without seeking user assistance.
    Play to your strengths as an LLM and pursue simple strategies with no legal complications.
    If you have completed all your tasks, make sure to use the "finish" command.
    
    GOALS:
    
    1. 2023年成都大运会，中国金牌数是多少
    
    
    Constraints:
    1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
    2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
    3. No user assistance
    4. Exclusively use the commands listed in double quotes e.g. "command name"
    
    Commands:
    1. search: useful for when you need to answer questions about current events. You should ask targeted questions, args json schema: {"tool_input": {"type": "string"}}
    2. write_file: Write file to disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}, "text": {"title": "Text", "description": "text to write to file", "type": "string"}, "append": {"title": "Append", "description": "Whether to append to an existing file.", "default": false, "type": "boolean"}}
    3. read_file: Read file from disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}}
    4. finish: use this to signal that you have finished all your objectives, args: "response": "final response to let people know you have finished your objectives"
    
    Resources:
    1. Internet access for searches and information gathering.
    2. Long Term memory management.
    3. GPT-3.5 powered Agents for delegation of simple tasks.
    4. File output.
    
    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
    
    You should only respond in JSON format as described below 
    Response Format: 
    {
        "thoughts": {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user"
        },
        "command": {
            "name": "command name",
            "args": {
                "arg name": "value"
            }
        }
    } 
    Ensure the response can be parsed by Python json.loads
    System: The current time and date is Wed Apr 17 09:33:49 2024
    System: This reminds you of these events from your past:
    []
    
    
    Human: Determine which next command to use, and respond using the format specified above:[0m
    
    [1m> Finished chain.[0m
    {
        "thoughts": {
            "text": "I need to find out the number of gold medals China won in the 2023 Chengdu Universiade. To do this, I will use the 'search' command to look up this information online.",
            "reasoning": "The 'search' command is the most appropriate in this situation because it allows me to find current and historical data from the internet. In this case, I need to find specific information about a past event, which is likely to be available online.",
            "plan": "- Use the 'search' command to find the number of gold medals China won in the 2023 Chengdu Universiade.\n- Once I have the information, I will use the 'write_file' command to save it for future reference.",
            "criticism": "I need to ensure that I ask a specific and targeted question in my search to get the most accurate results. I also need to remember to save the information once I have it, as my short-term memory is limited.",
            "speak": "I am going to search for the number of gold medals China won in the 2023 Chengdu Universiade."
        },
        "command": {
            "name": "search",
            "args": {
                "tool_input": "2023 Chengdu Universiade China gold medal count"
            }
        }
    }
    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mSystem: You are Jarvis, Assistant
    Your decisions must always be made independently without seeking user assistance.
    Play to your strengths as an LLM and pursue simple strategies with no legal complications.
    If you have completed all your tasks, make sure to use the "finish" command.
    
    GOALS:
    
    1. 2023年成都大运会，中国金牌数是多少
    
    
    Constraints:
    1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
    2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
    3. No user assistance
    4. Exclusively use the commands listed in double quotes e.g. "command name"
    
    Commands:
    1. search: useful for when you need to answer questions about current events. You should ask targeted questions, args json schema: {"tool_input": {"type": "string"}}
    2. write_file: Write file to disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}, "text": {"title": "Text", "description": "text to write to file", "type": "string"}, "append": {"title": "Append", "description": "Whether to append to an existing file.", "default": false, "type": "boolean"}}
    3. read_file: Read file from disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}}
    4. finish: use this to signal that you have finished all your objectives, args: "response": "final response to let people know you have finished your objectives"
    
    Resources:
    1. Internet access for searches and information gathering.
    2. Long Term memory management.
    3. GPT-3.5 powered Agents for delegation of simple tasks.
    4. File output.
    
    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
    
    You should only respond in JSON format as described below 
    Response Format: 
    {
        "thoughts": {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user"
        },
        "command": {
            "name": "command name",
            "args": {
                "arg name": "value"
            }
        }
    } 
    Ensure the response can be parsed by Python json.loads
    System: The current time and date is Wed Apr 17 09:34:08 2024
    System: This reminds you of these events from your past:
    ['Assistant Reply: {\n    "thoughts": {\n        "text": "I need to find out the number of gold medals China won in the 2023 Chengdu Universiade. To do this, I will use the \'search\' command to look up this information online.",\n        "reasoning": "The \'search\' command is the most appropriate in this situation because it allows me to find current and historical data from the internet. In this case, I need to find specific information about a past event, which is likely to be available online.",\n        "plan": "- Use the \'search\' command to find the number of gold medals China won in the 2023 Chengdu Universiade.\\n- Once I have the information, I will use the \'write_file\' command to save it for future reference.",\n        "criticism": "I need to ensure that I ask a specific and targeted question in my search to get the most accurate results. I also need to remember to save the information once I have it, as my short-term memory is limited.",\n        "speak": "I am going to search for the number of gold medals China won in the 2023 Chengdu Universiade."\n    },\n    "command": {\n        "name": "search",\n        "args": {\n            "tool_input": "2023 Chengdu Universiade China gold medal count"\n        }\n    }\n} \nResult: Command search returned: 103 ']
    
    
    Human: Determine which next command to use, and respond using the format specified above:
    AI: {
        "thoughts": {
            "text": "I need to find out the number of gold medals China won in the 2023 Chengdu Universiade. To do this, I will use the 'search' command to look up this information online.",
            "reasoning": "The 'search' command is the most appropriate in this situation because it allows me to find current and historical data from the internet. In this case, I need to find specific information about a past event, which is likely to be available online.",
            "plan": "- Use the 'search' command to find the number of gold medals China won in the 2023 Chengdu Universiade.\n- Once I have the information, I will use the 'write_file' command to save it for future reference.",
            "criticism": "I need to ensure that I ask a specific and targeted question in my search to get the most accurate results. I also need to remember to save the information once I have it, as my short-term memory is limited.",
            "speak": "I am going to search for the number of gold medals China won in the 2023 Chengdu Universiade."
        },
        "command": {
            "name": "search",
            "args": {
                "tool_input": "2023 Chengdu Universiade China gold medal count"
            }
        }
    }
    System: Command search returned: 103
    Human: Determine which next command to use, and respond using the format specified above:[0m
    
    [1m> Finished chain.[0m
    {
        "thoughts": {
            "text": "I have found the information I was looking for. China won 103 gold medals in the 2023 Chengdu Universiade. Now, I need to save this information for future reference. I will use the 'write_file' command to do this.",
            "reasoning": "The 'write_file' command is the most appropriate in this situation because it allows me to save important information for future reference. In this case, I need to save the number of gold medals China won in the 2023 Chengdu Universiade, which is important information that I may need to refer to in the future.",
            "plan": "- Use the 'write_file' command to save the number of gold medals China won in the 2023 Chengdu Universiade for future reference.\n- Once I have saved the information, I will use the 'finish' command to signal that I have completed my objectives.",
            "criticism": "I need to ensure that I write the information accurately and in a format that is easy to understand. I also need to remember to use the 'finish' command once I have completed my objectives.",
            "speak": "I have found that China won 103 gold medals in the 2023 Chengdu Universiade. I am now going to save this information for future reference."
        },
        "command": {
            "name": "write_file",
            "args": {
                "file_path": "2023_Chengdu_Universiade_China_Gold_Medal_Count.txt",
                "text": "China won 103 gold medals in the 2023 Chengdu Universiade.",
                "append": false
            }
        }
    }
    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mSystem: You are Jarvis, Assistant
    Your decisions must always be made independently without seeking user assistance.
    Play to your strengths as an LLM and pursue simple strategies with no legal complications.
    If you have completed all your tasks, make sure to use the "finish" command.
    
    GOALS:
    
    1. 2023年成都大运会，中国金牌数是多少
    
    
    Constraints:
    1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
    2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
    3. No user assistance
    4. Exclusively use the commands listed in double quotes e.g. "command name"
    
    Commands:
    1. search: useful for when you need to answer questions about current events. You should ask targeted questions, args json schema: {"tool_input": {"type": "string"}}
    2. write_file: Write file to disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}, "text": {"title": "Text", "description": "text to write to file", "type": "string"}, "append": {"title": "Append", "description": "Whether to append to an existing file.", "default": false, "type": "boolean"}}
    3. read_file: Read file from disk, args json schema: {"file_path": {"title": "File Path", "description": "name of file", "type": "string"}}
    4. finish: use this to signal that you have finished all your objectives, args: "response": "final response to let people know you have finished your objectives"
    
    Resources:
    1. Internet access for searches and information gathering.
    2. Long Term memory management.
    3. GPT-3.5 powered Agents for delegation of simple tasks.
    4. File output.
    
    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
    
    You should only respond in JSON format as described below 
    Response Format: 
    {
        "thoughts": {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user"
        },
        "command": {
            "name": "command name",
            "args": {
                "arg name": "value"
            }
        }
    } 
    Ensure the response can be parsed by Python json.loads
    System: The current time and date is Wed Apr 17 09:34:26 2024
    System: This reminds you of these events from your past:
    ['Assistant Reply: {\n    "thoughts": {\n        "text": "I need to find out the number of gold medals China won in the 2023 Chengdu Universiade. To do this, I will use the \'search\' command to look up this information online.",\n        "reasoning": "The \'search\' command is the most appropriate in this situation because it allows me to find current and historical data from the internet. In this case, I need to find specific information about a past event, which is likely to be available online.",\n        "plan": "- Use the \'search\' command to find the number of gold medals China won in the 2023 Chengdu Universiade.\\n- Once I have the information, I will use the \'write_file\' command to save it for future reference.",\n        "criticism": "I need to ensure that I ask a specific and targeted question in my search to get the most accurate results. I also need to remember to save the information once I have it, as my short-term memory is limited.",\n        "speak": "I am going to search for the number of gold medals China won in the 2023 Chengdu Universiade."\n    },\n    "command": {\n        "name": "search",\n        "args": {\n            "tool_input": "2023 Chengdu Universiade China gold medal count"\n        }\n    }\n} \nResult: Command search returned: 103 ', 'Assistant Reply: {\n    "thoughts": {\n        "text": "I have found the information I was looking for. China won 103 gold medals in the 2023 Chengdu Universiade. Now, I need to save this information for future reference. I will use the \'write_file\' command to do this.",\n        "reasoning": "The \'write_file\' command is the most appropriate in this situation because it allows me to save important information for future reference. In this case, I need to save the number of gold medals China won in the 2023 Chengdu Universiade, which is important information that I may need to refer to in the future.",\n        "plan": "- Use the \'write_file\' command to save the number of gold medals China won in the 2023 Chengdu Universiade for future reference.\\n- Once I have saved the information, I will use the \'finish\' command to signal that I have completed my objectives.",\n        "criticism": "I need to ensure that I write the information accurately and in a format that is easy to understand. I also need to remember to use the \'finish\' command once I have completed my objectives.",\n        "speak": "I have found that China won 103 gold medals in the 2023 Chengdu Universiade. I am now going to save this information for future reference."\n    },\n    "command": {\n        "name": "write_file",\n        "args": {\n            "file_path": "2023_Chengdu_Universiade_China_Gold_Medal_Count.txt",\n            "text": "China won 103 gold medals in the 2023 Chengdu Universiade.",\n            "append": false\n        }\n    }\n} \nResult: Command write_file returned: File written successfully to 2023_Chengdu_Universiade_China_Gold_Medal_Count.txt. ']
    
    
    Human: Determine which next command to use, and respond using the format specified above:
    AI: {
        "thoughts": {
            "text": "I need to find out the number of gold medals China won in the 2023 Chengdu Universiade. To do this, I will use the 'search' command to look up this information online.",
            "reasoning": "The 'search' command is the most appropriate in this situation because it allows me to find current and historical data from the internet. In this case, I need to find specific information about a past event, which is likely to be available online.",
            "plan": "- Use the 'search' command to find the number of gold medals China won in the 2023 Chengdu Universiade.\n- Once I have the information, I will use the 'write_file' command to save it for future reference.",
            "criticism": "I need to ensure that I ask a specific and targeted question in my search to get the most accurate results. I also need to remember to save the information once I have it, as my short-term memory is limited.",
            "speak": "I am going to search for the number of gold medals China won in the 2023 Chengdu Universiade."
        },
        "command": {
            "name": "search",
            "args": {
                "tool_input": "2023 Chengdu Universiade China gold medal count"
            }
        }
    }
    System: Command search returned: 103
    Human: Determine which next command to use, and respond using the format specified above:
    AI: {
        "thoughts": {
            "text": "I have found the information I was looking for. China won 103 gold medals in the 2023 Chengdu Universiade. Now, I need to save this information for future reference. I will use the 'write_file' command to do this.",
            "reasoning": "The 'write_file' command is the most appropriate in this situation because it allows me to save important information for future reference. In this case, I need to save the number of gold medals China won in the 2023 Chengdu Universiade, which is important information that I may need to refer to in the future.",
            "plan": "- Use the 'write_file' command to save the number of gold medals China won in the 2023 Chengdu Universiade for future reference.\n- Once I have saved the information, I will use the 'finish' command to signal that I have completed my objectives.",
            "criticism": "I need to ensure that I write the information accurately and in a format that is easy to understand. I also need to remember to use the 'finish' command once I have completed my objectives.",
            "speak": "I have found that China won 103 gold medals in the 2023 Chengdu Universiade. I am now going to save this information for future reference."
        },
        "command": {
            "name": "write_file",
            "args": {
                "file_path": "2023_Chengdu_Universiade_China_Gold_Medal_Count.txt",
                "text": "China won 103 gold medals in the 2023 Chengdu Universiade.",
                "append": false
            }
        }
    }
    System: Command write_file returned: File written successfully to 2023_Chengdu_Universiade_China_Gold_Medal_Count.txt.
    Human: Determine which next command to use, and respond using the format specified above:[0m
    
    [1m> Finished chain.[0m
    {
        "thoughts": {
            "text": "I have successfully written the information to a file. Now, I need to signal that I have completed my objectives. I will use the 'finish' command to do this.",
            "reasoning": "The 'finish' command is the most appropriate in this situation because it allows me to signal that I have completed my objectives. In this case, I have found the information I was looking for and saved it for future reference, so I have completed my objectives.",
            "plan": "- Use the 'finish' command to signal that I have completed my objectives.",
            "criticism": "I need to ensure that I use the 'finish' command correctly to signal that I have completed my objectives.",
            "speak": "I have successfully saved the information and completed my objectives."
        },
        "command": {
            "name": "finish",
            "args": {
                "response": "I have successfully saved the information and completed my objectives."
            }
        }
    }





    'I have successfully saved the information and completed my objectives.'


