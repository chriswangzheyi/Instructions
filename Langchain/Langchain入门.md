

# 准备安装包

```python
pip install langchain[llms]
```


```python
pip install openai
```

```python
pip install --upgrade langchain
```



# 环境设置

```python
from openai import OpenAI
api_key = 'code'
client = OpenAI(api_key=api_key)
```

# 调用 Text 模型


```python
response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  temperature=0.5,
  max_tokens=100,
  prompt="请给我的花店起个名")
```


```python
print(response.choices[0].text.strip())
```

    "花语小屋"


# 调用 Chat 模型


```python
response = client.chat.completions.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名"},
    ],
  temperature=0.8,
  max_tokens=60
)
```


```python
print(response.choices[0])
```

    Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='"花语秘境"', role='assistant', function_call=None, tool_calls=None))


# Chat 模型 vs Text 模型

Chat 模型和 Text 模型都有各自的优点，其适用性取决于具体的应用场景。

相较于 Text 模型，**Chat 模型的设计更适合处理对话或者多轮次交互的情况。**这是因为它可以接受一个消息列表作为输入，而不仅仅是一个字符串。这个消息列表可以包含 system、user 和 assistant 的历史信息，从而在处理交互式对话时提供更多的上下文信息。



这种设计的主要优点包括：

1.对话历史的管理：通过使用 Chat 模型，你可以更方便地管理对话的历史，并在需要时向模型提供这些历史信息。例如，你可以将过去的用户输入和模型的回复都包含在消息列表中，这样模型在生成新的回复时就可以考虑到这些历史信息。

2.角色模拟：通过 system 角色，你可以设定对话的背景，给模型提供额外的指导信息，从而更好地控制输出的结果。当然在 Text 模型中，你在提示中也可以为 AI 设定角色，作为输入的一部分。




然而，对于简单的单轮文本生成任务，使用 Text 模型可能会更简单、更直接。例如，如果你只需要模型根据一个简单的提示生成一段文本，那么 Text 模型可能更适合。从上面的结果看，Chat 模型给我们输出的文本更完善，是一句完整的话，而 Text 模型输出的是几个名字。这是因为 ChatGPT 经过了对齐（基于人类反馈的强化学习），输出的答案更像是真实聊天场景。

# 通过 LangChain 调用 Text模型


```python
import os
os.environ["OPENAI_API_KEY"] = api_key
from langchain.llms import OpenAI
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.8,
    max_tokens=60,)
response = llm.predict("请给我的花店起个名")
print(response)
```


    字
    
    "花漾花舍"


# 通过LangChain调用 Chat 模型


```python
import os
os.environ["OPENAI_API_KEY"] = api_key
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model="gpt-4",
                    temperature=0.8,
                    max_tokens=60)
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]

response = chat(messages)
print(response)
```


    content='"花语轩"' response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 34, 'total_tokens': 41}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-0fe6b964-494c-43f1-848d-40e160cb54e4-0'

