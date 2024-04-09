# 文本内容补全初探（Completions API）[Legacy]

使用 Completions API 实现各类文本生成任务


主要请求参数说明：


- **`model`** （string，必填）

  要使用的模型的 ID。可以参考 **模型端点兼容性表**。

- **`prompt`** （string or array，必填，Defaults to ）

  生成补全的提示，编码为字符串、字符串数组、token数组或token数组数组。

  注意，这是模型在训练过程中看到的文档分隔符，所以如果没有指定提示符，模型将像从新文档的开头一样生成。

- **`stream`** （boolean，选填，默认 false）

  当它设置为 true 时，API 会以 SSE（ Server Side Event ）方式返回内容，即会不断地输出内容直到完成响应，流通过 `data: [DONE]` 消息终止。

- **`max_tokens`** （integer，选填，默认是 16）

  补全时要生成的最大 token 数。

  提示 `max_tokens` 的 token 计数不能超过模型的上下文长度。大多数模型的上下文长度为 2048 个token（最新模型除外，它支持 4096）

- **`temperature`** （number，选填，默认是1）

  使用哪个采样温度，在 **0和2之间**。

  较高的值，如0.8会使输出更随机，而较低的值，如0.2会使其更加集中和确定性。

  通常建议修改这个（`temperature` ）或 `top_p` 但两者不能同时存在，二选一。

- **`n`** （integer，选填，默认为 1）

  每个 `prompt` 生成的补全次数。

  注意：由于此参数会生成许多补全，因此它会快速消耗token配额。小心使用，并确保对 `max_tokens` 和 `stop` 进行合理的设置。


## 生成英文文本


```python
import os
from openai import OpenAI
import openai


# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'

# 设置你的 OpenAI API 密钥
client = OpenAI(api_key="put key here")

data = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)
```


```python
print(data)
```

    Completion(id='cmpl-9C2jhfzc8t2uUeOGxtgNCGfsNI9Rx', choices=[CompletionChoice(finish_reason='stop', index=0, logprobs=None, text='\n\nThis is a test.')], created=1712657481, model='gpt-3.5-turbo-instruct', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=6, prompt_tokens=5, total_tokens=11))



```python
text = data.choices[0].text
```


```python
print(text)
```


​    
    This is a test.


## 生成中文文本


```python
data = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="讲10个给程序员听得笑话",
  max_tokens=1000,
  temperature=0.5
)
```


```python
text = data.choices[0].text
print(text)
```


​    
    1. 为什么程序员喜欢用黑色背景？因为黑色背景可以减少眼睛的疲劳，让他们更专注于代码。
    
    2. 为什么程序员总是把变量命名为i？因为i是最短的单词，节省时间和键盘敲击次数。
    
    3. 为什么程序员总是喜欢喝咖啡？因为咖啡可以提高代码的运行速度。
    
    4. 为什么程序员总是对电脑说“我爱你”？因为他们知道，如果电脑出了问题，它就会变得很顽固。
    
    5. 为什么程序员总是把bug称为“功能”？因为如果把它们称为bug，老板就会觉得他们做了很多错误。
    
    6. 为什么程序员总是喜欢用英语编程？因为他们不想和其他程序员分享自己的代码。
    
    7. 为什么程序员总是喜欢熬夜？因为他们知道，当他们睡觉时，服务器会更快。
    
    8. 为什么程序员总是喜欢用递归？因为他们喜欢把问题复杂化。
    
    9. 为什么程序员总是把代码放在GitHub上？因为他们知道，如果代码出了问题，可以把责任推给其他人。
    
    10. 为什么程序员总是喜欢玩电子游戏？因为他们觉得在游戏中解决问题比在现实生活中容易多了。


# 聊天机器人初探（Chat Completions API）

使用 Chat Completions API 实现对话任务

聊天补全(Chat Completions API)以消息列表作为输入，并返回模型生成的消息作为输出。尽管聊天格式旨在使多轮对话变得简单，但它同样适用于没有任何对话的单轮任务。

主要请求参数说明：


- **`model` （string，必填）**

  要使用的模型ID。有关哪些模型适用于Chat API的详细信息

- **`messages` （array，必填）**

  迄今为止描述对话的消息列表
    - **`role` （string，必填）**

  发送此消息的角色。`system` 、`user` 或 `assistant` 之一（一般用 user 发送用户问题，system 发送给模型提示信息）

    - **`content` （string，必填）**
    
      消息的内容
    
    - **`name` （string，选填）**
    
      此消息的发送者姓名。可以包含 a-z、A-Z、0-9 和下划线，最大长度为 64 个字符

- **`stream` （boolean，选填，是否按流的方式发送内容）**

  当它设置为 true 时，API 会以 SSE（ Server Side Event ）方式返回内容。SSE 本质上是一个长链接，会持续不断地输出内容直到完成响应。如果不是做实时聊天，默认false即可。

- **`max_tokens` （integer，选填）**

  在聊天补全中生成的最大 **tokens** 数。

  输入token和生成的token的总长度受模型上下文长度的限制。

- **`temperature` （number，选填，默认是 1）**

  采样温度，在 0和 2 之间。

  较高的值，如0.8会使输出更随机，而较低的值，如0.2会使其更加集中和确定性。

  通常建议修改这个（`temperature` ）或者 `top_p` ，但两者不能同时存在，二选一。


## 开启聊天模式
使用 `messages` 记录迄今为止对话的消息列表


```python
from openai import OpenAI
client = OpenAI(api_key="put api_key here")

messages=[
    {
        "role": "user", 
        "content": "Hello!"
    }
]


data = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages = messages
)

```


```python
print(data)
```

    ChatCompletion(id='chatcmpl-9C3DSgNLsnYXJvO3Eo2AZDAZU26oo', choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content='Hello! How can I assist you today?', role='assistant', function_call=None, tool_calls=None), logprobs=None)], created=1712659326, model='gpt-3.5-turbo-0125', object='chat.completion', system_fingerprint='fp_b28b39ffa8', usage=CompletionUsage(completion_tokens=9, prompt_tokens=9, total_tokens=18))



```python
# 从返回的数据中获取生成的消息
new_message = data.choices[0].message
# 打印 new_message
print(new_message)
```

    ChatCompletionMessage(content='Hello! How can I assist you today?', role='assistant', function_call=None, tool_calls=None)



```python
# 将消息追加到 messages 列表中
messages.append(new_message)
print(messages)
```

    [{'role': 'user', 'content': 'Hello!'}, ChatCompletionMessage(content='Hello! How can I assist you today?', role='assistant', function_call=None, tool_calls=None)]



```python
type(new_message)
```




    openai.types.chat.chat_completion_message.ChatCompletionMessage




```python
new_message.role
```




    'assistant'




```python
new_message.content
```




    'Hello! How can I assist you today?'




```python
messages.pop()
```




    ChatCompletionMessage(content='Hello! How can I assist you today?', role='assistant', function_call=None, tool_calls=None)




```python
print(messages)
```

    [{'role': 'user', 'content': 'Hello!'}]


#### Prompt: OpenAIObject -> Dict

```
打印 messages 列表后发现数据类型不对，messages 输出如下：

print(messages)

[{'role': 'user', 'content': 'Hello!'}, <OpenAIObject at 0x7f27582c13f0> JSON: {
  "content": "Hello! How can I assist you today?",
  "role": "assistant"
}]

将OpenAIObject 转换为一个如下数据类型格式：

    {
        "role": "user", 
        "content": "Hello!"
    }
```


```python
new_message = data.choices[0].message
new_message_dict = {"role": new_message.role, "content": new_message.content}
type(new_message_dict)
```




    dict




```python
print(new_message_dict)
```

    {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}



```python
# 将消息追加到 messages 列表中
messages.append(new_message_dict)
```


```python
print(messages)
```

    [{'role': 'user', 'content': 'Hello!'}, {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}]


#### 新一轮对话


```python
new_chat = {
    "role": "user",
    "content": "1.讲一个程序员才听得懂的冷笑话；2.今天是几号？3.明天星期几？"
}
```


```python
messages.append(new_chat)
```


```python
from pprint import pprint

pprint(messages)
```

    [{'content': 'Hello!', 'role': 'user'},
     {'content': 'Hello! How can I assist you today?', 'role': 'assistant'},
     {'content': '1.讲一个程序员才听得懂的冷笑话；2.今天是几号？3.明天星期几？', 'role': 'user'}]



```python
data = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages
)
```


```python
new_message = data.choices[0].message
# 打印 new_messages 
print(new_message)
```

    ChatCompletionMessage(content='1. 为什么程序员喜欢黑色的衣服？因为它们让错误消息更难看出来！\n2. 今天是10月27日。\n3. 明天是星期四。', role='assistant', function_call=None, tool_calls=None)



```python
# 打印 new_messages 内容
print(new_message.content)
```

    1. 为什么程序员喜欢黑色的衣服？因为它们让错误消息更难看出来！
    2. 今天是10月27日。
    3. 明天是星期四。



## 使用多种身份聊天对话

目前`role`参数支持3类身份： `system`, `user` `assistant`:



```python
# 构造聊天记录
messages=[
    {"role": "system", "content": "你是一个乐于助人的体育界专家。"},
    {"role": "user", "content": "2008年奥运会是在哪里举行的？"},
]
```


```python
import openai

data = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages
)
```


```python
message = data.choices[0].message.content
print(message)
```

    2008年奥运会是在中国的北京举行的。



```python
# 添加 GPT 返回结果到聊天记录
messages.append({"role": "assistant", "content": message})
```


```python
messages
```




    [{'role': 'system', 'content': '你是一个乐于助人的体育界专家。'},
     {'role': 'user', 'content': '2008年奥运会是在哪里举行的？'},
     {'role': 'assistant', 'content': '2008年奥运会是在中国的北京举行的。'}]




```python
# 第二轮对话
messages.append({"role": "user", "content": "1.金牌最多的是哪个国家？2.奖牌最多的是哪个国家？"})
```


```python
messages
```




    [{'role': 'system', 'content': '你是一个乐于助人的体育界专家。'},
     {'role': 'user', 'content': '2008年奥运会是在哪里举行的？'},
     {'role': 'assistant', 'content': '2008年奥运会是在中国的北京举行的。'},
     {'role': 'user', 'content': '1.金牌最多的是哪个国家？2.奖牌最多的是哪个国家？'}]




```python
data = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages
)
```


```python
message = data.choices[0].message.content
print(message)
```

    1. 2008年北京奥运会上，金牌最多的国家是中国，共获得51枚金牌。
    2. 在2008年北京奥运会上，奖牌最多的国家是美国，共获得110枚奖牌（包括36枚金牌、38枚银牌和36枚铜牌）。



```python
data = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{'role': 'user', 'content': '1.金牌最多的是哪个国家？2.奖牌最多的是哪个国家？'}]
)
```


```python
data.choices[0].message.content
```




    '1. 美国是金牌数量最多的国家。\n2. 美国是奖牌数量最多的国家。'

