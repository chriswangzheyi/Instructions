# 基于 Chat Completions API 实现外部函数调用

**2023年6月20日，OpenAI 官方在 Chat Completions API 原有的三种不同角色设定（System, Assistant, User）基础上，新增了 Function Calling 功能。**

`functions` 是 Chat Completion API 中的可选参数，用于提供函数定义。其目的是使 GPT 模型能够生成符合所提供定义的函数参数。请注意，API不会实际执行任何函数调用。开发人员需要使用GPT 模型输出来执行函数调用。

如果提供了`functions`参数，默认情况下，GPT 模型将决定在何时适当地使用其中一个函数。

可以通过将`function_call`参数设置为`{"name": "<insert-function-name>"}`来强制 API 使用指定函数。

同时，也支持通过将`function_call`参数设置为`"none"`来强制API不使用任何函数。

如果使用了某个函数，则响应中的输出将包含`"finish_reason": "function_call"`，以及一个具有该函数名称和生成的函数参数的`function_call`对象。




## 概述

本 Notebook 介绍了如何将 Chat Completions API 与外部函数结合使用，以扩展 GPT 模型的功能。包含以下2个部分：
- 如何使用 `functions` 参数
- 如何使用 `function_call` 参数
- 使用 GPT 模型生成函数和参数
- 实际执行 GPT 模型生成的函数（以 SQL 查询为例）

### 注意：本示例直接构造 HTTP 请求访问 OpenAI API，因此无需使用 openai Python SDK。


```python
!pip install scipy tenacity tiktoken openai requests
```

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple
    Requirement already satisfied: scipy in /opt/anaconda3/lib/python3.11/site-packages (1.11.4)
    Requirement already satisfied: tenacity in /opt/anaconda3/lib/python3.11/site-packages (8.2.2)
    Requirement already satisfied: tiktoken in /opt/anaconda3/lib/python3.11/site-packages (0.6.0)
    Requirement already satisfied: openai in /opt/anaconda3/lib/python3.11/site-packages (1.16.2)
    Requirement already satisfied: requests in /opt/anaconda3/lib/python3.11/site-packages (2.31.0)
    Requirement already satisfied: numpy<1.28.0,>=1.21.6 in /opt/anaconda3/lib/python3.11/site-packages (from scipy) (1.26.4)
    Requirement already satisfied: regex>=2022.1.18 in /opt/anaconda3/lib/python3.11/site-packages (from tiktoken) (2023.10.3)
    Requirement already satisfied: anyio<5,>=3.5.0 in /opt/anaconda3/lib/python3.11/site-packages (from openai) (4.2.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /opt/anaconda3/lib/python3.11/site-packages (from openai) (1.8.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /opt/anaconda3/lib/python3.11/site-packages (from openai) (0.27.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/anaconda3/lib/python3.11/site-packages (from openai) (1.10.12)
    Requirement already satisfied: sniffio in /opt/anaconda3/lib/python3.11/site-packages (from openai) (1.3.0)
    Requirement already satisfied: tqdm>4 in /opt/anaconda3/lib/python3.11/site-packages (from openai) (4.65.0)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /opt/anaconda3/lib/python3.11/site-packages (from openai) (4.9.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.11/site-packages (from requests) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.11/site-packages (from requests) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/lib/python3.11/site-packages (from requests) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.11/site-packages (from requests) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/lib/python3.11/site-packages (from httpx<1,>=0.23.0->openai) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)



```python
pip install termcolor
```

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple
    Requirement already satisfied: termcolor in /opt/anaconda3/envs/LLM/lib/python3.11/site-packages (2.4.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import json
import requests
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-3.5-turbo"

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'
```

### 定义工具函数

首先，让我们定义一些用于调用聊天完成 API 的实用工具，并维护和跟踪对话状态。


```python
# 使用了retry库，指定在请求失败时的重试策略。
# 这里设定的是指数等待（wait_random_exponential），时间间隔的最大值为40秒，并且最多重试3次（stop_after_attempt(3)）。
# 定义一个函数chat_completion_request，主要用于发送 聊天补全 请求到OpenAI服务器
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):

    # 设定请求的header信息，包括 API_KEY
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + "api key here",
    }

    # 设定请求的JSON数据，包括GPT 模型名和要进行补全的消息
    json_data = {"model": model, "messages": messages}

    # 如果传入了functions，将其加入到json_data中
    if functions is not None:
        json_data.update({"functions": functions})

    # 如果传入了function_call，将其加入到json_data中
    if function_call is not None:
        json_data.update({"function_call": function_call})

    # 尝试发送POST请求到OpenAI服务器的chat/completions接口
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        # 返回服务器的响应
        return response

    # 如果发送请求或处理响应时出现异常，打印异常信息并返回
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
```


```python
# 定义一个函数pretty_print_conversation，用于打印消息对话内容
def pretty_print_conversation(messages):

    # 为不同角色设置不同的颜色
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    # 遍历消息列表
    for message in messages:

        # 如果消息的角色是"system"，则用红色打印“content”
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"user"，则用绿色打印“content”
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"assistant"，并且消息中包含"function_call"，则用蓝色打印"function_call"
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant[function_call]: {message['function_call']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"assistant"，但是消息中不包含"function_call"，则用蓝色打印“content”
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant[content]: {message['content']}\n", role_to_color[message["role"]]))

        # 如果消息的角色是"function"，则用品红色打印“function”
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))
```

### 如何使用 functions 参数

这段代码定义了两个可以在程序中调用的函数，分别是获取当前天气和获取未来N天的天气预报。

每个函数(function)都有其名称、描述和需要的参数（包括参数的类型、描述等信息）。

我们将把这些传递给 Chat Completions API，以生成符合规范的函数。


```python
# 定义一个名为functions的列表，其中包含两个字典，这两个字典分别定义了两个功能的相关参数

# 第一个字典定义了一个名为"get_current_weather"的功能
functions = [
    {
        "name": "get_current_weather",  # 功能的名称
        "description": "Get the current weather",  # 功能的描述
        "parameters": {  # 定义该功能需要的参数
            "type": "object",
            "properties": {  # 参数的属性
                "location": {  # 地点参数
                    "type": "string",  # 参数类型为字符串
                    "description": "The city and state, e.g. San Francisco, CA",  # 参数的描述
                },
                "format": {  # 温度单位参数
                    "type": "string",  # 参数类型为字符串
                    "enum": ["celsius", "fahrenheit"],  # 参数的取值范围
                    "description": "The temperature unit to use. Infer this from the users location.",  # 参数的描述
                },
            },
            "required": ["location", "format"],  # 该功能需要的必要参数
        },
    },
    # 第二个字典定义了一个名为"get_n_day_weather_forecast"的功能
    {
        "name": "get_n_day_weather_forecast",  # 功能的名称
        "description": "Get an N-day weather forecast",  # 功能的描述
        "parameters": {  # 定义该功能需要的参数
            "type": "object",
            "properties": {  # 参数的属性
                "location": {  # 地点参数
                    "type": "string",  # 参数类型为字符串
                    "description": "The city and state, e.g. San Francisco, CA",  # 参数的描述
                },
                "format": {  # 温度单位参数
                    "type": "string",  # 参数类型为字符串
                    "enum": ["celsius", "fahrenheit"],  # 参数的取值范围
                    "description": "The temperature unit to use. Infer this from the users location.",  # 参数的描述
                },
                "num_days": {  # 预测天数参数
                    "type": "integer",  # 参数类型为整数
                    "description": "The number of days to forecast",  # 参数的描述
                }
            },
            "required": ["location", "format", "num_days"]  # 该功能需要的必要参数
        },
    },
]
```

这段代码首先定义了一个`messages`列表用来存储聊天的消息，然后向列表中添加了系统和用户的消息。

然后，它使用了之前定义的`chat_completion_request`函数发送一个请求，传入的参数包括消息列表和函数列表。

在接收到响应后，它从JSON响应中解析出助手的消息，并将其添加到消息列表中。

最后，它打印出 GPT 模型回复的消息。

**（如果我们询问当前天气，GPT 模型会回复让你给出更准确的问题。）**


```python
# 定义一个空列表messages，用于存储聊天的内容
messages = []

# 使用append方法向messages列表添加一条系统角色的消息
messages.append({
    "role": "system",  # 消息的角色是"system"
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."  # 消息的内容
})

# 向messages列表添加一条用户角色的消息
messages.append({
    "role": "user",  # 消息的角色是"user"
    "content": "What's the weather like today"  # 用户询问今天的天气情况
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取助手的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: What's the weather like today
    [0m
    [34massistant[content]: Sure, could you please provide me with the name of the city so I can check the weather for you?
    [0m



```python
type(assistant_message)
```




    dict



## 使用 GPT 模型生成函数和对应参数

下面这段代码先向messages列表中添加了用户的位置信息。

然后再次使用了chat_completion_request函数发起请求，只是这次传入的消息列表已经包括了用户的新消息。

在获取到响应后，它同样从JSON响应中解析出助手的消息，并将其添加到消息列表中。

最后，打印出助手的新的回复消息。


```python
# 向messages列表添加一条用户角色的消息，用户告知他们在苏格兰的格拉斯哥
messages.append({
    "role": "user",  # 消息的角色是"user"
    "content": "I'm in Shanghai, China."  # 用户的消息内容
})

# 再次使用定义的chat_completion_request函数发起一个请求，传入更新后的messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取助手的新的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的新的回复消息添加到messages列表中
messages.append(assistant_message)

pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: What's the weather like today
    [0m
    [34massistant[content]: Sure, could you please provide me with the name of the city so I can check the weather for you?
    [0m
    [32muser: I'm in Shanghai, China.
    [0m
    [34massistant[function_call]: {'name': 'get_current_weather', 'arguments': '{"location":"Shanghai, China","format":"celsius"}'}
    [0m


这段代码的逻辑大体与上一段代码相同，区别在于这次用户的询问中涉及到未来若干天（x天）的天气预报。

在获取到回复后，它同样从JSON响应中解析出助手的消息，并将其添加到消息列表中。

然后打印出助手的回复消息。

**（通过不同的prompt方式，我们可以让它针对我们告诉它的其他功能。）**


```python
# 初始化一个空的messages列表
messages = []

# 向messages列表添加一条系统角色的消息，要求不做关于函数参数值的假设，如果用户的请求模糊，应该寻求澄清
messages.append({
    "role": "system",  # 消息的角色是"system"
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
})

# 向messages列表添加一条用户角色的消息，用户询问在未来x天内苏格兰格拉斯哥的天气情况
messages.append({
    "role": "user",  # 消息的角色是"user"
    "content": "what is the weather going to be like in Shanghai, China over the next x days"
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取助手的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: what is the weather going to be like in Shanghai, China over the next x days
    [0m
    [34massistant[content]: Sure, I can help with that. Please provide the number of days you would like to forecast for Shanghai, China.
    [0m


**(GPT 模型再次要求我们澄清，因为它还没有足够的信息。在这种情况下，它已经知道预测的位置，但需要知道需要多少天的预测。)**

这段代码的主要目标是将用户指定的天数（5天）添加到消息列表中，然后再次调用chat_completion_request函数发起一个请求。

返回的响应中包含了助手对用户的回复，即未来5天的天气预报。

这个预报是基于用户指定的地点（上海）和天数（5天）生成的。

在代码的最后，它解析出返回的JSON响应中的第一个选项，这就是助手的回复消息。


```python
# 向messages列表添加一条用户角色的消息，用户指定接下来的天数为5天
messages.append({
    "role": "user",  # 消息的角色是"user"
    "content": "5 days"
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: what is the weather going to be like in Shanghai, China over the next x days
    [0m
    [34massistant[content]: Sure, I can help with that. Please provide the number of days you would like to forecast for Shanghai, China.
    [0m
    [32muser: 5 days
    [0m
    [34massistant[function_call]: {'name': 'get_n_day_weather_forecast', 'arguments': '{"location":"Shanghai, China","format":"celsius","num_days":5}'}
    [0m


#### 强制使用指定函数

我们可以通过使用`function_call`参数来强制GPT 模型使用指定函数，例如`get_n_day_weather_forecast`。

通过这种方式，可以让 GPT 模型学习如何使用该函数。


```python
# 在这个代码单元中，我们强制GPT 模型使用get_n_day_weather_forecast函数
messages = []  # 创建一个空的消息列表

# 添加系统角色的消息
messages.append({
    "role": "system",  # 角色为系统
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
})

# 添加用户角色的消息
messages.append({
    "role": "user",  # 角色为用户
    "content": "Give me a weather report for San Diego, USA."
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages、functions以及特定的function_call作为参数
chat_response = chat_completion_request(
    messages, functions=functions, function_call={"name": "get_n_day_weather_forecast"}
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: Give me a weather report for San Diego, USA.
    [0m
    [34massistant[function_call]: {'name': 'get_n_day_weather_forecast', 'arguments': '{"location":"San Diego, USA","format":"celsius","num_days":1}'}
    [0m


下面这段代码演示了在不强制使用特定函数（`get_n_day_weather_forecast`）的情况下，GPT 模型可能会选择不同的方式来回应用户的请求。对于给定的用户请求"Give me a weather report for San Diego, USA."，GPT 模型可能不会调用`get_n_day_weather_forecast`函数。


```python
# 如果我们不强制GPT 模型使用 get_n_day_weather_forecast，它可能不会使用
messages = []  # 创建一个空的消息列表

# 添加系统角色的消息
messages.append({
    "role": "system",  # 角色为系统
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
})

# 添加用户角色的消息
messages.append({
    "role": "user",  # 角色为用户
    "content": "Give me a weather report for San Diego, USA."
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(
    messages, functions=functions
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: Give me a weather report for San Diego, USA.
    [0m
    [34massistant[function_call]: {'name': 'get_current_weather', 'arguments': '{"location":"San Diego, USA","format":"celsius"}'}
    [0m


#### 强制不使用函数

然后，我们创建另一个消息列表，并添加系统和用户的消息。这次用户请求的是加拿大多伦多当前的天气（使用摄氏度）。

随后，代码再次调用`chat_completion_request`函数，

但这次在`function_call`参数中明确指定了"none"，表示GPT 模型在处理此请求时不能调用任何函数。

最后，代码解析返回的JSON响应，获取第一个选项的消息，即 GPT 模型的回应。


```python
# 创建另一个空的消息列表
messages = []

# 添加系统角色的消息
messages.append({
    "role": "system",  # 角色为系统
    "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
})

# 添加用户角色的消息
messages.append({
    "role": "user",  # 角色为用户
    "content": "Give me the current weather (use Celcius) for Toronto, Canada."
})

# 使用定义的chat_completion_request函数发起一个请求，传入messages、functions和function_call作为参数
chat_response = chat_completion_request(
    messages, functions=functions, function_call="none"
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)
```

    [31msystem: Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    [0m
    [32muser: Give me the current weather (use Celcius) for Toronto, Canada.
    [0m
    [34massistant[content]: Here is the current weather in Toronto, Canada in Celsius:
    
    - Temperature: 23°C
    - Weather: Sunny
    - Winds: 10 km/h
    
    Can I help you with anything else?
    [0m

