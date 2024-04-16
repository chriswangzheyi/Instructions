# LangChain 核心模块 - Chat Model 和 Chat Prompt Template

希望通过此示例，让大家深入理解 LangChain 的聊天模型。简而言之：
- `Chat Model` 不止是一个用于聊天对话的模型抽象，更重要的是提供了`多角色`提示能力（System,AI,Human,Function)。
- `Chat Prompt Template` 则为开发者提供了便捷维护`不同角色`的`提示模板`与`消息记录`的接口。



```python
import os

# 更换为自己的 Serp API KEY
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'
os.environ['OPENAI_API_KEY'] = 'key here'
```

## 温故：LangChain Chat Model 使用方法和流程

在最终调用 `Chat Model` 时，一定是直接传入`LangChain Schema Messages（消息记录）` 

```python
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

messages = [SystemMessage(content="You are a helpful assistant."),
 HumanMessage(content="Who won the world series in 2020?"),
 AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."), 
 HumanMessage(content="Where was it played?")]

print(messages)

chat_model(messages)
```

打印 messages 输出结果：
```
[
    SystemMessage(content="You are a helpful assistant.", additional_kwargs={}),
    HumanMessage(
        content="Who won the world series in 2020?", additional_kwargs={}, example=False
    ),
    AIMessage(
        content="The Los Angeles Dodgers won the World Series in 2020.",
        additional_kwargs={},
        example=False,
    ),
    HumanMessage(content="Where was it played?", additional_kwargs={}, example=False),
]
```

调用 chat_model(messages) 返回结果：

```
AIMessage(
    content="The 2020 World Series was played at Globe Life Field in Arlington, Texas.",
    additional_kwargs={},
    example=False,
)

```

## 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate

使用 `ChatPromptTemplate.from_messages` 方法，类似使用和维护`messages`的方式，构造 `chat_prompt_template`


```python
from langchain.schema import AIMessage, HumanMessage, SystemMessage
# 导入 Chat Model 即将使用的 Prompt Templates
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 翻译任务指令始终由 System 角色承担
template = (
    """You are a translation expert, proficient in various languages. \n
    Translates English to Chinese."""
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
```


```python
print(system_message_prompt)
```

    prompt=PromptTemplate(input_variables=[], template='You are a translation expert, proficient in various languages. \n\n    Translates English to Chinese.')



```python
# 待翻译文本由 Human 角色输入
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
```


```python
print(human_message_prompt)
```

    prompt=PromptTemplate(input_variables=['text'], template='{text}')



```python
# 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
```


```python
print(chat_prompt_template)
```

    input_variables=['text'] messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a translation expert, proficient in various languages. \n\n    Translates English to Chinese.')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['text'], template='{text}'))]


### 规范化 Python 复杂对象

- 使用在线工具 [Python Formatter](https://codebeautify.org/python-formatter-beautifier) 
- 规范化 `chat_prompt_template`后再查看
- 注意：不要同事输入多个复杂对象

```json
messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            output_parser=None,
            partial_variables={},
            template="You are a translation expert, proficient in various languages. \n\n    Translates English to Chinese.",
            template_format="f-string",
            validate_template=True,
        ),
        additional_kwargs={},
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["text"],
            output_parser=None,
            partial_variables={},
            template="{text}",
            template_format="f-string",
            validate_template=True,
        ),
        additional_kwargs={},
    ),
]

```


```python
# 生成用于翻译的 Chat Prompt
chat_prompt_template.format_prompt(text="I love programming.")
```




    ChatPromptValue(messages=[SystemMessage(content='You are a translation expert, proficient in various languages. \n\n    Translates English to Chinese.'), HumanMessage(content='I love programming.')])



## 使用 chat_prompt_template.to_messages 方法生成 Messages


```python
# 生成聊天模型真正可用的消息记录 Messages
chat_prompt = chat_prompt_template.format_prompt(text="I love programming.").to_messages()
```


```python
chat_prompt
```




    [SystemMessage(content='You are a translation expert, proficient in various languages. \n\n    Translates English to Chinese.'),
     HumanMessage(content='I love programming.')]



## 使用 Chat Model（GPT-3.5-turbo）实际执行翻译任务



```python
from langchain_openai import ChatOpenAI
# 为了翻译结果的稳定性，将 temperature 设置为 0
translation_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```


```python
translation_result = translation_model(chat_prompt)
```


```python
translation_result
```




    AIMessage(content='我喜欢编程。', response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 34, 'total_tokens': 42}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-fcd963e8-63e6-4830-9e86-ff64453c4f85-0')




```python
# 查看翻译结果
print(translation_result.content)
```

    我喜欢编程。


## 使用 LLMChain 简化重复构造 ChatPrompt


```python
from langchain.chains import LLMChain

# 无需再每次都使用 to_messages 方法构造 Chat Prompt
translation_chain = LLMChain(llm=translation_model, prompt=chat_prompt_template)
```


```python
# 等价于 translation_result.content (字符串类型)
chain_result = translation_chain.invoke({'text': "I love programming."})
```


```python
print(chain_result)
```

    {'text': '我喜欢编程。'}



```python
translation_chain.invoke({'text': "I love AI and Large Language Model."})
```




    {'text': '我喜欢人工智能和大型语言模型。'}




```python
translation_chain.invoke({'text': "[Fruit, Color, Price (USD)] [Apple, Red, 1.20] [Banana, Yellow, 0.50] [Orange, Orange, 0.80] [Strawberry, Red, 2.50] [Blueberry, Blue, 3.00] [Kiwi, Green, 1.00] [Mango, Orange, 1.50] [Grape, Purple, 2.00]"})

```




    '[水果，颜色，价格（美元）] [苹果，红色，1.20] [香蕉，黄色，0.50] [橙子，橙色，0.80] [草莓，红色，2.50] [蓝莓，蓝色，3.00] [猕猴桃，绿色，1.00] [芒果，橙色，1.50] [葡萄，紫色，2.00]'



## 扩展：支持多语言对翻译


```python
# System 增加 source_language 和 target_language
template = (
    """You are a translation expert, proficient in various languages. \n
    Translates {source_language} to {target_language}."""
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
```


```python
# 待翻译文本由 Human 角色输入
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
```


```python
# 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
m_chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
```


```python
m_translation_chain = LLMChain(llm=translation_model, prompt=m_chat_prompt_template) 
```


```python
m_translation_chain.run({
    "source_language": "Chinese",
    "target_language": "English",
    "text": "我喜欢学习大语言模型，轻松简单又愉快",
})
```




    'I enjoy studying large language models, which are easy, simple, and enjoyable.'




```python
m_translation_chain.run({
    "source_language": "Chinese",
    "target_language": "Japanese",
    "text": "我喜欢学习大语言模型，轻松简单又愉快",
})
```




    '私は大規模言語モデルの学習が好きで、簡単で楽しいです。'



## Homework
- 尝试不同的 System Prompt 和 Chat Model，对比翻译效果。
- 根据翻译任务的使用场景，是否可以在初次传入 source_language 和 target_language 后不再更新？


```python

```
