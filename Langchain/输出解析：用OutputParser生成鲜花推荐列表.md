# 输出解析：用OutputParser生成鲜花推荐列表



在Langchain中，针对不同的使用场景和目标，设计了各种输出解析器。



1. **列表解析器（List Parser）**：这个解析器用于处理模型生成的输出，当需要模型的输出是一个列表的时候使用。例如，如果你询问模型“列出所有鲜花的库存”，模型的回答应该是一个列表。
2. **日期时间解析器（Datetime Parser）**：这个解析器用于处理日期和时间相关的输出，确保模型的输出是正确的日期或时间格式。
3. **枚举解析器（Enum Parser）**：这个解析器用于处理预定义的一组值，当模型的输出应该是这组预定义值之一时使用。例如，如果你定义了一个问题的答案只能是“是”或“否”，那么枚举解析器可以确保模型的回答是这两个选项之一。
4. **结构化输出解析器（Structured Output Parser）**：这个解析器用于处理复杂的、结构化的输出。如果你的应用需要模型生成具有特定结构的复杂回答（例如一份报告、一篇文章等），那么可以使用结构化输出解析器来实现。
5. **Pydantic（JSON）解析器**：这个解析器用于处理模型的输出，当模型的输出应该是一个符合特定格式的 JSON 对象时使用。它使用 Pydantic 库，这是一个数据验证库，可以用于构建复杂的数据模型，并确保模型的输出符合预期的数据模型。
6. **自动修复解析器（Auto-Fixing Parser）**：这个解析器可以自动修复某些常见的模型输出错误。例如，如果模型的输出应该是一段文本，但是模型返回了一段包含语法或拼写错误的文本，自动修复解析器可以自动纠正这些错误。
7. **重试解析器（RetryWithErrorOutputParser）**：这个解析器用于在模型的初次输出不符合预期时，尝试修复或重新生成新的输出。例如，如果模型的输出应该是一个日期，但是模型返回了一个字符串，那么重试解析器可以重新提示模型生成正确的日期格式。



## Pydantic（JSON）解析器实战



### 第一步：创建模型实例

```python
# ------Part 1
# 设置OpenAI API密钥
import os
os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

# 创建模型实例
from langchain import OpenAI
model = OpenAI(model_name='gpt-3.5-turbo-instruct')
```



### 第二步：定义输出数据的格式

```python
# ------Part 2
# 创建一个空的DataFrame用于存储结果
import pandas as pd
df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")
```

在这里我们用到了负责数据格式验证的 Pydantic 库来创建带有类型注解的类 FlowerDescription，它可以自动验证输入数据，确保输入数据符合你指定的类型和其他验证条件。



### 第三步：创建输出解析器

在这一步中，我们创建输出解析器并获取输出格式指示。先使用 LangChain 库中的 PydanticOutputParser 创建了输出解析器，该解析器将用于解析模型的输出，以确保其符合 FlowerDescription 的格式。然后，使用解析器的 get_format_instructions 方法获取了输出格式的指示。

```python
# ------Part 3
# 创建输出解析器
from langchain.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
print("输出格式：",format_instructions)
```

打印：

~~~python
输出格式： The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": 
                                              {"title": "Foo", "description": "a list of strings", 
                                               "type": "array", 
                                               "items": {"type": "string"}}}, 
                               "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. 
The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"flower_type": {"description": "\u9c9c\u82b1\u7684\u79cd\u7c7b", "title": "Flower Type", "type": "string"}, "price": {"description": "\u9c9c\u82b1\u7684\u4ef7\u683c", "title": "Price", "type": "integer"}, "description": {"description": "\u9c9c\u82b1\u7684\u63cf\u8ff0\u6587\u6848", "title": "Description", "type": "string"}, "reason": {"description": "\u4e3a\u4ec0\u4e48\u8981\u8fd9\u6837\u5199\u8fd9\u4e2a\u6587\u6848", "title": "Reason", "type": "string"}}, "required": ["flower_type", "price", "description", "reason"]}
```
~~~



上面这个输出，这部分是通过 output_parser.get_format_instructions() 方法生成的，这是 Pydantic (JSON) 解析器的核心价值，值得你好好研究研究。同时它也算得上是一个很清晰的提示模板，能够为模型提供良好的指导，描述了模型输出应该符合的格式。（其中 description 中的中文被转成了 UTF-8 编码。）



它指示模型输出 JSON Schema 的形式，定义了一个有效的输出应该包含哪些字段，以及这些字段的数据类型。例如，它指定了 "flower_type" 字段应该是字符串类型，"price" 字段应该是整数类型。这个指示中还提供了一个例子，说明了什么是一个格式良好的输出。



### 第四步：创建提示模板

```python
# ------Part 4
# 创建提示模板
from langchain import PromptTemplate
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}"""

# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, 
       partial_variables={"format_instructions": format_instructions}) 

# 打印提示
print("提示：", prompt)
```



打印：



```python
提示： 
input_variables=['flower', 'price'] 

output_parser=None 

partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n
As an example, for the schema {
"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, 
"required": ["foo"]}}\n
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. 
The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\n
Here is the output schema:\n```\n
{"properties": {
"flower_type": {"title": "Flower Type", "description": "\\u9c9c\\u82b1\\u7684\\u79cd\\u7c7b", "type": "string"}, 
"price": {"title": "Price", "description": "\\u9c9c\\u82b1\\u7684\\u4ef7\\u683c", "type": "integer"}, 
"description": {"title": "Description", "description": "\\u9c9c\\u82b1\\u7684\\u63cf\\u8ff0\\u6587\\u6848", "type": "string"}, 
"reason": {"title": "Reason", "description": "\\u4e3a\\u4ec0\\u4e48\\u8981\\u8fd9\\u6837\\u5199\\u8fd9\\u4e2a\\u6587\\u6848", "type": "string"}}, 
"required": ["flower_type", "price", "description", "reason"]}\n```'} 

template='您是一位专业的鲜花店文案撰写员。
\n对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？\n
{format_instructions}' 

template_format='f-string' 

validate_template=True
```

这就是包含了 format_instructions 信息的提示模板。



1. input_variables=['flower', 'price']：这是一个包含你想要在模板中使用的输入变量的列表。我们在模板中使用了 'flower' 和 'price' 两个变量，后面我们会用具体的值（如玫瑰、20 元）来替换这两个变量。
2. output_parser=None：这是你可以选择在模板中使用的一个输出解析器。在此例中，我们并没有选择在模板中使用输出解析器，而是在模型外部进行输出解析，所以这里是 None。
3. partial_variables：包含了你想要在模板中使用，但在生成模板时无法立即提供的变量。在这里，我们通过 'format_instructions' 传入输出格式的详细说明。
4. template：这是模板字符串本身。它包含了你想要模型生成的文本的结构。在此例中，模板字符串是你询问鲜花描述的问题，以及关于输出格式的说明。
5. template_format='f-string'：这是一个表示模板字符串格式的选项。此处是 f-string 格式。
6. validate_template=True：表示是否在创建模板时检查模板的有效性。这里选择了在创建模板时进行检查，以确保模板是有效的。





### 第五步：生成提示，传入模型并解析输出



```python
# ------Part 5
for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower=flower, price=price)
    # 打印提示
    print("提示：", input)

    # 获取模型的输出
    output = model(input)

    # 解析模型的输出
    parsed_output = output_parser.parse(output)
    parsed_output_dict = parsed_output.dict()  # 将Pydantic格式转换为字典

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output.dict()

# 打印字典
print("输出的数据：", df.to_dict(orient='records'))
```



打印：

```python
输出的数据： 
[{'flower_type': 'Rose', 'price': 50, 'description': '玫瑰是最浪漫的花，它具有柔和的粉红色，有着浓浓的爱意，价格实惠，50元就可以拥有一束玫瑰。', 'reason': '玫瑰代表着爱情，是最浪漫的礼物，以实惠的价格，可以让您尽情体验爱的浪漫。'}, 
{'flower_type': '百合', 'price': 30, 'description': '这支百合，柔美的花蕾，在你的手中摇曳，仿佛在与你深情的交谈', 'reason': '营造浪漫氛围'}, 
{'flower_type': 'Carnation', 'price': 20, 'description': '艳丽缤纷的康乃馨，带给你温馨、浪漫的气氛，是最佳的礼物选择！', 'reason': '康乃馨是一种颜色鲜艳、芬芳淡雅、具有浪漫寓意的鲜花，非常适合作为礼物，而且20元的价格比较实惠。'}]
```



## 自动修复解析器（OutputFixingParser）实战



首先，让我们来设计一个解析时出现的错误。



```python
# 导入所需要的库和模块
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 使用Pydantic创建一个数据格式，表示花
class Flower(BaseModel):
    name: str = Field(description="name of a flower")
    colors: List[str] = Field(description="the colors of this flower")
# 定义一个用于获取某种花的颜色列表的查询
flower_query = "Generate the charaters for a random flower."

# 定义一个格式不正确的输出
misformatted = "{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"

# 创建一个用于解析输出的Pydantic解析器，此处希望解析为Flower格式
parser = PydanticOutputParser(pydantic_object=Flower)
# 使用Pydantic解析器解析不正确的输出
parser.parse(misformatted)
```



报错：

```python
langchain.schema.output_parser.OutputParserException: Failed to parse Flower from completion {'name': '康乃馨', 'colors': ['粉红色','白色']}. Got: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

这个错误消息来自 Python 的内建 JSON 解析器发现我们输入的 JSON 格式不正确。程序尝试用 PydanticOutputParser 来解析 JSON 字符串时，Python 期望属性名称被双引号包围，但在给定的 JSON 字符串中是单引号。



问题在于 misformatted 字符串的内容



```python
"{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"
```

应该改为：

```python
'{"name": "康乃馨", "colors": ["粉红色","白色","红色","紫色","黄色"]}'
```



尝试使用 OutputFixingParser 来帮助咱们自动解决类似的格式错误。



```python
# 从langchain库导入所需的模块
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser

# 设置OpenAI API密钥
import os
os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

# 使用OutputFixingParser创建一个新的解析器，该解析器能够纠正格式不正确的输出
new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

# 使用新的解析器解析不正确的输出
result = new_parser.parse(misformatted) # 错误被自动修正
print(result) # 打印解析后的输出结果
```

输出如下：

```python
name='Rose' colors=['red', 'pink', 'white']
```

OutputFixingParser 内部，调用了原有的 PydanticOutputParser，如果成功，就返回；如果失败，它会将格式错误的输出以及格式化的指令传递给大模型，并要求 LLM 进行相关的修复。





## 重试解析器（RetryWithErrorOutputParser）实战



OutputFixingParser 不错，但它只能做简单的格式修复。如果出错的不只是格式，比如，输出根本不完整，有缺失内容，那么仅仅根据输出和格式本身，是无法修复它的。



通过实现输出解析器中 parse_with_prompt 方法，LangChain 提供的重试解析器可以帮助我们利用大模型的推理能力根据原始提示找回相关信息。



```python
# 定义一个模板字符串，这个模板将用于生成提问
template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""

# 定义一个Pydantic数据格式，它描述了一个"行动"类及其属性
from pydantic import BaseModel, Field
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

# 使用Pydantic格式Action来初始化一个输出解析器
from langchain.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Action)

# 定义一个提示模板，它将用于向模型提问
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="What are the colors of Orchid?")

# 定义一个错误格式的字符串
bad_response = '{"action": "search"}'
parser.parse(bad_response) # 如果直接解析，它会引发一个错误
```



由于 bad_response 只提供了 action 字段，而没有提供 action_input 字段，这与 Action 数据格式的预期不符，所以解析会失败。



我们首先尝试用 OutputFixingParser 来解决这个错误。



```python
from langchain.output_parsers import OutputFixingParser
from langchain.chat_models import ChatOpenAI
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
parse_result = fix_parser.parse(bad_response)
print('OutputFixingParser的parse结果:',parse_result)
```



OutputFixingParser 的 parse 结果：

```
action='search' action_input='query'
```



尝试用**RetryWithErrorOutputParser**



```python
# 初始化RetryWithErrorOutputParser，它会尝试再次提问来得到一个正确的输出
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser, llm=OpenAI(temperature=0)
)
parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print('RetryWithErrorOutputParser的parse结果:',parse_result)
```

输出：

```
action='search' action_input='colors of Orchid'
```

