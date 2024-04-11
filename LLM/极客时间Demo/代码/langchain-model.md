# LangChain 核心模块学习：Model I/O

`Model I/O` 是 LangChain 为开发者提供的一套面向 LLM 的标准化模型接口，包括模型输入（Prompts）、模型输出（Output Parsers）和模型本身（Models）。

- Prompts：模板化、动态选择和管理模型输入
- Models：以通用接口调用语言模型
- Output Parser：从模型输出中提取信息，并规范化内容


## 模型抽象 Model

- 语言模型(LLMs): LangChain 的核心组件。LangChain并不提供自己的LLMs，而是为与许多不同的LLMs（OpenAI、Cohere、Hugging Face等）进行交互提供了一个标准接口。
- 聊天模型(Chat Models): 语言模型的一种变体。虽然聊天模型在内部使用了语言模型，但它们提供的接口略有不同。与其暴露一个“输入文本，输出文本”的API不同，它们提供了一个以“聊天消息”作为输入和输出的接口。

（注：对比 OpenAI Completion API和 Chat Completion API）

## 语言模型（LLMs)

类继承关系：

```
BaseLanguageModel --> BaseLLM --> LLM --> <name>  # Examples: AI21, HuggingFaceHub, OpenAI
```

主要抽象:

```
LLMResult, PromptValue,
CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun,
CallbackManager, AsyncCallbackManager,
AIMessage, BaseMessage
```

### BaseLanguageModel Class


这个基类为语言模型定义了一个接口，该接口允许用户以不同的方式与模型交互（例如通过提示或消息）。`generate_prompt` 是其中的一个主要方法，它接受一系列提示，并返回模型的生成结果。


```python
# 定义 BaseLanguageModel 抽象基类，它从 Serializable, Runnable 和 ABC 继承
class BaseLanguageModel(
    Serializable, Runnable[LanguageModelInput, LanguageModelOutput], ABC
):
    """
    与语言模型交互的抽象基类。

    所有语言模型的封装器都应从 BaseLanguageModel 继承。

    主要提供三种方法：
    - generate_prompt: 为一系列的提示值生成语言模型输出。提示值是可以转换为任何语言模型输入格式的模型输入（如字符串或消息）。
    - predict: 将单个字符串传递给语言模型并返回字符串预测。
    - predict_messages: 将一系列 BaseMessages（对应于单个模型调用）传递给语言模型，并返回 BaseMessage 预测。

    每种方法都有对应的异步方法。
    """

    # 定义一个抽象方法 generate_prompt，需要子类进行实现
    @abstractmethod
    def generate_prompt(
        self,
        prompts: List[PromptValue],  # 输入提示的列表
        stop: Optional[List[str]] = None,  # 生成时的停止词列表
        callbacks: Callbacks = None,  # 回调，用于执行例如日志记录或流式处理的额外功能
        **kwargs: Any,  # 任意的额外关键字参数，通常会传递给模型提供者的 API 调用
    ) -> LLMResult:
        """
        将一系列的提示传递给模型并返回模型的生成。

        对于提供批处理 API 的模型，此方法应使用批处理调用。

        使用此方法时：
            1. 希望利用批处理调用，
            2. 需要从模型中获取的输出不仅仅是最顶部生成的值，
            3. 构建与底层语言模型类型无关的链（例如，纯文本完成模型与聊天模型）。

        参数:
            prompts: 提示值的列表。提示值是一个可以转换为与任何语言模型匹配的格式的对象（对于纯文本生成模型为字符串，对于聊天模型为 BaseMessages）。
            stop: 生成时使用的停止词。模型输出在这些子字符串的首次出现处截断。
            callbacks: 要传递的回调。用于执行额外功能，例如在生成过程中进行日志记录或流式处理。
            **kwargs: 任意的额外关键字参数。通常这些会传递给模型提供者的 API 调用。

        返回值:
            LLMResult，它包含每个输入提示的候选生成列表以及特定于模型提供者的额外输出。
        """
```


### BaseLLM Class

这段代码定义了一个名为 BaseLLM 的抽象基类。这个基类的主要目的是提供一个基本的接口来处理大型语言模型 (LLM)。

```python
# 定义 BaseLLM 抽象基类，它从 BaseLanguageModel[str] 和 ABC（Abstract Base Class）继承
class BaseLLM(BaseLanguageModel[str], ABC):
    """Base LLM abstract interface.
    
    It should take in a prompt and return a string."""

    # 定义可选的缓存属性，其初始值为 None
    cache: Optional[bool] = None

    # 定义 verbose 属性，该属性决定是否打印响应文本
    # 默认值使用 _get_verbosity 函数的结果
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""

    # 定义 callbacks 属性，其初始值为 None，并从序列化中排除
    callbacks: Callbacks = Field(default=None, exclude=True)

    # 定义 callback_manager 属性，其初始值为 None，并从序列化中排除
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)

    # 定义 tags 属性，这些标签会被添加到运行追踪中，其初始值为 None，并从序列化中排除
    tags: Optional[List[str]] = Field(default=None, exclude=True)
    """Tags to add to the run trace."""

    # 定义 metadata 属性，这些元数据会被添加到运行追踪中，其初始值为 None，并从序列化中排除
    metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""

    # 内部类定义了这个 pydantic 对象的配置
    class Config:
        """Configuration for this pydantic object."""

        # 允许使用任意类型
        arbitrary_types_allowed = True

```
这个基类使用了 Pydantic 的功能，特别是 Field 方法，用于定义默认值和序列化行为。BaseLLM 的子类需要提供实现具体功能的方法。

### LLM Class


这段代码定义了一个名为 LLM 的类，该类继承自 BaseLLM。这个类的目的是为了为用户提供一个简化的接口来处理LLM（大型语言模型），而不期望用户实现完整的 _generate 方法。

```python

# 继承自 BaseLLM 的 LLM 类
class LLM(BaseLLM):
    """Base LLM abstract class.

    The purpose of this class is to expose a simpler interface for working
    with LLMs, rather than expect the user to implement the full _generate method.
    """

    # 使用 @abstractmethod 装饰器定义一个抽象方法，子类需要实现这个方法
    @abstractmethod
    def _call(
        self,
        prompt: str,  # 输入提示
        stop: Optional[List[str]] = None,  # 停止词列表
        run_manager: Optional[CallbackManagerForLLMRun] = None,  # 运行管理器
        **kwargs: Any,  # 其他关键字参数
    ) -> str:
        """Run the LLM on the given prompt and input."""
        # 此方法的实现应在子类中提供

    # _generate 方法使用了 _call 方法，用于处理多个提示
    def _generate(
        self,
        prompts: List[str],  # 多个输入提示的列表
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: 在此处添加缓存逻辑
        generations = []  # 用于存储生成的文本
        # 检查 _call 方法的签名是否支持 run_manager 参数
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        for prompt in prompts:  # 遍历每个提示
            # 根据是否支持 run_manager 参数来选择调用方法
            text = (
                self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else self._call(prompt, stop=stop, **kwargs)
            )
            # 将生成的文本添加到 generations 列表中
            generations.append([Generation(text=text)])
        # 返回 LLMResult 对象，其中包含 generations 列表
        return LLMResult(generations=generations)
```

### LLMs 已支持模型清单

### 使用 LangChain 调用 OpenAI GPT Completion API

#### BaseOpenAI Class

```python
class BaseOpenAI(BaseLLM):
    """OpenAI 大语言模型的基类。"""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field("text-davinci-003", alias="model")
    """使用的模型名。"""
    temperature: float = 0.7
    """要使用的采样温度。"""
    max_tokens: int = 256
    """完成中生成的最大令牌数。 
    -1表示根据提示和模型的最大上下文大小返回尽可能多的令牌。"""
    top_p: float = 1
    """在每一步考虑的令牌的总概率质量。"""
    frequency_penalty: float = 0
    """根据频率惩罚重复的令牌。"""
    presence_penalty: float = 0
    """惩罚重复的令牌。"""
    n: int = 1
    """为每个提示生成多少完成。"""
    best_of: int = 1
    """在服务器端生成best_of完成并返回“最佳”。"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """保存任何未明确指定的`create`调用的有效模型参数。"""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    # 支持OpenAI的显式代理
    openai_proxy: Optional[str] = None
    batch_size: int = 20
    """传递多个文档以生成时使用的批处理大小。"""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """向OpenAI完成API的请求超时。 默认为600秒。"""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """调整生成特定令牌的概率。"""
    max_retries: int = 6
    """生成时尝试的最大次数。"""
    streaming: bool = False
    """是否流式传输结果。"""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """允许的特殊令牌集。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """不允许的特殊令牌集。"""
    tiktoken_model_name: Optional[str] = None
    """使用此类时传递给tiktoken的模型名。
    Tiktoken用于计算文档中的令牌数量以限制它们在某个限制以下。
    默认情况下，设置为None时，这将与嵌入模型名称相同。
    但是，在某些情况下，您可能希望使用此嵌入类与tiktoken不支持的模型名称。
    这可以包括使用Azure嵌入或使用多个模型提供商的情况，这些提供商公开了类似OpenAI的API但模型不同。
    在这些情况下，为了避免在调用tiktoken时出错，您可以在此处指定要使用的模型名称。"""
```

### OpenAI LLM 模型默认使用 gpt-3.5-turbo-instruct


```python
from langchain_openai import OpenAI
import openai
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'
os.environ['OPENAI_API_KEY'] = 'api here'

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
```

#### 对比直接调用 OpenAI API：

```python
from openai import OpenAI

client = OpenAI()

data = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Tell me a Joke",
)
```


```python
print(llm("Tell me a Joke"))
```


​    
    Why don't scientists trust atoms?
    Because they make up everything.



```python
print(llm("讲10个给程序员听得笑话"))
```


​    
    1. 为什么程序员喜欢用纸杯？
    因为它有很多bug，可以一直重复使用。
    
    2. 为什么程序员不喜欢跳舞？
    因为他们不善于处理跨平台的问题。
    
    3. 为什么程序员总是迟到？
    因为他们总是低估完成时间。
    
    4. 为什么程序员不喜欢做饭？
    因为他们总是喜欢用主键来切菜。
    
    5. 为什么程序员经常熬夜？
    因为他们总是想要加班。
    
    6. 为什么程序员喜欢用黑色的键盘？
    因为黑色的键盘能让他们的代码看起来更酷。
    
    7. 为什么程序员不喜欢健身？
    因为他们觉得锻炼肌肉并不能提高代码质量。


​    


#### 生成10个笑话，显然超过了 max_token 默认值


```python
llm.max_tokens
```




    256




```python
# 修改 max_token 值为 1024
llm.max_tokens = 1024
```


```python
llm.max_tokens
```




    1024



#### LangChain 的 LLM 抽象维护了 OpenAI 连接状态（参数设定）


```python
result = llm("讲10个给程序员听得笑话")
print(result)
```


​    
    1. 为什么程序员喜欢在黑暗中工作？因为他们喜欢用命令行！
    2. 一个程序员走进一家酒吧，他说：“我想要一杯水。”酒吧老板说：“为什么不要一杯酒？”程序员回答：“我今天已经喝了太多bug了。”
    3. 为什么程序员总是用黑色键盘？因为他们喜欢黑客帝国！
    4. 为什么程序员总是喜欢在深夜工作？因为那时候代码会更黑。
    5. 为什么程序员总是带着耳机工作？因为他们喜欢听着“码农歌曲”。
    6. 为什么程序员总是说自己的代码很优雅？因为他们喜欢在代码中留下自己的印记。
    7. 有一天，一个程序员被问到：“你的代码为什么这么优秀？”他回答：“因为我有一只神奇的猴子，每次我遇到bug，它就会帮我解决。”老板问：“那么你的代码为什么还会有bug呢？”程序员回答：“那是因为我不养猴子！”
    8. 为什么程序员总是在写代码时喝咖啡？因为他们需要一些Java。
    9. 有一天，一个程序员走进一家餐厅，点了一份鸡肉，然后餐厅老板问：“你要吃骨头吗？”程序员回答：“不，我已经吃够了。”
    10. 为什么程序员总是喜欢用英语命名变量？因为他们不喜欢中文的变量名，总是觉得太拗口。


#### 再次生成10个笑话时成功了，但是两次笑话不一样

将 `temperature` 参数设置为0（值越大生成多样性越高）


```python
llm.temperature=0
```


```python
result = llm("生成可执行的快速排序 Python 代码")
print(result)
```


​    
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left = [x for x in arr[1:] if x <= pivot]
        right = [x for x in arr[1:] if x > pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)



```python
# 使用 `exec` 定义 `quick_sort` 函数
exec(result)
```


```python
# 调用 GPT 生成的快排代码，测试是否可用
print(quick_sort([3,6,8,10,1,2,1,1024]))
```

    [1, 1, 2, 3, 6, 8, 10, 1024]


## 聊天模型（Chat Models)

类继承关系：

```
BaseLanguageModel --> BaseChatModel --> <name>  # Examples: ChatOpenAI, ChatGooglePalm
```

主要抽象：

```
AIMessage, BaseMessage, HumanMessage
```


### BaseChatModel Class

```python
class BaseChatModel(BaseLanguageModel[BaseMessageChunk], ABC):
    cache: Optional[bool] = None
    """是否缓存响应。"""
    verbose: bool = Field(default_factory=_get_verbosity)
    """是否打印响应文本。"""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """添加到运行追踪的回调函数。"""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """添加到运行追踪的回调函数管理器。"""
    tags: Optional[List[str]] = Field(default=None, exclude=True)
    """添加到运行追踪的标签。"""
    metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    """添加到运行追踪的元数据。"""

    # 需要子类实现的 _generate 抽象方法
    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

```

### ChatOpenAI Class（调用 Chat Completion API）


```python
class ChatOpenAI(BaseChatModel):
    """OpenAI Chat大语言模型的包装器。

    要使用，您应该已经安装了``openai`` python包，并且
    环境变量``OPENAI_API_KEY``已使用您的API密钥进行设置。

    即使未在此类上明确保存，也可以传入任何有效的参数
    至openai.create调用。
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any = None  #: :meta private:
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """要使用的模型名。"""
    temperature: float = 0.7
    """使用的采样温度。"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """保存任何未明确指定的`create`调用的有效模型参数。"""
    openai_api_key: Optional[str] = None
    """API请求的基础URL路径，
    如果不使用代理或服务仿真器，请留空。"""
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    # 支持OpenAI的显式代理
    openai_proxy: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """请求OpenAI完成API的超时。默认为600秒。"""
    max_retries: int = 6
    """生成时尝试的最大次数。"""
    streaming: bool = False
    """是否流式传输结果。"""
    n: int = 1
    """为每个提示生成的聊天完成数。"""
    max_tokens: Optional[int] = None
    """生成的最大令牌数。"""
    tiktoken_model_name: Optional[str] = None
    """使用此类时传递给tiktoken的模型名称。
    Tiktoken用于计算文档中的令牌数以限制
    它们在某个限制之下。默认情况下，当设置为None时，这将
    与嵌入模型名称相同。但是，在某些情况下，
    您可能希望使用此嵌入类，模型名称不
    由tiktoken支持。这可能包括使用Azure嵌入或
    使用其中之一的多个模型提供商公开类似OpenAI的
    API但模型不同。在这些情况下，为了避免在调用tiktoken时出错，
    您可以在这里指定要使用的模型名称。"""


```


```python
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
```

对比调用 OpenAI API：

```python
import openai

data = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
```


```python
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

messages = [SystemMessage(content="You are a helpful assistant."),
 HumanMessage(content="Who won the world series in 2020?"),
 AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."), 
 HumanMessage(content="Where was it played?")]
```


```python
print(messages)
```

    [SystemMessage(content='You are a helpful assistant.'), HumanMessage(content='Who won the world series in 2020?'), AIMessage(content='The Los Angeles Dodgers won the World Series in 2020.'), HumanMessage(content='Where was it played?')]



```python
chat_model(messages)
```

    /opt/anaconda3/envs/LLM/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(





    AIMessage(content='The 2020 World Series was played at Globe Life Field in Arlington, Texas.', response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 53, 'total_tokens': 70}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'stop', 'logprobs': None}, id='run-9510d2a4-1d99-4374-8766-2c9839d6fb60-0')




```python
chat_result = chat_model(messages)
```


```python
type(chat_result)
```




    langchain_core.messages.ai.AIMessage

