# 调用模型：使用OpenAI API还是微调开源Llama2/ChatGLM？



## 背景

大语言模型，不止 ChatGPT 一种。调用 OpenAI 的 API，当然方便且高效，不过，如果我就是想用其他的模型（比如说开源的 Llama2 或者 ChatGLM），该怎么做？再进一步，如果我就是想在本机上从头训练出来一个新模型，然后在 LangChain 中使用自己的模型，又该怎么做？



Google 2018 年的论文名篇 Attention is all you need，提出了 Transformer 架构，也给这一次 AI 的腾飞点了火。Transformer 是几乎所有预训练模型的核心底层架构。**基于 Transformer 预训练所得的大规模语言模型也被叫做“基础模型”（Foundation Model 或 Base Model）**。



各种预训练模型：



![](Images/8.jpeg)



现今的预训练模型的趋势是参数越来越多，模型也越来越大，训练一次的费用可达几百万美元。



![](Images/9.jpeg)



## 预训练 + 微调的模式

经过预训练的大模型中所习得的语义信息和所蕴含的语言知识，能够非常容易地向下游任务迁移。NLP 应用人员可以对模型的头部或者部分参数根据自己的需要进行适应性的调整，这通常涉及在相对较小的有标注数据集上进行有监督学习，让模型适应特定任务的需求。



这就是对预训练模型的微调（Fine-tuning）。微调过程相比于从头训练一个模型要快得多，且需要的数据量也要少得多，这使得作为工程师的我们能够更高效地开发和部署各种 NLP 解决方案



![](Images/10.jpeg)

- **预训练：**在大规模无标注文本数据上进行模型的训练，目标是让模型学习自然语言的基础表达、上下文信息和语义知识，为后续任务提供一个通用的、丰富的语言表示基础。
- **微调：**在预训练模型的基础上，可以根据特定的下游任务对模型进行微调。现在你经常会听到各行各业的人说：我们的优势就是领域知识嘛！我们比不过国内外大模型，我们可以拿开源模型做垂直领域嘛！做垂类模型！—— 啥叫垂类？指的其实就是根据领域数据微调开源模型这件事儿。



这种预训练 + 微调的大模型应用模式优势明显。首先，预训练模型能够将大量的通用语言知识迁移到各种下游任务上，作为应用人员，我们不需要自己寻找语料库，从头开始训练大模型，这减少了训练时间和数据需求；其次，微调过程可以快速地根据特定任务进行优化，简化了模型部署的难度；最后，预训练 + 微调的架构具有很强的可扩展性，可以方便地应用于各种自然语言处理任务，大大提高了 NLP 技术在实际应用中的可用性和普及程度，给我们带来了巨大的便利。





## 用 HuggingFace 跑开源模型



```python
import os
# 设置 API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'token here'
```



```python
# 导入必要的库
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 加载预训练的模型
# 使用 device_map 参数将模型自动加载到可用的硬件设备上，例如GPU
model = AutoModelForCausalLM.from_pretrained(
          "meta-llama/Llama-2-7b-chat-hf", 
          device_map = 'auto')  

# 定义一个提示，希望模型基于此提示生成故事
prompt = "请给我讲个玫瑰的爱情故事?"

# 使用分词器将提示转化为模型可以理解的格式，并将其移动到GPU上
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 使用模型生成文本，设置最大生成令牌数为2000
outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

# 将生成的令牌解码成文本，并跳过任何特殊的令牌，例如[CLS], [SEP]等
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的响应
print(response)
```



- 导入 AutoTokenizer：这是一个用于自动加载预训练模型的相关分词器的工具。分词器负责将文本转化为模型可以理解的数字格式。
- 导入 AutoModelForCausalLM：这是用于加载因果语言模型（用于文本生成）的工具。
- 使用 from_pretrained 方法来加载预训练的分词器和模型。其中，device_map = 'auto' 是为了自动地将模型加载到可用的设备上，例如 GPU。
- 然后，给定一个提示（prompt）："请给我讲个玫瑰的爱情故事?"，并使用分词器将该提示转换为模型可以接受的格式，return_tensors="pt" 表示返回 PyTorch 张量。语句中的 .to("cuda") 是 GPU 设备格式转换，因为我在 GPU 上跑程序，不用这个的话会报错，如果你使用 CPU，可以试一下删掉它。
- 最后使用模型的 .generate() 方法生成响应。max_new_tokens=2000 限制生成的文本的长度。使用分词器的 .decode() 方法将输出的数字转化回文本，并且跳过任何特殊的标记。



打印：

![](Images/12.webp)



开源模型，尤其是 7B 的小模型和 Open AI 的 ChatGPT 还是有一定差距的。



## LangChain 和 HuggingFace 的接口



### 通过 HuggingFace Hub



```python
# 导入HuggingFace API Token
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '你的HuggingFace API Token'

# 导入必要的库
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# 初始化HF LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    #repo_id="meta-llama/Llama-2-7b-chat-hf",
)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """

# 创建Prompt          
prompt = PromptTemplate(template=template, input_variables=["question"])

# 调用LLM Chain --- 我们以后会详细讲LLM Chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
)

# 准备问题
question = "Rose is which type of flower?"

# 调用模型并返回结果
print(llm_chain.run(question))
```



### 通过 HuggingFace Pipeline

~~~python
# 指定预训练模型的名称
model = "meta-llama/Llama-2-7b-chat-hf"

# 从预训练模型中加载词汇器
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# 创建一个文本生成的管道
import transformers
import torch
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length = 1000
)

# 创建HuggingFacePipeline实例
from langchain import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline = pipeline, 
                          model_kwargs = {'temperature':0})

# 定义输入模板，该模板用于生成花束的描述
template = """
              为以下的花束生成一个详细且吸引人的描述：
              花束的详细信息：
              ```{flower_details}```
           """

# 使用模板创建提示
from langchain import PromptTemplate,  LLMChain
prompt = PromptTemplate(template=template, 
                     input_variables=["flower_details"])

# 创建LLMChain实例
from langchain import PromptTemplate
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 需要生成描述的花束的详细信息
flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"

# 打印生成的花束描述
print(llm_chain.run(flower_details))
~~~



附录：（transformers pipeline 的配置参数）

![](Images/11.webp)

生成结果：

![](Images/13.jpeg)



结果不好。



##  用 LangChain 调用自定义语言模型



### 不想上传模型？

假设你就是想训练属于自己的模型。而且出于商业秘密的原因，不想开源它，不想上传到 HuggingFace，就是要在本机运行模型。此时应该如何利用 LangChain 的功能？



以创建一个 LLM 的衍生类，自己定义模型。而 LLM 这个基类，则位于 langchain.llms.base 中，通过 from langchain.llms.base import LLM 语句导入。



### demo

从 https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main ，下载一个 llama-2-7b-chat.ggmlv3.q4_K_S.bin 模型，并保存在本地。它并不是原始的 Llama 模型，而是 TheBloke 用他的手段为我们量化过的新模型，你也可以理解成，**他已经为我们压缩或者说微调了 Llama 模型**。



> **量化**是 AI 模型大小和性能优化的常用技术，它将模型的权重简化到较少的位数，以减少模型的大小和计算需求，让大模型甚至能够在 CPU 上面运行。当你看到模型的后缀有 GGML 或者 GPTQ，就说明模型已经被量化过，其中 GPTQ 是一种仅适用于 GPU 的特定格式。GGML 专为 CPU 和 Apple M 系列设计，但也可以加速 GPU 上的某些层。llama-cpp-python 这个包就是为了实现 GGML 而制作的。



```python
pip install llama-cpp-python
```



```python
# 导入需要的库
from llama_cpp import Llama
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM

# 模型的名称和路径常量
MODEL_NAME = 'llama-2-7b-chat.ggmlv3.q4_K_S.bin'
MODEL_PATH = '/home/huangj/03_Llama/'

# 自定义的LLM类，继承自基础LLM类
class CustomLLM(LLM):
    model_name = MODEL_NAME

    # 该方法使用Llama库调用模型生成回复
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt) + 5
        # 初始化Llama模型，指定模型路径和线程数
        llm = Llama(model_path=MODEL_PATH+MODEL_NAME, n_threads=4)
        # 使用Llama模型生成回复
        response = llm(f"Q: {prompt} A: ", max_tokens=256)
        
        # 从返回的回复中提取文本部分
        output = response['choices'][0]['text'].replace('A: ', '').strip()

        # 返回生成的回复，同时剔除了问题部分和额外字符
        return output[prompt_length:]

    # 返回模型的标识参数，这里只是返回模型的名称
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    # 返回模型的类型，这里是"custom"
    @property
    def _llm_type(self) -> str:
        return "custom"
    

# 初始化自定义LLM类
llm = CustomLLM()

# 使用自定义LLM生成一个回复
result = llm("昨天有一个客户抱怨他买了花给女朋友之后，两天花就枯了，你说作为客服我应该怎么解释？")

# 打印生成的回复
print(result)
```



![](Images/14.webp)