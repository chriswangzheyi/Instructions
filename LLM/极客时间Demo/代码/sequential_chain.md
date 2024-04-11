# LangChain 核心模块学习：Chains

对于简单的大模型应用，单独使用语言模型（LLMs）是可以的。

**但更复杂的大模型应用需要将 `LLMs` 和 `Chat Models` 链接在一起 - 要么彼此链接，要么与其他组件链接。**

LangChain 为这种“链式”应用程序提供了 `Chain` 接口。

LangChain 以通用方式定义了 `Chain`，它是对组件进行调用序列的集合，其中可以包含其他链。

## Chain Class 基类

类继承关系：

```
Chain --> <name>Chain  # Examples: LLMChain, MapReduceChain, RouterChain
```

```python
# 定义一个名为Chain的基础类
class Chain(Serializable, Runnable[Dict[str, Any], Dict[str, Any]], ABC):
    """为创建结构化的组件调用序列的抽象基类。
    
    链应该用来编码对组件的一系列调用，如模型、文档检索器、其他链等，并为此序列提供一个简单的接口。
    
    Chain接口使创建应用程序变得容易，这些应用程序是：
    - 有状态的：给任何Chain添加Memory可以使它具有状态，
    - 可观察的：向Chain传递Callbacks来执行额外的功能，如记录，这在主要的组件调用序列之外，
    - 可组合的：Chain API足够灵活，可以轻松地将Chains与其他组件结合起来，包括其他Chains。
    
    链公开的主要方法是：
    - `__call__`：链是可以调用的。`__call__`方法是执行Chain的主要方式。它将输入作为一个字典接收，并返回一个字典输出。
    - `run`：一个方便的方法，它以args/kwargs的形式接收输入，并将输出作为字符串或对象返回。这种方法只能用于一部分链，不能像`__call__`那样返回丰富的输出。
    """

    # 调用链
    def invoke(
        self, input: Dict[str, Any], config: Optional[runnableConfig] = None
    ) -> Dict[str, Any]:
        """传统调用方法。"""
        return self(input, **(config or {}))

    # 链的记忆，保存状态和变量
    memory: Optional[BaseMemory] = None
    """可选的内存对象，默认为None。
    内存是一个在每个链的开始和结束时被调用的类。在开始时，内存加载变量并在链中传递它们。在结束时，它保存任何返回的变量。
    有许多不同类型的内存，请查看内存文档以获取完整的目录。"""

    # 回调，可能用于链的某些操作或事件。
    callbacks: Callbacks = Field(default=None, exclude=True)
    """可选的回调处理程序列表（或回调管理器）。默认为None。
    在对链的调用的生命周期中，从on_chain_start开始，到on_chain_end或on_chain_error结束，都会调用回调处理程序。
    每个自定义链可以选择调用额外的回调方法，详细信息请参见Callback文档。"""

    # 是否详细输出模式
    verbose: bool = Field(default_factory=_get_verbosity)
    """是否以详细模式运行。在详细模式下，一些中间日志将打印到控制台。默认值为`langchain.verbose`。"""

    # 与链关联的标签
    tags: Optional[List[str]] = None
    """与链关联的可选标签列表，默认为None。
    这些标签将与对这个链的每次调用关联起来，并作为参数传递给在`callbacks`中定义的处理程序。
    你可以使用这些来例如识别链的特定实例与其用例。"""

    # 与链关联的元数据
    metadata: Optional[Dict[str, Any]] = None
    """与链关联的可选元数据，默认为None。
    这些元数据将与对这个链的每次调用关联起来，并作为参数传递给在`callbacks`中定义的处理程序。
    你可以使用这些来例如识别链的特定实例与其用例。"""
```

## LLMChain

LLMChain 是 LangChain 中最简单的链，作为其他复杂 Chains 和 Agents 的内部调用，被广泛应用。

一个LLMChain由PromptTemplate和语言模型（LLM or Chat Model）组成。它使用直接传入（或 memory 提供）的 key-value 来规范化生成 Prompt Template（提示模板），并将生成的 prompt （格式化后的字符串）传递给大模型，并返回大模型输出。


```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import openai
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'
os.environ['OPENAI_API_KEY'] = 'api code here'

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)
```


```python
prompt = PromptTemplate(
    input_variables=["product"],
    template="给制造{product}的有限公司取10个好名字，并给出完整的公司名称",
)
```


```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({
    'product': "性能卓越的GPU"
    }))
```


​    
    1. 创想视界科技有限公司 (ImagiTech Limited)
    2. 动力图形科技有限公司 (PowerGraphix Limited)
    3. 融智绘图技术有限公司 (IntelliDraw Technology Limited)
    4. 未来映像科技有限公司 (FutureVisionTech Limited)
    5. 火焰引擎设计有限公司 (FireEngine Design Limited)
    6. 顶点创新科技有限公司 (Vertex Innovations Limited)
    7. 幻像渲染技术有限公司 (IllusionRender Technology Limited)
    8. 极速渲染系统有限公司 (TurboRender Systems Limited)
    9. 超级像素科技有限公司 (SuperPixel Technologies Limited)
    10. 异形绘图工程有限公司 (OutShape Graphics Engineering Limited)



```python
chain.verbose = True
```


```python
chain.verbose
```




    True




```python
print(chain.run({
    'product': "性能卓越的GPU"
    }))
```


​    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3m给制造性能卓越的GPU的有限公司取10个好名字，并给出完整的公司名称[0m
    
    [1m> Finished chain.[0m


​    
    1. 创新视界科技有限公司 (VisionTech Innovation Co., Ltd.)
    2. 光驰科技有限公司 (OptimaTech Co., Ltd.)
    3. 强力极限科技有限公司 (ExtremeForce Tech Co., Ltd.)
    4. 巨能芯片科技有限公司 (MegaChipTech Co., Ltd.)
    5. 星际流动科技有限公司 (InterstellarFlow Tech Co., Ltd.)
    6. 稳定崛起科技有限公司 (SteadyRise Tech Co., Ltd.)
    7. 环球力量科技有限公司 (GlobalPowerTech Co., Ltd.)
    8. 卓越创新科技有限公司 (SuperiorInnovation Tech Co., Ltd.)
    9. 飞速进化科技有限公司 (RapidEvolution Tech Co., Ltd.)
    10. 至美极限科技有限公司 (UltimatePerfection Tech Co., Ltd.) 



```python

```

## Sequential Chain

串联式调用语言模型（将一个调用的输出作为另一个调用的输入）。

顺序链（Sequential Chain ）允许用户连接多个链并将它们组合成执行特定场景的流水线（Pipeline）。有两种类型的顺序链：

- SimpleSequentialChain：最简单形式的顺序链，每个步骤都具有单一输入/输出，并且一个步骤的输出是下一个步骤的输入。
- SequentialChain：更通用形式的顺序链，允许多个输入/输出。

### 使用 SimpleSequentialChain 实现戏剧摘要和评论（单输入/单输出）



```python
# 这是一个 LLMChain，用于根据剧目的标题撰写简介。

llm = OpenAI(temperature=0.7, max_tokens=1000)

template = """你是一位剧作家。根据戏剧的标题，你的任务是为该标题写一个简介。

标题：{title}
剧作家：以下是对上述戏剧的简介："""

prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)
```


```python
# 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。
# llm = OpenAI(temperature=0.7, max_tokens=1000)
template = """你是《纽约时报》的戏剧评论家。根据剧情简介，你的工作是为该剧撰写一篇评论。

剧情简介：
{synopsis}

以下是来自《纽约时报》戏剧评论家对上述剧目的评论："""

prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)
```


```python
# 这是一个SimpleSequentialChain，按顺序运行这两个链
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
```


```python
review = overall_chain.run("三体人不是无法战胜的")
```


​    
    [1m> Entering new SimpleSequentialChain chain...[0m
    [36;1m[1;3m
    
    《三体人不是无法战胜的》是一部关于勇气、团结和希望的戏剧作品。故事的背景是一个被外星人入侵的未来世界，人类面临着前所未有的危机。在这个世界中，有一群叫做“三体人”的外星生物，他们的技术和力量远超人类，似乎无法被打败。但是，一群普通人类的勇敢和团结，却让他们发现了三体人并不是无敌的。在与三体人的殊死搏斗中，人类不断展现出自己的智慧和勇气，最终找到了战胜敌人的方法。这部戏剧通过展现人类的坚强意志和不屈不挠的精神，向观众传递出希望和力量的信息。它提醒我们，无论面对多大的困难，只要我们团结一心，就能战胜一切。[0m
    [33;1m[1;3m
    
    《三体人不是无法战胜的》是一部令人振奋的戏剧作品，它通过展现人类的勇气、团结和希望，向观众传递出一种强大的力量。故事的背景是一个被外星人入侵的未来世界，人类面临着前所未有的危机。而在这样的绝境中，一群普通人类的勇敢和团结，却让他们发现了三体人并不是无敌的。
    
    剧中的人物形象栩栩如生，每个角色都有着自己的故事和背景，让观众能够真切地感受到他们的内心世界。特别是主角们的成长过程，从最初的无助和恐惧，到最终的勇敢和坚定，让人们看到了人类的突破自我和战胜困难的力量。这种积极向上的精神，无疑会给观众带来深刻的共鸣。
    
    该剧还通过精彩的舞台表演和惊险的场景设计，将观众带入到一个充满未知和挑战的世界。在与三体人的殊死搏斗中，人类不断展现出自己的智慧和勇气，最终找到了战胜敌人的方法。这种在极限环境下的求生和突破，让人们感受到了人类的无穷潜力。
    
    最令人感动的是，该剧传递出的团结和希望的信息。在面对强大的敌人和无法预料的挑战时，人类团结一心，共同战胜困难，最终找到了胜利的方法。这种精神无疑会给观众带来一种强大的鼓舞和信心，让人们相信，只要我们团结一心，就能战胜一切。
    
    总的来说，《三体人不是无法战胜的》是一部充满希望和力量的戏剧作品。它通过展现人类的勇气、团结和希望，向观众传递出一种强大的力量，让我们相信，无论面对多大的困难，只要我们团结一心，就能战胜一切。这部戏剧不仅是一场视觉盛宴，更是一次精神的洗礼。我相信，观众们将会被它所感染，带来深刻的触动和思考。强烈推荐！[0m
    
    [1m> Finished chain.[0m



```python
review = overall_chain.run("星球大战第九季")
```


​    
    [1m> Entering new SimpleSequentialChain chain...[0m
    [36;1m[1;3m
    
    《星球大战第九季》是一部充满惊险、动作和冒险的史诗级戏剧。故事发生在遥远的宇宙中，讲述了帝国与反抗军之间的永恒斗争。在第八部中，反抗军取得了一次胜利，但帝国并未被击败。在这第九部中，帝国将展开最终决战，试图彻底摧毁反抗军。同时，反抗军内部也发生了分歧，两位关键人物光明面和黑暗面之间的冲突将决定整个宇宙的命运。在这场终极的星球大战中，谁将最终获胜？谁将被抛弃？谁将成为英雄？随着双方力量的不断对决，这场戏剧充满着悬念和意想不到的转折，将带给观众一场惊心动魄的视觉盛宴。不容错过的《星球大战第九季》将为观众带来超越想象的冒险旅程。[0m
    [33;1m[1;3m
    
    《星球大战第九季》是一部引人入胜的史诗级戏剧，它将带领观众们进入一个充满惊险、动作和冒险的宇宙。这部剧不仅延续了前几部的精彩故事，同时也在故事情节和视觉效果上有了更进一步的提升。
    
    故事情节紧凑而扣人心弦，帝国与反抗军之间的永恒斗争在第九部中达到了高潮。反抗军不仅要面对帝国的最终决战，还要面对内部的分歧和光明面与黑暗面之间的冲突。这些复杂的关系将决定整个宇宙的命运，为故事增添了更多的戏剧性和悬念。
    
    《星球大战第九季》在视觉效果方面也是令人惊叹的。剧组精心打造的宇宙场景和特效让观众仿佛置身其中，每一场战斗都让人身历其境。同时，两位关键人物光明面和黑暗面之间的对决也是一场视觉盛宴，令人惊叹不已。
    
    总的来说，《星球大战第九季》是一部精彩绝伦的戏剧，它不仅延续了前几部的魅力，同时也在故事和视觉上有了更进一步的提升。不容错过的《星球大战第九季》将为观众带来一场超越想象的冒险旅程。强烈推荐给所有《星球大战》的粉丝和喜爱史诗戏剧的观众们。[0m
    
    [1m> Finished chain.[0m


### 使用 SequentialChain 实现戏剧摘要和评论（多输入/多输出）



```python
# # 这是一个 LLMChain，根据剧名和设定的时代来撰写剧情简介。
llm = OpenAI(temperature=.7, max_tokens=1000)
template = """你是一位剧作家。根据戏剧的标题和设定的时代，你的任务是为该标题写一个简介。

标题：{title}
时代：{era}
剧作家：以下是对上述戏剧的简介："""

prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
# output_key
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis", verbose=True)
```


```python
# 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。

template = """你是《纽约时报》的戏剧评论家。根据该剧的剧情简介，你需要撰写一篇关于该剧的评论。

剧情简介：
{synopsis}

来自《纽约时报》戏剧评论家对上述剧目的评价："""

prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review", verbose=True)
```


```python
from langchain.chains import SequentialChain

m_overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True)
```


```python
m_overall_chain({"title":"三体人不是无法战胜的", "era": "二十一世纪的新中国"})
```


​    
    [1m> Entering new SequentialChain chain...[0m


​    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3m你是一位剧作家。根据戏剧的标题和设定的时代，你的任务是为该标题写一个简介。
    
    标题：三体人不是无法战胜的
    时代：二十一世纪的新中国
    剧作家：以下是对上述戏剧的简介：[0m


    /opt/anaconda3/envs/LLM/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(


​    
    [1m> Finished chain.[0m


​    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3m你是《纽约时报》的戏剧评论家。根据该剧的剧情简介，你需要撰写一篇关于该剧的评论。
    
    剧情简介：


​    
    在二十一世纪的新中国，人类社会已经进入了一个高度发达的科技时代。然而，人类面临的挑战也越来越多，其中最大的威胁来自于外星人种族“三体人”。
    
    这些拥有超强科技的外星人种族对人类社会造成了巨大的威胁，他们的目的是要征服地球并摧毁人类文明。面对这样的威胁，人类社会陷入了一片恐慌和混乱之中。
    
    然而，就在人类准备放弃希望的时候，一群年轻的科学家和战士站了出来。他们的使命是要阻止三体人的入侵，保卫人类文明的未来。在经历了无数的挑战和牺牲后，他们终于找到了对抗三体人的方法。
    
    在最终的决战中，人类和三体人展开了一场惊心动魄的战斗。最终，人类战胜了三体人，守护了自己的家园。而这场战争也让人类意识到，他们并不是无法战胜的，只要团结一心，勇敢面对挑战，就能战胜任何敌人。
    
    这部戏剧讲述了人类面对外来威胁时的勇气和团结精神，同时也反映了二十一世纪新中国的科技发展和人类社会面临的挑战。它也提醒我们，只有团结一心，人类才能战胜任何困难，创造出美好的未来。
    
    来自《纽约时报》戏剧评论家对上述剧目的评价：[0m
    
    [1m> Finished chain.[0m
    
    [1m> Finished chain.[0m





    {'title': '三体人不是无法战胜的',
     'era': '二十一世纪的新中国',
     'synopsis': '\n\n在二十一世纪的新中国，人类社会已经进入了一个高度发达的科技时代。然而，人类面临的挑战也越来越多，其中最大的威胁来自于外星人种族“三体人”。\n\n这些拥有超强科技的外星人种族对人类社会造成了巨大的威胁，他们的目的是要征服地球并摧毁人类文明。面对这样的威胁，人类社会陷入了一片恐慌和混乱之中。\n\n然而，就在人类准备放弃希望的时候，一群年轻的科学家和战士站了出来。他们的使命是要阻止三体人的入侵，保卫人类文明的未来。在经历了无数的挑战和牺牲后，他们终于找到了对抗三体人的方法。\n\n在最终的决战中，人类和三体人展开了一场惊心动魄的战斗。最终，人类战胜了三体人，守护了自己的家园。而这场战争也让人类意识到，他们并不是无法战胜的，只要团结一心，勇敢面对挑战，就能战胜任何敌人。\n\n这部戏剧讲述了人类面对外来威胁时的勇气和团结精神，同时也反映了二十一世纪新中国的科技发展和人类社会面临的挑战。它也提醒我们，只有团结一心，人类才能战胜任何困难，创造出美好的未来。',
     'review': '\n\n《三体人》是一部充满惊险和感动的戏剧，它不仅展现了人类面对外来威胁时的勇气和团结精神，也反映了二十一世纪新中国的科技发展和人类社会面临的挑战。该剧巧妙地将科幻元素与现实社会问题结合，为观众带来了一场思想上的冲击。\n\n剧中的故事情节扣人心弦，将观众带入一个充满未知和挑战的世界。每一个角色都有自己的鲜明个性和动人情感，他们的命运与人类的命运紧密相连，让观众在紧张的剧情中感受到情感的共鸣。\n\n同时，该剧也对人类社会面临的现实问题提出了深刻的思考。随着科技的进步，人类社会也面临着越来越多的挑战，如何应对这些挑战，保护自己的文明和家园，是每个人都需要思考的问题。\n\n总的来说，《三体人》是一部具有强烈现实意义的戏剧作品。它不仅让观众感受到了科幻的魅力，更让我们反思人类面临的挑战，以及团结和勇气的重要性。这部戏剧不仅是一场视听盛宴，更是一次心灵的洗礼，值得观众们用心去感受。'}

