# LangChain 核心模块学习：Model I/O

`Model I/O` 是 LangChain 为开发者提供的一套面向 LLM 的标准化模型接口，包括模型输入（Prompts）、模型输出（Output Parsers）和模型本身（Models）。

- Prompts：模板化、动态选择和管理模型输入
- Models：以通用接口调用语言模型
- Output Parser：从模型输出中提取信息，并规范化内容


## 模型输入 Prompts

一个语言模型的提示是用户提供的一组指令或输入，用于引导模型的响应，帮助它理解上下文并生成相关和连贯的基于语言的输出，例如回答问题、完成句子或进行对话。


- 提示模板（Prompt Templates）：参数化的模型输入
- 示例选择器（Example Selectors）：动态选择要包含在提示中的示例


## 提示模板 Prompt Templates

**Prompt Templates 提供了一种预定义、动态注入、模型无关和参数化的提示词生成方式，以便在不同的语言模型之间重用模板。**

一个模板可能包括指令、少量示例以及适用于特定任务的具体背景和问题。

通常，提示要么是一个字符串（LLMs），要么是一组聊天消息（Chat Model）。


类继承关系:

```
BasePromptTemplate --> PipelinePromptTemplate
                       StringPromptTemplate --> PromptTemplate
                                                FewShotPromptTemplate
                                                FewShotPromptWithTemplates
                       BaseChatPromptTemplate --> AutoGPTPrompt
                                                  ChatPromptTemplate --> AgentScratchPadChatPromptTemplate



BaseMessagePromptTemplate --> MessagesPlaceholder
                              BaseStringMessagePromptTemplate --> ChatMessagePromptTemplate
                                                                  HumanMessagePromptTemplate
                                                                  AIMessagePromptTemplate
                                                                  SystemMessagePromptTemplate

PromptValue --> StringPromptValue
                ChatPromptValue
```




### 使用 PromptTemplate 类生成提升词

**通常，`PromptTemplate` 类的实例，使用Python的`str.format`语法生成模板化提示；也可以使用其他模板语法（例如jinja2）。**

#### 使用 from_template 方法实例化 PromptTemplate


```python
from langchain import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

# 使用 format 生成提示
prompt = prompt_template.format(adjective="funny", content="chickens")
print(prompt)
```

    Tell me a funny joke about chickens.



```python
print(prompt_template)
```

    input_variables=['adjective', 'content'] template='Tell me a {adjective} joke about {content}.'



```python
prompt_template = PromptTemplate.from_template(
    "Tell me a joke"
)
# 生成提示
prompt = prompt_template.format()
print(prompt)
```

    Tell me a joke


#### 使用构造函数（Initializer）实例化 PromptTemplate

使用构造函数实例化 `prompt_template` 时必须传入参数：`input_variables` 和 `template`。

在生成提示过程中，会检查输入变量与模板字符串中的变量是否匹配，如果不匹配，则会引发异常；


```python
invalid_prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke about {content}."
)
```

传入 content 后才能生成可用的 prompt


```python
valid_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}."
)
```


```python
print(valid_prompt)
```

    input_variables=['adjective', 'content'] template='Tell me a {adjective} joke about {content}.'



```python
valid_prompt.format(adjective="funny", content="chickens")
```




    'Tell me a funny joke about chickens.'




```python
prompt_template = PromptTemplate.from_template(
    "讲{num}个给程序员听得笑话"
)
```


```python
from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", max_tokens=1000)

prompt = prompt_template.format(num=2)
print(f"prompt: {prompt}")

result = llm(prompt)
print(f"result: {result}")
```

    prompt: 讲2个给程序员听得笑话


    /home/ubuntu/miniconda3/envs/langchain/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(


    result: ：
    
    1. 有一个程序员不小心把自己的代码里的密码提交到了 Github 上面。这可是一个非常大的失误，他立马赶紧删除了自己的代码。可是这个时候，他发现自己删除的代码比提交的代码还要多！这令他大惊失色，他赶紧查看了一下提交的代码，发现自己的密码在里面。可是令他更惊讶的是，他发现自己的密码明明是 123456，而提交的代码里面的密码却是 654321。
    
    2. 有一天，一个程序员发现自己的电脑上的一个文件夹里有一个文件名为“最后一次”的文件，他好奇地打开文件，发现里面只有一行代码： while(true) { }。他不知道这是什么意思，就把文件改名为“第一次”，保存后发现电脑蹦出了一个错误提示：“第一次已存在，请使用其他名称”。这时，他才意识到，这个文件夹里的文件名都是“最后一次”，说明这是一个无限循环，他的电脑已经被卡住了。



```python
print(llm(prompt_template.format(num=3)))
```

    
    
    1、程序员是一个充满激情的职业，当你的工作遇到瓶颈时，不要放弃，多找几个程序员来帮忙，因为“程序员的两个头比一个强”。
    
    2、有一天，程序员的老婆问他：“你是怎么做到每天上班都那么有激情的？”程序员回答：“因为我每天都在和Bug做斗争，我一定要打败它们！”
    
    3、程序员的生活就像是一场“等待的游戏”，等待编译、等待运行、等待调试，等来等去，最后都等得没有耐心了。


#### 使用 jinja2 生成模板化提示


```python
jinja2_template = "Tell me a {{ adjective }} joke about {{ content }}"
prompt = PromptTemplate.from_template(jinja2_template, template_format="jinja2")

prompt.format(adjective="funny", content="chickens")
```




    'Tell me a funny joke about chickens'




```python
print(prompt)
```

    input_variables=['adjective', 'content'] template='Tell me a {{ adjective }} joke about {{ content }}' template_format='jinja2'



```python

```


```python

```

#### 实测：生成多种编程语言版本的快速排序


```python
sort_prompt_template = PromptTemplate.from_template(
    "生成可执行的快速排序 {programming_language} 代码"
)
```


```python
print(llm(sort_prompt_template.format(programming_language="python")))
```

    
    
    def quickSort(array, low = 0, high = None):
        if high == None:
            high = len(array) - 1
        if low < high:
            pivot = partition(array, low, high)
            quickSort(array, low, pivot - 1)
            quickSort(array, pivot + 1, high)
    
    def partition(array, low, high):
        pivot = array[high]
        i = low - 1
        for j in range(low, high):
            if array[j] < pivot:
                i += 1
                array[i], array[j] = array[j], array[i]
        array[i+1], array[high] = array[high], array[i+1]
        return i + 1
    
    #测试代码
    array = [5, 3, 8, 1, 9, 2, 4, 7, 6]
    quickSort(array)
    print(array) #输出结果为 [1, 2, 3, 4, 5, 6, 7, 8, 9]



```python
print(llm(sort_prompt_template.format(programming_language="java")))
```

    
    +
    +```
    +public class QuickSort{
    +
    +    public static void quickSort(int[] arr, int left, int right){
    +        if(left >= right) return;
    +        int i = left, j = right;
    +        int temp = arr[left];
    +        while(i < j){
    +            while(i < j && arr[j] >= temp) j--;
    +            arr[i] = arr[j];
    +            while(i < j && arr[i] <= temp) i++;
    +            arr[j] = arr[i];
    +        }
    +        arr[i] = temp;
    +        quickSort(arr, left, i - 1);
    +        quickSort(arr, i + 1, right);
    +    }
    +
    +    public static void main(String[] args){
    +        int[] arr = new int[]{3, 2, 1, 5, 6, 4};
    +        quickSort(arr, 0, arr.length - 1);
    +        for(int i : arr){
    +            System.out.print(i + " ");
    +        }
    +    }
    +}
    +```
    



```python
print(llm(sort_prompt_template.format(programming_language="C++")))
```

    
    
    #include <iostream>
    using namespace std;
    
    // 交换两个数字的值
    void swap(int *a, int *b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
    
    // 获取基准值的索引
    int partition(int arr[], int low, int high) {
        // 初始化基准值为最后一位
        int pivot = arr[high];
        int i = (low - 1);
    
        for (int j = low; j <= high - 1; j++) {
            // 如果当前的值小于基准值，就将它放到左边
            if (arr[j] < pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        // 将基准值放到正确的位置上
        swap(&arr[i + 1], &arr[high]);
        // 返回基准值的索引
        return (i + 1);
    }
    
    // 排序函数
    void quickSort(int arr[], int low, int high) {
        if (low < high) {
            // 获取基准值的索引
            int pivot = partition(arr, low, high);
            // 对基准值左边的数组进行排序
            quickSort(arr, low, pivot - 1);
            // 对基准值右边的数组进行排序
            quickSort(arr, pivot + 1, high);
        }
    }
    
    // 打印数组
    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;
    }
    
    // 主函数
    int main() {
        int arr[] = { 10, 7, 8, 9, 1, 5 };
        int size = sizeof(arr) / sizeof(arr[0]);
        // 调用快速排序函数
        quickSort(arr, 0, size - 1);
        // 打印排序后的数组
        cout << "排序后的数组：" << endl;
        printArray(arr, size);
        return 0;
    }
    
    /* 输出：
    排序后的数组：
    1 5 7 8 9 10 
    */



```python

```

## 使用 ChatPromptTemplate 类生成适用于聊天模型的聊天记录

**`ChatPromptTemplate` 类的实例，使用`format_messages`方法生成适用于聊天模型的提示。**

### 使用 from_messages 方法实例化 ChatPromptTemplate


```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

# 生成提示
messages = template.format_messages(
    name="Bob",
    user_input="What is your name?"
)
```


```python
print(messages)
```

    [SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content='What is your name?')]



```python
print(messages[0].content)
print(messages[-1].content)
```

    You are a helpful AI bot. Your name is Bob.
    What is your name?



```python
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)
```

    /home/ubuntu/miniconda3/envs/langchain/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The class `langchain_community.chat_models.openai.ChatOpenAI` was deprecated in langchain-community 0.0.10 and will be removed in 0.2.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import ChatOpenAI`.
      warn_deprecated(



```python
chat_model(messages)
```

    /home/ubuntu/miniconda3/envs/langchain/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(





    AIMessage(content='My name is Bob. How can I assist you today?', response_metadata={'token_usage': {'completion_tokens': 12, 'prompt_tokens': 50, 'total_tokens': 62}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None})



### 摘要总结


```python
summary_template = ChatPromptTemplate.from_messages([
    ("system", "你将获得关于同一主题的{num}篇文章（用-----------标签分隔）。首先总结每篇文章的论点。然后指出哪篇文章提出了更好的论点，并解释原因。"),
    ("human", "{user_input}"),
])
```


```python
messages = summary_template.format_messages(
    num=3,
    user_input='''1. [PHP是世界上最好的语言]
PHP是世界上最好的情感派编程语言，无需逻辑和算法，只要情绪。它能被蛰伏在冰箱里的PHP大神轻易驾驭，会话结束后的感叹号也能传达对代码的热情。写PHP就像是在做披萨，不需要想那么多，只需把配料全部扔进一个碗，然后放到服务器上，热乎乎出炉的网页就好了。
-----------
2. [Python是世界上最好的语言]
Python是世界上最好的拜金主义者语言。它坚信：美丽就是力量，简洁就是灵魂。Python就像是那个永远在你皱眉的那一刻扔给你言情小说的好友。只有Python，你才能够在两行代码之间感受到飘逸的花香和清新的微风。记住，这世上只有一种语言可以使用空格来领导全世界的进步，那就是Python。
-----------
3. [Java是世界上最好的语言]
Java是世界上最好的德育课编程语言，它始终坚守了严谨、安全的编程信条。Java就像一个严格的老师，他不会对你怀柔，不会让你偷懒，也不会让你走捷径，但他教会你规范和自律。Java就像是那个喝咖啡也算加班费的上司，拥有对邪恶的深度厌恶和对善良的深度拥护。
'''
)
```


```python
print(messages[-1].content)
```

    1. [PHP是世界上最好的语言]
    PHP是世界上最好的情感派编程语言，无需逻辑和算法，只要情绪。它能被蛰伏在冰箱里的PHP大神轻易驾驭，会话结束后的感叹号也能传达对代码的热情。写PHP就像是在做披萨，不需要想那么多，只需把配料全部扔进一个碗，然后放到服务器上，热乎乎出炉的网页就好了。
    -----------
    2. [Python是世界上最好的语言]
    Python是世界上最好的拜金主义者语言。它坚信：美丽就是力量，简洁就是灵魂。Python就像是那个永远在你皱眉的那一刻扔给你言情小说的好友。只有Python，你才能够在两行代码之间感受到飘逸的花香和清新的微风。记住，这世上只有一种语言可以使用空格来领导全世界的进步，那就是Python。
    -----------
    3. [Java是世界上最好的语言]
    Java是世界上最好的德育课编程语言，它始终坚守了严谨、安全的编程信条。Java就像一个严格的老师，他不会对你怀柔，不会让你偷懒，也不会让你走捷径，但他教会你规范和自律。Java就像是那个喝咖啡也算加班费的上司，拥有对邪恶的深度厌恶和对善良的深度拥护。
    



```python
chat_result = chat_model(messages)
```


```python
print(chat_result.content)
```

    第一篇文章认为PHP是世界上最好的语言，强调它是一种情感派编程语言，无需逻辑和算法，只需情绪，比喻写PHP就像做披萨一样简单。第二篇文章则主张Python是最好的语言，强调其美丽和简洁性，称Python能够通过空格领导世界进步。第三篇文章认为Java是最好的语言，将其比作严格的老师，注重规范和自律，强调其严谨和安全性。
    
    在这三篇文章中，我认为第三篇关于Java的论点提出得更好。它强调了编程中的严谨和自律的重要性，将Java比作一个严格的老师，这种比喻更加贴近编程实践中的现实情况。相比之下，第一篇和第二篇的论点更多地侧重于情感和美学方面，缺乏对编程实质的深入讨论。因此，第三篇文章提出的论点更有说服力，因为它强调了编程中的重要原则和价值。



```python
messages = summary_template.format_messages(
    num=2,
    user_input='''1.认为“道可道”中的第一个“道”，指的是道理，如仁义礼智之类；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，所谓“道可道，非常道”，指的是可以言说的道理，不是恒久存在的“道”，恒久存在的“道”不可言说。如苏辙说：“莫非道也。而可道者不可常，惟不可道，而后可常耳。今夫仁义礼智，此道之可道者也。然而仁不可以为义，而礼不可以为智，可道之不可常如此。……而道常不变，不可道之能常如此。”蒋锡昌说：“此道为世人所习称之道，即今人所谓‘道理’也，第一‘道’字应从是解。《广雅·释诂》二：‘道，说也’，第二‘道’字应从是解。‘常’乃真常不易之义，在文法上为区别词。……第三‘道’字即二十五章‘道法自然’之‘道’，……乃老子学说之总名也”。陈鼓应说：“第一个‘道’字是人们习称之道，即今人所谓‘道理’。第二个‘道’字，是指言说的意思。第三个‘道’字，是老子哲学上的专有名词，在本章它意指构成宇宙的实体与动力。……‘常道’之‘常’，为真常、永恒之意。……可以用言词表达的道，就不是常道”。
-----------
2.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，“道可道，非常道”，指可以言说的“道”，就不是恒久存在的“道”。如张默生说：“‘道’，指宇宙的本体而言。……‘常’，是经常不变的意思。……可以说出来的道，便不是经常不变的道”。董平说：“第一个‘道’字与‘可道’之‘道’，内涵并不相同。第一个‘道’字，是老子所揭示的作为宇宙本根之‘道’；‘可道’之‘道’，则是‘言说’的意思。……这里的大意就是说：凡一切可以言说之‘道’，都不是‘常道’或永恒之‘道’”。汤漳平等说：“第一句中的三个‘道’，第一、三均指形上之‘道’，中间的‘道’作动词，为可言之义。……道可知而可行，但非恒久不变之道”。
--------
3.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，则指的是平常人所讲之道、常俗之道。因此，“道可道，非常道”，指“道”是可以言说的，但它不是平常人所谓的道或常俗之道。如李荣说：“道者，虚极之理也。夫论虚极之理，不可以有无分其象，不可以上下格其真。……圣人欲坦兹玄路，开以教门，借圆通之名，目虚极之理，以理可名，称之可道。故曰‘吾不知其名，字之曰道’。非常道者，非是人间常俗之道也。人间常俗之道，贵之以礼义，尚之以浮华，丧身以成名，忘己而徇利。”司马光说：“世俗之谈道者，皆曰道体微妙，不可名言。老子以为不然，曰道亦可言道耳，然非常人之所谓道也。……常人之所谓道者，凝滞于物。”裘锡圭说：“到目前为止，可以说，几乎从战国开始，大家都把‘可道’之‘道’……看成老子所否定的，把‘常道’‘常名’看成老子所肯定的。这种看法其实有它不合理的地方，……‘道’是可以说的。《老子》这个《道经》第一章，开宗明义是要讲他的‘道’。第一个‘道’字，理所应当，也是讲他要讲的‘道’：道是可以言说的。……那么这个‘恒’字应该怎么讲？我认为很简单，‘恒’字在古代作定语用，经常是‘平常’‘恒常’的意思。……‘道’是可以言说的，但是我要讲的这个‘道’，不是‘恒道’，它不是一般人所讲的‘道’。
'''
)
print(messages)
```

    [SystemMessage(content='你将获得关于同一主题的2篇文章（用-----------标签分隔）。首先总结每篇文章的论点。然后指出哪篇文章提出了更好的论点，并解释原因。'), HumanMessage(content='1.认为“道可道”中的第一个“道”，指的是道理，如仁义礼智之类；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，所谓“道可道，非常道”，指的是可以言说的道理，不是恒久存在的“道”，恒久存在的“道”不可言说。如苏辙说：“莫非道也。而可道者不可常，惟不可道，而后可常耳。今夫仁义礼智，此道之可道者也。然而仁不可以为义，而礼不可以为智，可道之不可常如此。……而道常不变，不可道之能常如此。”蒋锡昌说：“此道为世人所习称之道，即今人所谓‘道理’也，第一‘道’字应从是解。《广雅·释诂》二：‘道，说也’，第二‘道’字应从是解。‘常’乃真常不易之义，在文法上为区别词。……第三‘道’字即二十五章‘道法自然’之‘道’，……乃老子学说之总名也”。陈鼓应说：“第一个‘道’字是人们习称之道，即今人所谓‘道理’。第二个‘道’字，是指言说的意思。第三个‘道’字，是老子哲学上的专有名词，在本章它意指构成宇宙的实体与动力。……‘常道’之‘常’，为真常、永恒之意。……可以用言词表达的道，就不是常道”。\n-----------\n2.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，“道可道，非常道”，指可以言说的“道”，就不是恒久存在的“道”。如张默生说：“‘道’，指宇宙的本体而言。……‘常’，是经常不变的意思。……可以说出来的道，便不是经常不变的道”。董平说：“第一个‘道’字与‘可道’之‘道’，内涵并不相同。第一个‘道’字，是老子所揭示的作为宇宙本根之‘道’；‘可道’之‘道’，则是‘言说’的意思。……这里的大意就是说：凡一切可以言说之‘道’，都不是‘常道’或永恒之‘道’”。汤漳平等说：“第一句中的三个‘道’，第一、三均指形上之‘道’，中间的‘道’作动词，为可言之义。……道可知而可行，但非恒久不变之道”。\n--------\n3.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，则指的是平常人所讲之道、常俗之道。因此，“道可道，非常道”，指“道”是可以言说的，但它不是平常人所谓的道或常俗之道。如李荣说：“道者，虚极之理也。夫论虚极之理，不可以有无分其象，不可以上下格其真。……圣人欲坦兹玄路，开以教门，借圆通之名，目虚极之理，以理可名，称之可道。故曰‘吾不知其名，字之曰道’。非常道者，非是人间常俗之道也。人间常俗之道，贵之以礼义，尚之以浮华，丧身以成名，忘己而徇利。”司马光说：“世俗之谈道者，皆曰道体微妙，不可名言。老子以为不然，曰道亦可言道耳，然非常人之所谓道也。……常人之所谓道者，凝滞于物。”裘锡圭说：“到目前为止，可以说，几乎从战国开始，大家都把‘可道’之‘道’……看成老子所否定的，把‘常道’‘常名’看成老子所肯定的。这种看法其实有它不合理的地方，……‘道’是可以说的。《老子》这个《道经》第一章，开宗明义是要讲他的‘道’。第一个‘道’字，理所应当，也是讲他要讲的‘道’：道是可以言说的。……那么这个‘恒’字应该怎么讲？我认为很简单，‘恒’字在古代作定语用，经常是‘平常’‘恒常’的意思。……‘道’是可以言说的，但是我要讲的这个‘道’，不是‘恒道’，它不是一般人所讲的‘道’。\n')]



```python
chat_result = chat_model(messages)
print(chat_result.content)
```

    第一篇文章认为，“道可道，非常道”中的第一个“道”指的是可以言说的道理，而“常道”则指恒久存在的道。文章引用了不同学者的观点，强调了可言说的道理与恒久存在的道之间的区别。
    
    第二篇文章认为，“道可道，非常道”中的第一个“道”指的是宇宙万物的本原，而“常道”指恒久存在的道。通过引用不同学者的观点，文章强调了可言说的道与恒久存在的道之间的区别。
    
    第一篇文章提出了更好的论点。它通过引用不同学者的观点，清晰地解释了“道可道，非常道”中的每一个字的含义，并强调了可言说的道理与恒久存在的道之间的区别。这样的论点更加深入和逻辑清晰，让读者更容易理解“道”的不同含义。



```python
messages = summary_template.format_messages(
    num=2,
    user_input='''1.认为“道可道”中的第一个“道”，指的是道理，如仁义礼智之类；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，所谓“道可道，非常道”，指的是可以言说的道理，不是恒久存在的“道”，恒久存在的“道”不可言说。如苏辙说：“莫非道也。而可道者不可常，惟不可道，而后可常耳。今夫仁义礼智，此道之可道者也。然而仁不可以为义，而礼不可以为智，可道之不可常如此。……而道常不变，不可道之能常如此。”蒋锡昌说：“此道为世人所习称之道，即今人所谓‘道理’也，第一‘道’字应从是解。《广雅·释诂》二：‘道，说也’，第二‘道’字应从是解。‘常’乃真常不易之义，在文法上为区别词。……第三‘道’字即二十五章‘道法自然’之‘道’，……乃老子学说之总名也”。陈鼓应说：“第一个‘道’字是人们习称之道，即今人所谓‘道理’。第二个‘道’字，是指言说的意思。第三个‘道’字，是老子哲学上的专有名词，在本章它意指构成宇宙的实体与动力。……‘常道’之‘常’，为真常、永恒之意。……可以用言词表达的道，就不是常道”。
-----------
2.认为“道可道”中的第一个“道”，指的是宇宙万物的本原；“可道”中的“道”，指言说的意思；“常道”，指恒久存在的“道”。因此，“道可道，非常道”，指可以言说的“道”，就不是恒久存在的“道”。如张默生说：“‘道’，指宇宙的本体而言。……‘常’，是经常不变的意思。……可以说出来的道，便不是经常不变的道”。董平说：“第一个‘道’字与‘可道’之‘道’，内涵并不相同。第一个‘道’字，是老子所揭示的作为宇宙本根之‘道’；‘可道’之‘道’，则是‘言说’的意思。……这里的大意就是说：凡一切可以言说之‘道’，都不是‘常道’或永恒之‘道’”。汤漳平等说：“第一句中的三个‘道’，第一、三均指形上之‘道’，中间的‘道’作动词，为可言之义。……道可知而可行，但非恒久不变之道”。
'''
)
```


```python
chat_result = chat_model(messages)
print(chat_result.content)
```

    第一篇文章认为，“道可道”中的第一个“道”指的是道理，第二个“道”指言说的意思，而“常道”指恒久存在的“道”。因此，“道可道，非常道”表明可以言说的道理并非恒久存在的道，而恒久存在的道则无法言说。作者引用了苏辙、蒋锡昌和陈鼓应的观点来支持这一解释。
    
    第二篇文章则认为，“道可道”中的第一个“道”指的是宇宙万物的本原，第二个“道”指言说的意思，而“常道”指恒久存在的“道”。作者解释道可以言说的道并非恒久存在的道，引用了张默生、董平和汤漳平等人的观点。
    
    在这两篇文章中，第二篇提出的论点更为清晰和一致。它明确指出第一个“道”是宇宙的本原，第二个“道”是言说的意思，第三个“道”是恒久存在的道。通过这种解释，文章的逻辑更加连贯，也更符合老子《道德经》中“道可道，非常道”的概念。因此，第二篇文章提出的论点更好，因为它更清晰地阐明了“道可道，非常道”的意义。


### 使用 FewShotPromptTemplate 类生成 Few-shot Prompt 

构造 few-shot prompt 的方法通常有两种：
- 从示例集（set of examples）中手动选择；
- 通过示例选择器（Example Selector）自动选择.


```python
from langchain.prompts.prompt import PromptTemplate


examples = [
  {
    "question": "谁活得更久，穆罕默德·阿里还是艾伦·图灵？",
    "answer": 
"""
这里需要进一步的问题吗：是的。
追问：穆罕默德·阿里去世时多大了？
中间答案：穆罕默德·阿里去世时74岁。
追问：艾伦·图灵去世时多大了？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里
"""
  },
  {
    "question": "craigslist的创始人是什么时候出生的？",
    "answer": 
"""
这里需要进一步的问题吗：是的。
追问：谁是craigslist的创始人？
中间答案：Craigslist是由Craig Newmark创办的。
追问：Craig Newmark是什么时候出生的？
中间答案：Craig Newmark出生于1952年12月6日。
所以最终答案是：1952年12月6日
"""
  },
  {
    "question": "乔治·华盛顿的外祖父是谁？",
    "answer":
"""
这里需要进一步的问题吗：是的。
追问：谁是乔治·华盛顿的母亲？
中间答案：乔治·华盛顿的母亲是Mary Ball Washington。
追问：Mary Ball Washington的父亲是谁？
中间答案：Mary Ball Washington的父亲是Joseph Ball。
所以最终答案是：Joseph Ball
"""
  },
  {
    "question": "《大白鲨》和《皇家赌场》的导演是同一个国家的吗？",
    "answer":
"""
这里需要进一步的问题吗：是的。
追问：谁是《大白鲨》的导演？
中间答案：《大白鲨》的导演是Steven Spielberg。
追问：Steven Spielberg来自哪里？
中间答案：美国。
追问：谁是《皇家赌场》的导演？
中间答案：《皇家赌场》的导演是Martin Campbell。
追问：Martin Campbell来自哪里？
中间答案：新西兰。
所以最终答案是：不是
"""
  }
]
```


```python
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}"
)

# **examples[0] 是将examples[0] 字典的键值对（question-answer）解包并传递给format，作为函数参数
print(example_prompt.format(**examples[0]))
```

    Question: 谁活得更久，穆罕默德·阿里还是艾伦·图灵？
    
    这里需要进一步的问题吗：是的。
    追问：穆罕默德·阿里去世时多大了？
    中间答案：穆罕默德·阿里去世时74岁。
    追问：艾伦·图灵去世时多大了？
    中间答案：艾伦·图灵去世时41岁。
    所以最终答案是：穆罕默德·阿里
    



```python
print(example_prompt)
```

    input_variables=['answer', 'question'] template='Question: {question}\n{answer}'



```python
print(example_prompt.format(**examples[-1]))
```

    Question: 《大白鲨》和《皇家赌场》的导演是同一个国家的吗？
    
    这里需要进一步的问题吗：是的。
    追问：谁是《大白鲨》的导演？
    中间答案：《大白鲨》的导演是Steven Spielberg。
    追问：Steven Spielberg来自哪里？
    中间答案：美国。
    追问：谁是《皇家赌场》的导演？
    中间答案：《皇家赌场》的导演是Martin Campbell。
    追问：Martin Campbell来自哪里？
    中间答案：新西兰。
    所以最终答案是：不是
    


#### 关于解包的示例


```python
def print_info(question, answer):
    print(f"Question: {question}")
    print(f"Answer: {answer}")

print_info(**examples[0]) 
```

    Question: 谁活得更久，穆罕默德·阿里还是艾伦·图灵？
    Answer: 
    这里需要进一步的问题吗：是的。
    追问：穆罕默德·阿里去世时多大了？
    中间答案：穆罕默德·阿里去世时74岁。
    追问：艾伦·图灵去世时多大了？
    中间答案：艾伦·图灵去世时41岁。
    所以最终答案是：穆罕默德·阿里
    


### 生成 Few-shot Prompt


```python
# 导入 FewShotPromptTemplate 类
from langchain.prompts.few_shot import FewShotPromptTemplate

# 创建一个 FewShotPromptTemplate 对象
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,           # 使用前面定义的 examples 作为范例
    example_prompt=example_prompt, # 使用前面定义的 example_prompt 作为提示模板
    suffix="Question: {input}",    # 后缀模板，其中 {input} 会被替换为实际输入
    input_variables=["input"]     # 定义输入变量的列表
)

# 使用给定的输入格式化 prompt，并打印结果
# 这里的 {input} 将被 "玛丽·波尔·华盛顿的父亲是谁?" 替换
print(few_shot_prompt.format(input="玛丽·波尔·华盛顿的父亲是谁?"))
```

    Question: 谁活得更久，穆罕默德·阿里还是艾伦·图灵？
    
    这里需要进一步的问题吗：是的。
    追问：穆罕默德·阿里去世时多大了？
    中间答案：穆罕默德·阿里去世时74岁。
    追问：艾伦·图灵去世时多大了？
    中间答案：艾伦·图灵去世时41岁。
    所以最终答案是：穆罕默德·阿里
    
    
    Question: craigslist的创始人是什么时候出生的？
    
    这里需要进一步的问题吗：是的。
    追问：谁是craigslist的创始人？
    中间答案：Craigslist是由Craig Newmark创办的。
    追问：Craig Newmark是什么时候出生的？
    中间答案：Craig Newmark出生于1952年12月6日。
    所以最终答案是：1952年12月6日
    
    
    Question: 乔治·华盛顿的外祖父是谁？
    
    这里需要进一步的问题吗：是的。
    追问：谁是乔治·华盛顿的母亲？
    中间答案：乔治·华盛顿的母亲是Mary Ball Washington。
    追问：Mary Ball Washington的父亲是谁？
    中间答案：Mary Ball Washington的父亲是Joseph Ball。
    所以最终答案是：Joseph Ball
    
    
    Question: 《大白鲨》和《皇家赌场》的导演是同一个国家的吗？
    
    这里需要进一步的问题吗：是的。
    追问：谁是《大白鲨》的导演？
    中间答案：《大白鲨》的导演是Steven Spielberg。
    追问：Steven Spielberg来自哪里？
    中间答案：美国。
    追问：谁是《皇家赌场》的导演？
    中间答案：《皇家赌场》的导演是Martin Campbell。
    追问：Martin Campbell来自哪里？
    中间答案：新西兰。
    所以最终答案是：不是
    
    
    Question: 玛丽·波尔·华盛顿的父亲是谁?


## 示例选择器 Example Selectors

**如果你有大量的参考示例，就得选择哪些要包含在提示中。最好还是根据某种条件或者规则来自动选择，Example Selector 是负责这个任务的类。**

BaseExampleSelector 定义如下：

```python
class BaseExampleSelector(ABC):
    """用于选择包含在提示中的示例的接口。"""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """根据输入选择要使用的示例。"""

```

`ABC` 是 Python 中的 `abc` 模块中的一个缩写，它表示 "Abstract Base Class"（抽象基类）。在 Python 中，抽象基类用于定义其他类必须遵循的基本接口或蓝图，但不能直接实例化。其主要目的是为了提供一种形式化的方式来定义和检查子类的接口。

使用抽象基类的几点关键信息：

1. **抽象方法**：在抽象基类中，你可以定义抽象方法，它没有实现（也就是说，它没有方法体）。任何继承该抽象基类的子类都必须提供这些抽象方法的实现。

2. **不能直接实例化**：你不能直接创建抽象基类的实例。试图这样做会引发错误。它们的主要目的是为了被继承，并在子类中实现其方法。

3. **强制子类实现**：如果子类没有实现所有的抽象方法，那么试图实例化该子类也会引发错误。这确保了继承抽象基类的所有子类都遵循了预定的接口。



```python

```


```python
# 导入需要的模块和类
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 定义一个提示模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],     # 输入变量的名字
    template="Input: {input}\nOutput: {output}",  # 实际的模板字符串
)

# 这是一个假设的任务示例列表，用于创建反义词
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
```

### Pandas 相关包首次导入错误后，再次执行即可正确导入


```python

# 从给定的示例中创建一个语义相似性选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                          # 可供选择的示例列表
    OpenAIEmbeddings(),                # 用于生成嵌入向量的嵌入类，用于衡量语义相似性
    Chroma,                            # 用于存储嵌入向量并进行相似性搜索的 VectorStore 类
    k=1                                # 要生成的示例数量
)

# 创建一个 FewShotPromptTemplate 对象
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 提供一个 ExampleSelector 替代示例
    example_prompt=example_prompt,      # 前面定义的提示模板
    prefix="Give the antonym of every input", # 前缀模板
    suffix="Input: {adjective}\nOutput:",     # 后缀模板
    input_variables=["adjective"],           # 输入变量的名字
)

```

    /home/ubuntu/miniconda3/envs/langchain/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The class `langchain_community.embeddings.openai.OpenAIEmbeddings` was deprecated in langchain-community 0.0.9 and will be removed in 0.2.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAIEmbeddings`.
      warn_deprecated(



```python
# 输入是一种感受，所以应该选择 happy/sad 的示例。
print(similar_prompt.format(adjective="worried"))
```

    Give the antonym of every input
    
    Input: happy
    Output: sad
    
    Input: worried
    Output:



```python
# 输入是一种度量，所以应该选择 tall/short的示例。
print(similar_prompt.format(adjective="long"))
```

    Give the antonym of every input
    
    Input: tall
    Output: short
    
    Input: long
    Output:



```python
print(similar_prompt.format(adjective="rain"))
```

    Give the antonym of every input
    
    Input: windy
    Output: calm
    
    Input: rain
    Output:

