# HF Transformers 核心模块学习：Pipelines 进阶

- 使用 Pipeline 如何与现代的大语言模型结合，以完成各类下游任务
- 使用 Tokenizer 编解码文本
- 使用 Models 加载和保存模型

## 使用 Pipeline 调用大语言模型

### Language Modeling

语言建模是一项预测文本序列中的单词的任务。它已经成为非常流行的自然语言处理任务，因为预训练的语言模型可以用于许多其他下游任务的微调。最近，对大型语言模型（LLMs）产生了很大兴趣，这些模型展示了零或少量样本学习能力。这意味着该模型可以解决其未经明确训练过的任务！虽然语言模型可用于生成流畅且令人信服的文本，但需要小心使用，因为文本可能并不总是准确无误。

通过理论篇学习，我们了解到有两种典型的语言模型：

- 自回归：模型目标是预测序列中的下一个 Token（文本），训练时对下文进行了掩码。如：GPT-3。
- 自编码：模型目标是理解上下文后，补全句子中丢失/掩码的 Token（文本）。如：BERT。

### 使用 GPT-2 实现文本生成


```python
from transformers import pipeline

prompt = "Hugging Face is a community-based open-source platform for machine learning."
generator = pipeline(task="text-generation", model="gpt2")
generator(prompt)
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.





    [{'generated_text': 'Hugging Face is a community-based open-source platform for machine learning. While it is an open-source framework, it is not designed for use on the production network yet. We have a project in our community based on this. We are'}]



在这个例子中，给定了一个提示（prompt）文本："Hugging Face is a community-based open-source platform for machine learning."，然后使用 GPT-2 模型来生成接下来的文本。

GPT-2 是一个生成式预训练模型，它可以接收一个输入文本，并根据这个输入文本生成接下来的文本。在这个例子中，模型将根据给定的提示文本生成一个或多个与该提示相关的文本片段。

#### 设置文本生成返回条数


```python
prompt = "You are very smart"
generator = pipeline(task="text-generation", model="gpt2", num_return_sequences=3)
```


```python
generator(prompt)
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.





    [{'generated_text': 'You are very smart. You will learn to play." When asked about a potential matchmaking solution on Summoner\'s Rift, Lee said of what he did in the past, "I just went against the wrong people."\n\nThis was, as Lee'},
     {'generated_text': 'You are very smart. You understand. Yes. You understand that.\n\n[Chalk]\n\nNow I do not see the point of asking you to take time off to get the book out then, you are asking myself to spend my'},
     {'generated_text': 'You are very smart."\n\nBenson said her own parents were not.\n\nShe said they were trying to learn from the first day of work, but got so angry they sent her a video showing their daughter being verbally abused by one.'}]




```python
generator(prompt, num_return_sequences=2)
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.





    [{'generated_text': 'You are very smart and I am not a child and not intelligent and that I could be making mistakes or exaggerating," he said as he stood in the corner of his room at home with three other friends.\n\n"If you don\'t understand'},
     {'generated_text': "You are very smart. She has always given her mind to things outside of my life. I don't mean to imply I'm a psychopath or that I've been mentally ill for the last decade or two, but she is very smart. I don"}]



#### 设置文本生成最大长度


```python
generator(prompt, num_return_sequences=2, max_length=16)
```

    Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.





    [{'generated_text': 'You are very smart and highly skilled in your job. Let the boss know she'},
     {'generated_text': 'You are very smart of you," he says, as you turn to the second'}]



### 使用 BERT-Base-Chinese 实现中文补全



```python
from transformers import pipeline

fill_mask = pipeline(task="fill-mask", model="bert-base-chinese")
```

    Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
text = "人民是[MASK]可战胜的"

fill_mask(text, top_k=1)
```




    [{'score': 0.9203746318817139,
      'token': 679,
      'token_str': '不',
      'sequence': '人 民 是 不 可 战 胜 的'}]



#### 设置文本补全的条数


```python
text = "美国的首都是[MASK]"

fill_mask(text, top_k=1)
```




    [{'score': 0.7596935629844666,
      'token': 8043,
      'token_str': '？',
      'sequence': '美 国 的 首 都 是 ？'}]




这个结果是填充式模型的预测结果，其中包含了以下信息：

score: 表示模型预测该词的置信度得分，得分为 0.7597。
token: 表示预测的词在词汇表中的索引。
token_str: 表示预测的词的字符串形式，这里是一个中文问号 "？"。
sequence: 表示填充后的完整文本序列，即模型填充了 "[MASK]" 标记后得到的完整句子，这里是 "美国的首都是？"。
因此，


```python
text = "巴黎是[MASK]国的首都。"
fill_mask(text, top_k=1)
```




    [{'score': 0.9911921620368958,
      'token': 3791,
      'token_str': '法',
      'sequence': '巴 黎 是 法 国 的 首 都 。'}]




```python
text = "美国的首都是[MASK]"
fill_mask(text, top_k=3)
```




    [{'score': 0.7596935629844666,
      'token': 8043,
      'token_str': '？',
      'sequence': '美 国 的 首 都 是 ？'},
     {'score': 0.21126732230186462,
      'token': 511,
      'token_str': '。',
      'sequence': '美 国 的 首 都 是 。'},
     {'score': 0.02683420106768608,
      'token': 8013,
      'token_str': '！',
      'sequence': '美 国 的 首 都 是 ！'}]




```python
text = "美国的首都是[MASK][MASK][MASK]"

fill_mask(text, top_k=1)
```




    [[{'score': 0.5740304589271545,
       'token': 5294,
       'token_str': '纽',
       'sequence': '[CLS] 美 国 的 首 都 是 纽 [MASK] [MASK] [SEP]'}],
     [{'score': 0.4926770329475403,
       'token': 5276,
       'token_str': '约',
       'sequence': '[CLS] 美 国 的 首 都 是 [MASK] 约 [MASK] [SEP]'}],
     [{'score': 0.9353275895118713,
       'token': 511,
       'token_str': '。',
       'sequence': '[CLS] 美 国 的 首 都 是 [MASK] [MASK] 。 [SEP]'}]]



在 BERT 模型中，[SEP] 标记表示分隔符（separator）。在输入文本中，它通常用于分隔不同的句子或文本段落。

## 使用 AutoClass 高效管理 `Tokenizer` 和 `Model`

通常，您想要使用的模型（网络架构）可以从您提供给 `from_pretrained()` 方法的预训练模型的名称或路径中推测出来。

AutoClasses就是为了帮助用户完成这个工作，以便根据`预训练权重/配置文件/词汇表的名称/路径自动检索相关模型`。

比如手动加载`bert-base-chinese`模型以及对应的 `tokenizer` 方法如下：

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
```

以下是我们实际操作和演示：

### 使用 `from_pretrained` 方法加载指定 Model 和 Tokenizer


```python
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

#### 使用 BERT Tokenizer 编码文本

编码 (Encoding) 过程包含两个步骤：

- 分词：使用分词器按某种策略将文本切分为 tokens；
- 映射：将 tokens 转化为对应的 token IDs。


```python
# 第一步：分词
sequence = "美国的首都是华盛顿特区"
tokens = tokenizer.tokenize(sequence)
print(tokens)
```

    ['美', '国', '的', '首', '都', '是', '华', '盛', '顿', '特', '区']



```python
# 第二步：映射
token_ids = tokenizer.convert_tokens_to_ids(tokens)
```


```python
print(token_ids)
```

    [5401, 1744, 4638, 7674, 6963, 3221, 1290, 4670, 7561, 4294, 1277]


#### 使用 Tokenizer.encode 方法端到端处理


```python
token_ids_e2e = tokenizer.encode(sequence) #具体来说，tokenizer.encode(sequence) 的作用是将输入的文本序列经过分词器处理后，转换为模型所需的输入表示形式。
```


```python
token_ids_e2e
```




    [101, 5401, 1744, 4638, 7674, 6963, 3221, 1290, 4670, 7561, 4294, 1277, 102]




```python
tokenizer.decode(token_ids)
```




    '美 国 的 首 都 是 华 盛 顿 特 区'



#### 编解码多段文本


```python
sequence_batch = ["美国的首都是华盛顿特区", "中国的首都是北京"]
```


```python
token_ids_batch = tokenizer.encode(sequence_batch)
```


```python
tokenizer.decode(token_ids_batch)
```




    '[CLS] 美 国 的 首 都 是 华 盛 顿 特 区 [SEP] 中 国 的 首 都 是 北 京 [SEP]'




```python
embedding_batch = tokenizer("美国的首都是华盛顿特区", "中国的首都是北京")
print(embedding_batch)
```

    {'input_ids': [101, 5401, 1744, 4638, 7674, 6963, 3221, 1290, 4670, 7561, 4294, 1277, 102, 704, 1744, 4638, 7674, 6963, 3221, 1266, 776, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



```python
# 优化下输出结构
for key, value in embedding_batch.items():
    print(f"{key}: {value}\n")
```

    input_ids: [101, 5401, 1744, 4638, 7674, 6963, 3221, 1290, 4670, 7561, 4294, 1277, 102, 704, 1744, 4638, 7674, 6963, 3221, 1266, 776, 102]
    
    token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    


### 添加新 Token

当出现了词表或嵌入空间中不存在的新Token，需要使用 Tokenizer 将其添加到词表中。 Transformers 库提供了两种不同方法：

- add_tokens: 添加常规的正文文本 Token，以追加（append）的方式添加到词表末尾。
- add_special_tokens: 添加特殊用途的 Token，优先在已有特殊词表中选择（`bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token`）。如果预定义均不满足，则都添加到`additional_special_tokens`。

#### 添加常规 Token

先查看已有词表，确保新添加的 Token 不在词表中：


```python
len(tokenizer.vocab.keys())
```




    21128




```python
from itertools import islice

# 使用 islice 查看词表部分内容
for key, value in islice(tokenizer.vocab.items(), 10):
    print(f"{key}: {value}")
```

    10℃: 9115
    ##鰲: 20872
    喺: 1615
    钥: 7170
    life: 8562
    営: 1612
    芎: 5694
    ##へて: 12864
    ヌ: 616
    ##ᆯ: 11596



```python
new_tokens = ["天干", "地支"]
```


```python
# 将集合作差结果添加到词表中
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
```


```python
new_tokens
```




    {'地支', '天干'}




```python
tokenizer.add_tokens(list(new_tokens))
```




    2




```python
# 新增加了2个Token，词表总数由 21128 增加到 21130
len(tokenizer.vocab.keys())
```




    21130




```python
new_special_token = {"sep_token": "NEW_SPECIAL_TOKEN"}
```


```python
tokenizer.add_special_tokens(new_special_token)
```




    1




```python
# 新增加了1个特殊Token，词表总数由 21128 增加到 21131
len(tokenizer.vocab.keys())
```




    21131



### 使用 `save_pretrained` 方法保存指定 Model 和 Tokenizer

借助 `AutoClass` 的设计理念，保存 Model 和 Tokenizer 的方法也相当高效便捷。

假设我们对`bert-base-chinese`模型以及对应的 `tokenizer` 做了修改，并更名为`new-bert-base-chinese`，方法如下：

```python
tokenizer.save_pretrained("./models/new-bert-base-chinese")
model.save_pretrained("./models/new-bert-base-chinese")
```

保存 Tokenizer 会在指定路径下创建以下文件：
- tokenizer.json: Tokenizer 元数据文件；
- special_tokens_map.json: 特殊字符映射关系配置文件；
- tokenizer_config.json: Tokenizer 基础配置文件，存储构建 Tokenizer 需要的参数；
- vocab.txt: 词表文件；
- added_tokens.json: 单独存放新增 Tokens 的配置文件。

保存 Model 会在指定路径下创建以下文件：
- config.json：模型配置文件，存储模型结构参数，例如 Transformer 层数、特征空间维度等；
- pytorch_model.bin：又称为 state dictionary，存储模型的权重。


```python
tokenizer.save_pretrained("./models/new-bert-base-chinese")
```




    ('./models/new-bert-base-chinese/tokenizer_config.json',
     './models/new-bert-base-chinese/special_tokens_map.json',
     './models/new-bert-base-chinese/vocab.txt',
     './models/new-bert-base-chinese/added_tokens.json',
     './models/new-bert-base-chinese/tokenizer.json')




```python
model.save_pretrained("./models/new-bert-base-chinese")
```


