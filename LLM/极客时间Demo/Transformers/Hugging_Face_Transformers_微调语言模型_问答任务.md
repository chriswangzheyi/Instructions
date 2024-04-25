# Hugging Face Transformers 微调语言模型-问答任务

我们已经学会使用 Pipeline 加载支持问答任务的预训练模型，本教程代码将展示如何微调训练一个支持问答任务的模型。

**注意：微调后的模型仍然是通过提取上下文的子串来回答问题的，而不是生成新的文本。**

### 模型执行问答效果示例

![Widget inference representing the QA task](docs/images/question_answering.png)

---




```python
# 根据你使用的模型和GPU资源情况，调整以下关键参数
squad_v2 = False
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```


```python
pip install datasets
```

    Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (2.19.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets) (3.13.4)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.25.2)
    Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (14.0.2)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets) (0.6)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.0.3)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.31.0)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.2)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.4.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.16)
    Requirement already satisfied: fsspec[http]<=2024.3.1,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.9.5)
    Requirement already satisfied: huggingface-hub>=0.21.2 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.22.2)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.21.2->datasets) (4.11.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2024.2.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)


## 下载数据集

在本教程中，我们将使用[斯坦福问答数据集(SQuAD）](https://rajpurkar.github.io/SQuAD-explorer/)。

### SQuAD 数据集

**斯坦福问答数据集(SQuAD)** 是一个阅读理解数据集，由众包工作者在一系列维基百科文章上提出问题组成。每个问题的答案都是相应阅读段落中的文本片段或范围，或者该问题可能无法回答。

SQuAD2.0将SQuAD1.1中的10万个问题与由众包工作者对抗性地撰写的5万多个无法回答的问题相结合，使其看起来与可回答的问题类似。要在SQuAD2.0上表现良好，系统不仅必须在可能时回答问题，还必须确定段落中没有支持任何答案，并放弃回答。


```python
from datasets import load_dataset
```


```python
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



```python
datasets
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 87599
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 10570
        })
    })



#### 对比数据集

相比快速入门使用的 Yelp 评论数据集，我们可以看到 SQuAD 训练和测试集都新增了用于上下文、问题以及问题答案的列：

**YelpReviewFull Dataset：**

```json

DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 650000
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 50000
    })
})
```


```python
datasets["train"][0]
```




    {'id': '5733be284776f41900661182',
     'title': 'University_of_Notre_Dame',
     'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
     'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
     'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}





#### 从上下文中组织回复内容

我们可以看到答案是通过它们在文本中的起始位置（这里是第515个字符）以及它们的完整文本表示的，这是上面提到的上下文的子字符串。


```python
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
```


```python
show_random_elements(datasets["train"])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>context</th>
      <th>question</th>
      <th>answers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56d12fe417492d1400aabbad</td>
      <td>IPod</td>
      <td>iPods have also gained popularity for use in education. Apple offers more information on educational uses for iPods on their website, including a collection of lesson plans. There has also been academic research done in this area in nursing education and more general K-16 education. Duke University provided iPods to all incoming freshmen in the fall of 2004, and the iPod program continues today with modifications. Entertainment Weekly put it on its end-of-the-decade, "best-of" list, saying, "Yes, children, there really was a time when we roamed the earth without thousands of our favorite jams tucked comfortably into our hip pockets. Weird."</td>
      <td>Which magazine placed the iPod on its Best of the Decade list for the 00's?</td>
      <td>{'text': ['Entertainment Weekly'], 'answer_start': [418]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56d6311c1c85041400946fd0</td>
      <td>Dog</td>
      <td>The most popular Korean dog dish is gaejang-guk (also called bosintang), a spicy stew meant to balance the body's heat during the summer months; followers of the custom claim this is done to ensure good health by balancing one's gi, or vital energy of the body. A 19th century version of gaejang-guk explains that the dish is prepared by boiling dog meat with scallions and chili powder. Variations of the dish contain chicken and bamboo shoots. While the dishes are still popular in Korea with a segment of the population, dog is not as widely consumed as beef, chicken, and pork.</td>
      <td>Why do people eat Gaejang-guk in the summer months?</td>
      <td>{'text': ['to balance the body's heat'], 'answer_start': [92]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57282bd74b864d1900164652</td>
      <td>St._John%27s,_Newfoundland_and_Labrador</td>
      <td>Metrobus Transit is responsible for public transit in the region. Metrobus has a total of 19 routes, 53 buses and an annual ridership of 3,014,073. Destinations include the Avalon Mall, The Village Shopping Centre, Memorial University, Academy Canada, the College of the North Atlantic, the Marine Institute, the Confederation Building, downtown, Stavanger Drive Business Park, Kelsey Drive, Goulds, Kilbride, Shea Heights, the four hospitals in the city as well as other important areas in St. John's and Mount Pearl.</td>
      <td>How many hospitals does the city have?</td>
      <td>{'text': ['four'], 'answer_start': [428]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5726b79bdd62a815002e8de8</td>
      <td>Raleigh,_North_Carolina</td>
      <td>The Time Warner Cable Music Pavilion at Walnut Creek hosts major international touring acts. In 2011, the Downtown Raleigh Amphitheater opened (now sponsored as the Red Hat Amphitheater), which hosts numerous concerts primarily in the summer months. An additional amphitheater sits on the grounds of the North Carolina Museum of Art, which hosts a summer concert series and outdoor movies. Nearby Cary is home to the Koka Booth Amphitheatre which hosts additional summer concerts and outdoor movies, and serves as the venue for regularly scheduled outdoor concerts by the North Carolina Symphony based in Raleigh. During the North Carolina State Fair, Dorton Arena hosts headline acts. The private Lincoln Theatre is one of several clubs in downtown Raleigh that schedules many concerts throughout the year in multiple formats (rock, pop, country).</td>
      <td>Where are major touring acts hosted in the city?</td>
      <td>{'text': ['The Time Warner Cable Music Pavilion'], 'answer_start': [0]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56e7adfc37bdd419002c4324</td>
      <td>Nanjing</td>
      <td>In the 1960s, the first Nanjing Yangtze River Bridge was completed, and served as the only bridge crossing over the Lower Yangtze in eastern China at that time. The bridge was a source of pride and an important symbol of modern China, having been built and designed by the Chinese themselves following failed surveys by other nations and the reliance on and then rejection of Soviet expertise. Begun in 1960 and opened to traffic in 1968, the bridge is a two-tiered road and rail design spanning 4,600 metres on the upper deck, with approximately 1,580 metres spanning the river itself. Since then four more bridges and two tunnels have been built. Going in the downstream direction, the Yangtze crossings in Nanjing are: Dashengguan Bridge, Line 10 Metro Tunnel, Third Bridge, Nanjing Yangtze River Tunnel, First Bridge, Second Bridge and Fourth Bridge.</td>
      <td>When did the Nanjing Yangtze River Bridge first open for traffic?</td>
      <td>{'text': ['1968'], 'answer_start': [433]}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5727b42d4b864d1900163acd</td>
      <td>New_Haven,_Connecticut</td>
      <td>The city hosts numerous theatres and production houses, including the Yale Repertory Theatre, the Long Wharf Theatre, and the Shubert Theatre. There is also theatre activity from the Yale School of Drama, which works through the Yale University Theatre and the student-run Yale Cabaret. Southern Connecticut State University hosts the Lyman Center for the Performing Arts. The shuttered Palace Theatre (opposite the Shubert Theatre) is being renovated and will reopen as the College Street Music Hall in May, 2015. Smaller theatres include the Little Theater on Lincoln Street. Cooperative Arts and Humanities High School also boasts a state-of-the-art theatre on College Street. The theatre is used for student productions as well as the home to weekly services to a local non-denominational church, the City Church New Haven.</td>
      <td>What was the original name of the College Street Music Hall?</td>
      <td>{'text': ['Palace Theatre'], 'answer_start': [387]}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5726e6bcdd62a815002e947a</td>
      <td>Yale_University</td>
      <td>Yale's Office of Sustainability develops and implements sustainability practices at Yale. Yale is committed to reduce its greenhouse gas emissions 10% below 1990 levels by the year 2020. As part of this commitment, the university allocates renewable energy credits to offset some of the energy used by residential colleges. Eleven campus buildings are candidates for LEED design and certification. Yale Sustainable Food Project initiated the introduction of local, organic vegetables, fruits, and beef to all residential college dining halls. Yale was listed as a Campus Sustainability Leader on the Sustainable Endowments Institute’s College Sustainability Report Card 2008, and received a "B+" grade overall.</td>
      <td>How many campus buildings are candidates for LEED design and certification?</td>
      <td>{'text': ['Eleven'], 'answer_start': [324]}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5727fb1b3acd2414000df165</td>
      <td>USB</td>
      <td>In July 2012, the USB Promoters Group announced the finalization of the USB Power Delivery ("PD") specification, an extension that specifies using certified "PD aware" USB cables with standard USB Type-A and Type-B connectors to deliver increased power (more than 7.5 W) to devices with larger power demand. Devices can request higher currents and supply voltages from compliant hosts –  up to 2 A at 5 V (for a power consumption of up to 10 W), and optionally up to 3 A or 5 A at either 12 V (36 W or 60 W) or 20 V (60 W or 100 W). In all cases, both host-to-device and device-to-host configurations are supported.</td>
      <td>When did the USB Promoters Group announce the finalization of the USB Power Delivery specification?</td>
      <td>{'text': ['In July 2012'], 'answer_start': [0]}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>57320e1ee17f3d1400422636</td>
      <td>Protestantism</td>
      <td>The Plymouth Brethren are a conservative, low church, evangelical movement, whose history can be traced to Dublin, Ireland, in the late 1820s, originating from Anglicanism. Among other beliefs, the group emphasizes sola scriptura. Brethren generally see themselves not as a denomination, but as a network, or even as a collection of overlapping networks, of like-minded independent churches. Although the group refused for many years to take any denominational name to itself—a stance that some of them still maintain—the title The Brethren, is one that many of their number are comfortable with in that the Bible designates all believers as brethren.</td>
      <td>When did the Plymouth Brethren originate?</td>
      <td>{'text': ['late 1820s'], 'answer_start': [131]}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5726bf09f1498d1400e8e9fa</td>
      <td>Alloy</td>
      <td>In 1906, precipitation hardening alloys were discovered by Alfred Wilm. Precipitation hardening alloys, such as certain alloys of aluminium, titanium, and copper, are heat-treatable alloys that soften when quenched (cooled quickly), and then harden over time. After quenching a ternary alloy of aluminium, copper, and magnesium, Wilm discovered that the alloy increased in hardness when left to age at room temperature. Although an explanation for the phenomenon was not provided until 1919, duralumin was one of the first "age hardening" alloys to be used, and was soon followed by many others. Because they often exhibit a combination of high strength and low weight, these alloys became widely used in many forms of industry, including the construction of modern aircraft.</td>
      <td>Who discovered precipitation hardening alloys?</td>
      <td>{'text': ['Alfred Wilm'], 'answer_start': [59]}</td>
    </tr>
  </tbody>
</table>


## 预处理数据


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

以下断言确保我们的 Tokenizers 使用的是 FastTokenizer（Rust 实现，速度和功能性上有一定优势）。


```python
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
```

您可以在大模型表上查看哪种类型的模型具有可用的快速标记器，哪种类型没有。

您可以直接在两个句子上调用此标记器（一个用于答案，一个用于上下文）：


```python
tokenizer("What is your name?", "My name is Sylvain.")
```




    {'input_ids': [101, 2054, 2003, 2115, 2171, 1029, 102, 2026, 2171, 2003, 25353, 22144, 2378, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



### Tokenizer 进阶操作

在问答预处理中的一个特定问题是如何处理非常长的文档。

在其他任务中，当文档的长度超过模型最大句子长度时，我们通常会截断它们，但在这里，删除上下文的一部分可能会导致我们丢失正在寻找的答案。

为了解决这个问题，我们允许数据集中的一个（长）示例生成多个输入特征，每个特征的长度都小于模型的最大长度（或我们设置的超参数）。


```python
# The maximum length of a feature (question and context)
max_length = 384
# The authorized overlap between two part of the context when splitting it is needed.
doc_stride = 128
```

#### 超出最大长度的文本数据处理

下面，我们从训练集中找出一个超过最大长度（384）的文本：


```python
for i, example in enumerate(datasets["train"]):
    if len(tokenizer(example["question"], example["context"])["input_ids"]) > 384:
        break
# 挑选出来超过384（最大长度）的数据样例
example = datasets["train"][i]
```


```python
len(tokenizer(example["question"], example["context"])["input_ids"])
```




    396




```python
#### 截断上下文不保留超出部分
```


```python
len(tokenizer(example["question"],
              example["context"],
              max_length=max_length,
              truncation="only_second")["input_ids"])
```




    384



#### 关于截断的策略

- 直接截断超出部分: truncation=`only_second`
- 仅截断上下文（context），保留问题（question）：`return_overflowing_tokens=True` & 设置`stride`


```python
tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    stride=doc_stride
)
```

使用此策略截断后，Tokenizer 将返回多个 `input_ids` 列表。


```python
[len(x) for x in tokenized_example["input_ids"]]
```




    [384, 157]



解码两个输入特征，可以看到重叠的部分：


```python
for x in tokenized_example["input_ids"][:2]:
    print(tokenizer.decode(x))
```

    [CLS] how many wins does the notre dame men's basketball team have? [SEP] the men's basketball team has over 1, 600 wins, one of only 12 schools who have reached that mark, and have appeared in 28 ncaa tournaments. former player austin carr holds the record for most points scored in a single game of the tournament with 61. although the team has never won the ncaa tournament, they were named by the helms athletic foundation as national champions twice. the team has orchestrated a number of upsets of number one ranked teams, the most notable of which was ending ucla's record 88 - game winning streak in 1974. the team has beaten an additional eight number - one teams, and those nine wins rank second, to ucla's 10, all - time in wins against the top team. the team plays in newly renovated purcell pavilion ( within the edmund p. joyce center ), which reopened for the beginning of the 2009 – 2010 season. the team is coached by mike brey, who, as of the 2014 – 15 season, his fifteenth at notre dame, has achieved a 332 - 165 record. in 2009 they were invited to the nit, where they advanced to the semifinals but were beaten by penn state who went on and beat baylor in the championship. the 2010 – 11 team concluded its regular season ranked number seven in the country, with a record of 25 – 5, brey's fifth straight 20 - win season, and a second - place finish in the big east. during the 2014 - 15 season, the team went 32 - 6 and won the acc conference tournament, later advancing to the elite 8, where the fighting irish lost on a missed buzzer - beater against then undefeated kentucky. led by nba draft picks jerian grant and pat connaughton, the fighting irish beat the eventual national champion duke blue devils twice during the season. the 32 wins were [SEP]
    [CLS] how many wins does the notre dame men's basketball team have? [SEP] championship. the 2010 – 11 team concluded its regular season ranked number seven in the country, with a record of 25 – 5, brey's fifth straight 20 - win season, and a second - place finish in the big east. during the 2014 - 15 season, the team went 32 - 6 and won the acc conference tournament, later advancing to the elite 8, where the fighting irish lost on a missed buzzer - beater against then undefeated kentucky. led by nba draft picks jerian grant and pat connaughton, the fighting irish beat the eventual national champion duke blue devils twice during the season. the 32 wins were the most by the fighting irish team since 1908 - 09. [SEP]


#### 使用 offsets_mapping 获取原始的 input_ids

设置 `return_offsets_mapping=True`，将使得截断分割生成的多个 input_ids 列表中的 token，通过映射保留原始文本的 input_ids。

如下所示：第一个标记（[CLS]）的起始和结束字符都是（0, 0），因为它不对应问题/答案的任何部分，然后第二个标记与问题(question)的字符0到3相同.


```python
tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_length,
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    stride=doc_stride
)
print(tokenized_example["offset_mapping"][0][:100])
```

    [(0, 0), (0, 3), (4, 8), (9, 13), (14, 18), (19, 22), (23, 28), (29, 33), (34, 37), (37, 38), (38, 39), (40, 50), (51, 55), (56, 60), (60, 61), (0, 0), (0, 3), (4, 7), (7, 8), (8, 9), (10, 20), (21, 25), (26, 29), (30, 34), (35, 36), (36, 37), (37, 40), (41, 45), (45, 46), (47, 50), (51, 53), (54, 58), (59, 61), (62, 69), (70, 73), (74, 78), (79, 86), (87, 91), (92, 96), (96, 97), (98, 101), (102, 106), (107, 115), (116, 118), (119, 121), (122, 126), (127, 138), (138, 139), (140, 146), (147, 153), (154, 160), (161, 165), (166, 171), (172, 175), (176, 182), (183, 186), (187, 191), (192, 198), (199, 205), (206, 208), (209, 210), (211, 217), (218, 222), (223, 225), (226, 229), (230, 240), (241, 245), (246, 248), (248, 249), (250, 258), (259, 262), (263, 267), (268, 271), (272, 277), (278, 281), (282, 285), (286, 290), (291, 301), (301, 302), (303, 307), (308, 312), (313, 318), (319, 321), (322, 325), (326, 330), (330, 331), (332, 340), (341, 351), (352, 354), (355, 363), (364, 373), (374, 379), (379, 380), (381, 384), (385, 389), (390, 393), (394, 406), (407, 408), (409, 415), (416, 418)]


因此，我们可以使用这个映射来找到答案在给定特征中的起始和结束标记的位置。

我们只需区分偏移的哪些部分对应于问题，哪些部分对应于上下文。


```python
first_token_id = tokenized_example["input_ids"][0][1]
offsets = tokenized_example["offset_mapping"][0][1]
print(tokenizer.convert_ids_to_tokens([first_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    how How



```python
second_token_id = tokenized_example["input_ids"][0][2]
offsets = tokenized_example["offset_mapping"][0][2]
print(tokenizer.convert_ids_to_tokens([second_token_id])[0], example["question"][offsets[0]:offsets[1]])
```

    many many



```python
example["question"]
```




    "How many wins does the Notre Dame men's basketball team have?"



借助`tokenized_example`的`sequence_ids`方法，我们可以方便的区分token的来源编号：

- 对于特殊标记：返回None，
- 对于正文Token：返回句子编号（从0开始编号）。

综上，现在我们可以很方便的在一个输入特征中找到答案的起始和结束 Token。


```python
sequence_ids = tokenized_example.sequence_ids()
print(sequence_ids)
```

    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]



```python
answers = example["answers"]
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

# 当前span在文本中的起始标记索引。
token_start_index = 0
while sequence_ids[token_start_index] != 1:
    token_start_index += 1

# 当前span在文本中的结束标记索引。
token_end_index = len(tokenized_example["input_ids"][0]) - 1
while sequence_ids[token_end_index] != 1:
    token_end_index -= 1

# 检测答案是否超出span范围（如果超出范围，该特征将以CLS标记索引标记）。
offsets = tokenized_example["offset_mapping"][0]
if (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
    # 将token_start_index和token_end_index移动到答案的两端。
    # 注意：如果答案是最后一个单词，我们可以移到最后一个标记之后（边界情况）。
    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
    start_position = token_start_index - 1
    while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1
    end_position = token_end_index + 1
    print(start_position, end_position)
else:
    print("答案不在此特征中。")

```

    23 26


打印检查是否准确找到了起始位置：


```python
# 通过查找 offset mapping 位置，解码 context 中的答案
print(tokenizer.decode(tokenized_example["input_ids"][0][start_position: end_position+1]))
# 直接打印 数据集中的标准答案（answer["text"])
print(answers["text"][0])
```

    over 1, 600
    over 1,600


#### 关于填充的策略

- 对于没有超过最大长度的文本，填充补齐长度。
- 对于需要左侧填充的模型，交换 question 和 context 顺序


```python
pad_on_right = tokenizer.padding_side == "right"
```

### 整合以上所有预处理步骤

让我们将所有内容整合到一个函数中，并将其应用到训练集。

针对不可回答的情况（上下文过长，答案在另一个特征中），我们为开始和结束位置都设置了cls索引。

如果allow_impossible_answers标志为False，我们还可以简单地从训练集中丢弃这些示例。


```python
def prepare_train_features(examples):
    # 一些问题的左侧可能有很多空白字符，这对我们没有用，而且会导致上下文的截断失败
    # （标记化的问题将占用大量空间）。因此，我们删除左侧的空白字符。
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # 使用截断和填充对我们的示例进行标记化，但保留溢出部分，使用步幅（stride）。
    # 当上下文很长时，这会导致一个示例可能提供多个特征，其中每个特征的上下文都与前一个特征的上下文有一些重叠。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于一个示例可能给我们提供多个特征（如果它具有很长的上下文），我们需要一个从特征到其对应示例的映射。这个键就提供了这个映射关系。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # 偏移映射将为我们提供从令牌到原始上下文中的字符位置的映射。这将帮助我们计算开始位置和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 让我们为这些示例进行标记！
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 我们将使用 CLS 特殊 token 的索引来标记不可能的答案。
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 获取与该示例对应的序列（以了解上下文和问题是什么）。
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 一个示例可以提供多个跨度，这是包含此文本跨度的示例的索引。
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有给出答案，则将cls_index设置为答案。
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案在文本中的开始和结束字符索引。
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 当前跨度在文本中的开始令牌索引。
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # 当前跨度在文本中的结束令牌索引。
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 检测答案是否超出跨度（在这种情况下，该特征的标签将使用CLS索引）。
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 否则，将token_start_index和token_end_index移到答案的两端。
                # 注意：如果答案是最后一个单词（边缘情况），我们可以在最后一个偏移之后继续。
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples
```

#### datasets.map 的进阶使用

使用 `datasets.map` 方法将 `prepare_train_features` 应用于所有训练、验证和测试数据：

- batched: 批量处理数据。
- remove_columns: 因为预处理更改了样本的数量，所以在应用它时需要删除旧列。
- load_from_cache_file：是否使用datasets库的自动缓存

datasets 库针对大规模数据，实现了高效缓存机制，能够自动检测传递给 map 的函数是否已更改（因此需要不使用缓存数据）。如果在调用 map 时设置 `load_from_cache_file=False`，可以强制重新应用预处理。


```python
tokenized_datasets = datasets.map(prepare_train_features,
                                  batched=True,
                                  remove_columns=datasets["train"].column_names)
```


    Map:   0%|          | 0/87599 [00:00<?, ? examples/s]



    Map:   0%|          | 0/10570 [00:00<?, ? examples/s]


## 微调模型

现在我们的数据已经准备好用于训练，我们可以下载预训练模型并进行微调。

由于我们的任务是问答，我们使用 `AutoModelForQuestionAnswering` 类。(对比 Yelp 评论打分使用的是 `AutoModelForSequenceClassification` 类）

警告通知我们正在丢弃一些权重（`vocab_transform` 和 `vocab_layer_norm` 层），并随机初始化其他一些权重（`pre_classifier` 和 `classifier` 层）。在微调模型情况下是绝对正常的，因为我们正在删除用于预训练模型的掩码语言建模任务的头部，并用一个新的头部替换它，对于这个新头部，我们没有预训练的权重，所以库会警告我们在用它进行推理之前应该对这个模型进行微调，而这正是我们要做的事情。


```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForQuestionAnswering: ['vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.bias', 'vocab_projector.bias']
    - This IS expected if you are initializing DistilBertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


#### 训练超参数（TrainingArguments）


```python
pip install transformers[torch]
```

    Requirement already satisfied: transformers[torch] in /usr/local/lib/python3.10/dist-packages (4.24.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (3.13.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.10.0 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (0.22.2)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (2.31.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (0.13.3)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (4.66.2)
    Requirement already satisfied: torch!=1.12.0,>=1.7 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]) (2.2.1+cu121)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.10.0->transformers[torch]) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.10.0->transformers[torch]) (4.11.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (3.1.3)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (12.1.105)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.19.3 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (2.19.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (12.1.105)
    Requirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch!=1.12.0,>=1.7->transformers[torch]) (2.2.0)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch!=1.12.0,>=1.7->transformers[torch]) (12.4.127)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers[torch]) (2024.2.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch!=1.12.0,>=1.7->transformers[torch]) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch!=1.12.0,>=1.7->transformers[torch]) (1.3.0)



```python
pip install transformers==4.24.0
```

    Requirement already satisfied: transformers==4.24.0 in /usr/local/lib/python3.10/dist-packages (4.24.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (3.13.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.10.0 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (0.22.2)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (2.31.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (0.13.3)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers==4.24.0) (4.66.2)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.10.0->transformers==4.24.0) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.10.0->transformers==4.24.0) (4.11.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.24.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.24.0) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.24.0) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.24.0) (2024.2.2)



```python
pip install accelerate -U
```

    Requirement already satisfied: accelerate in /usr/local/lib/python3.10/dist-packages (0.29.3)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from accelerate) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (24.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from accelerate) (6.0.1)
    Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (2.2.1+cu121)
    Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.10/dist-packages (from accelerate) (0.22.2)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from accelerate) (0.4.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.13.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (4.11.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2023.6.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.19.3 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.19.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.2.0)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.10.0->accelerate) (12.4.127)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (4.66.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (2024.2.2)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)



```python
batch_size=64
model_dir = f"models/{model_checkpoint}-finetuned-squad"

args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)
```

#### Data Collator（数据整理器）

数据整理器将训练数据整理为批次数据，用于模型训练时的批次处理。本教程使用默认的 `default_data_collator`。



```python
from transformers import default_data_collator

data_collator = default_data_collator
```

### 实例化训练器（Trainer）

为了减少训练时间（需要大量算力支持），我们不在本教程的训练模型过程中计算模型评估指标。

而是训练完成后，再独立进行模型评估。


```python
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```


```python
trainer.train()
```

    /usr/local/lib/python3.10/dist-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(
    ***** Running training *****
      Num examples = 88524
      Num Epochs = 3
      Instantaneous batch size per device = 64
      Total train batch size (w. parallel, distributed & accumulation) = 64
      Gradient Accumulation steps = 1
      Total optimization steps = 4152
      Number of trainable parameters = 66364418




    <div>

      <progress value='4153' max='4152' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [4152/4152 1:19:44, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.499000</td>
      <td>1.265836</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.120600</td>
      <td>1.161048</td>
    </tr>
  </tbody>
</table><p>
    <div>

      <progress value='126' max='169' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [126/169 00:50 < 00:17, 2.47 it/s]
    </div>



    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-500
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-500/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-500/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-500/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-500/special_tokens_map.json
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-1000
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1000/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1000/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1000/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1000/special_tokens_map.json
    ***** Running Evaluation *****
      Num examples = 10784
      Batch size = 64
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-1500
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1500/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1500/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1500/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-1500/special_tokens_map.json
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-2000
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2000/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2000/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2000/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2000/special_tokens_map.json
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-2500
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2500/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2500/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2500/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-2500/special_tokens_map.json
    ***** Running Evaluation *****
      Num examples = 10784
      Batch size = 64
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-3000
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3000/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3000/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3000/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3000/special_tokens_map.json
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-3500
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3500/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3500/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3500/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-3500/special_tokens_map.json
    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad/checkpoint-4000
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-4000/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-4000/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-4000/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/checkpoint-4000/special_tokens_map.json
    ***** Running Evaluation *****
      Num examples = 10784
      Batch size = 64




    <div>

      <progress value='4152' max='4152' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [4152/4152 1:20:53, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.499000</td>
      <td>1.265836</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.120600</td>
      <td>1.161048</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.989800</td>
      <td>1.162277</td>
    </tr>
  </tbody>
</table><p>


    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    





    TrainOutput(global_step=4152, training_loss=1.3128410920012663, metrics={'train_runtime': 4855.1524, 'train_samples_per_second': 54.699, 'train_steps_per_second': 0.855, 'total_flos': 2.602335381127373e+16, 'train_loss': 1.3128410920012663, 'epoch': 3.0})



### 训练完成后，第一时间保存模型权重文件。


```python
model_to_save = trainer.save_model(model_dir)
```

    Saving model checkpoint to models/distilbert-base-uncased-finetuned-squad
    Configuration saved in models/distilbert-base-uncased-finetuned-squad/config.json
    Model weights saved in models/distilbert-base-uncased-finetuned-squad/pytorch_model.bin
    tokenizer config file saved in models/distilbert-base-uncased-finetuned-squad/tokenizer_config.json
    Special tokens file saved in models/distilbert-base-uncased-finetuned-squad/special_tokens_map.json


## 模型评估

**评估模型输出需要一些额外的处理：将模型的预测映射回上下文的部分。**

模型直接输出的是预测答案的`起始位置`和`结束位置`的**logits**


```python
import torch

for batch in trainer.get_eval_dataloader():
    break
batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
with torch.no_grad():
    output = trainer.model(**batch)
output.keys()
```




    odict_keys(['loss', 'start_logits', 'end_logits'])



模型的输出是一个类似字典的对象，其中包含损失（因为我们提供了标签），以及起始和结束logits。我们不需要损失来进行预测，让我们看一下logits：


```python
output.start_logits.shape, output.end_logits.shape
```




    (torch.Size([64, 384]), torch.Size([64, 384]))




```python
output.start_logits.argmax(dim=-1), output.end_logits.argmax(dim=-1)
```




    (tensor([ 46,  57,  78,  43, 118, 108,  72,  35, 108,  34,  73,  41,  80,  86,
             156,  35,  83,  86,  80,  58,  77,  31,  42,  53,  41,  35,  42,  77,
              11,  44,  27, 133,  66,  40,  87,  44,  43,  41, 127,  26,  28,  33,
              87, 127,  95,  25,  43, 132,  42,  29,  44,  46,  24,  44,  65,  58,
              81,  14,  59,  72,  25,  36,  55,  43], device='cuda:0'),
     tensor([ 47,  58,  81,  44, 118, 109,  75,  37, 109,  36,  76,  42,  83,  94,
             158,  35,  83,  94,  83,  60,  37,  31,  43,  54,  42,  35,  43,  80,
              13,  45,  28, 133,  66,  41,  89,  45,  44,  42, 127,  27,  30,  34,
              89, 127,  97,  26,  44, 132,  43,  30,  45,  47,  25,  45,  65,  59,
              81,  14,  60,  72,  25,  36,  58,  43], device='cuda:0'))



#### 如何从模型输出的位置 logit 组合成答案

我们有每个特征和每个标记的logit。在每个特征中为每个标记预测答案最明显的方法是，将起始logits的最大索引作为起始位置，将结束logits的最大索引作为结束位置。

在许多情况下这种方式效果很好，但是如果此预测给出了不可能的结果该怎么办？比如：起始位置可能大于结束位置，或者指向问题中的文本片段而不是答案。在这种情况下，我们可能希望查看第二好的预测，看它是否给出了一个可能的答案，并选择它。

选择第二好的答案并不像选择最佳答案那么容易：
- 它是起始logits中第二佳索引与结束logits中最佳索引吗？
- 还是起始logits中最佳索引与结束logits中第二佳索引？
- 如果第二好的答案也不可能，那么对于第三好的答案，情况会更加棘手。

为了对答案进行分类，
1. 将使用通过添加起始和结束logits获得的分数
1. 设计一个名为`n_best_size`的超参数，限制不对所有可能的答案进行排序。
1. 我们将选择起始和结束logits中的最佳索引，并收集这些预测的所有答案。
1. 在检查每一个是否有效后，我们将按照其分数对它们进行排序，并保留最佳的答案。

以下是我们如何在批次中的第一个特征上执行此操作的示例：


```python
n_best_size = 20
```


```python
import numpy as np

start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()

# 获取最佳的起始和结束位置的索引：
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

valid_answers = []

# 遍历起始位置和结束位置的索引组合
for start_index in start_indexes:
    for end_index in end_indexes:
        if start_index <= end_index:  # 需要进一步测试以检查答案是否在上下文中
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": ""  # 我们需要找到一种方法来获取与上下文中答案对应的原始子字符串
                }
            )

```


然后，我们可以根据它们的得分对`valid_answers`进行排序，并仅保留最佳答案。唯一剩下的问题是如何检查给定的跨度是否在上下文中（而不是问题中），以及如何获取其中的文本。为此，我们需要向我们的验证特征添加两个内容：

- 生成该特征的示例的ID（因为每个示例可以生成多个特征，如前所示）；
- 偏移映射，它将为我们提供从标记索引到上下文中字符位置的映射。

这就是为什么我们将使用以下函数稍微不同于`prepare_train_features`来重新处理验证集：


```python
def prepare_validation_features(examples):
    # 一些问题的左侧有很多空白，这些空白并不有用且会导致上下文截断失败（分词后的问题会占用很多空间）。
    # 因此我们移除这些左侧空白
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # 使用截断和可能的填充对我们的示例进行分词，但使用步长保留溢出的令牌。这导致一个长上下文的示例可能产生
    # 几个特征，每个特征的上下文都会稍微与前一个特征的上下文重叠。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于一个示例在上下文很长时可能会产生几个特征，我们需要一个从特征映射到其对应示例的映射。这个键就是为了这个目的。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # 我们保留产生这个特征的示例ID，并且会存储偏移映射。
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # 获取与该示例对应的序列（以了解哪些是上下文，哪些是问题）。
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # 一个示例可以产生几个文本段，这里是包含该文本段的示例的索引。
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # 将不属于上下文的偏移映射设置为None，以便容易确定一个令牌位置是否属于上下文。
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

```

将`prepare_validation_features`应用到整个验证集：


```python
validation_features = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names
)
```


    Map:   0%|          | 0/10570 [00:00<?, ? examples/s]


Now we can grab the predictions for all features by using the `Trainer.predict` method:


```python
raw_predictions = trainer.predict(validation_features)
```

    The following columns in the test set don't have a corresponding argument in `DistilBertForQuestionAnswering.forward` and have been ignored: offset_mapping, example_id. If offset_mapping, example_id are not expected by `DistilBertForQuestionAnswering.forward`,  you can safely ignore this message.
    ***** Running Prediction *****
      Num examples = 10784
      Batch size = 64




<div>

  <progress value='19' max='169' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [ 19/169 00:07 < 01:02, 2.40 it/s]
</div>



`Trainer`会隐藏模型不使用的列（在这里是`example_id`和`offset_mapping`，我们需要它们进行后处理），所以我们需要将它们重新设置回来：


```python
validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
```

现在，我们可以改进之前的测试：

由于在偏移映射中，当它对应于问题的一部分时，我们将其设置为None，因此可以轻松检查答案是否完全在上下文中。我们还可以从考虑中排除非常长的答案（可以调整的超参数）。

展开说下具体实现：
- 首先从模型输出中获取起始和结束的逻辑值（logits），这些值表明答案在文本中可能开始和结束的位置。
- 然后，它使用偏移映射（offset_mapping）来找到这些逻辑值在原始文本中的具体位置。
- 接下来，代码遍历可能的开始和结束索引组合，排除那些不在上下文范围内或长度不合适的答案。
- 对于有效的答案，它计算出一个分数（基于开始和结束逻辑值的和），并将答案及其分数存储起来。
- 最后，它根据分数对答案进行排序，并返回得分最高的几个答案。


```python
max_answer_length = 30
```


```python
start_logits = output.start_logits[0].cpu().numpy()
end_logits = output.end_logits[0].cpu().numpy()
offset_mapping = validation_features[0]["offset_mapping"]

# 第一个特征来自第一个示例。对于更一般的情况，我们需要将example_id匹配到一个示例索引
context = datasets["validation"][0]["context"]

# 收集最佳开始/结束逻辑的索引：
start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
valid_answers = []
for start_index in start_indexes:
    for end_index in end_indexes:
        # 不考虑超出范围的答案，原因是索引超出范围或对应于输入ID的部分不在上下文中。
        if (
            start_index >= len(offset_mapping)
            or end_index >= len(offset_mapping)
            or offset_mapping[start_index] is None
            or offset_mapping[end_index] is None
        ):
            continue
        # 不考虑长度小于0或大于max_answer_length的答案。
        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
            continue
        if start_index <= end_index: # 我们需要细化这个测试，以检查答案是否在上下文中
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            valid_answers.append(
                {
                    "score": start_logits[start_index] + end_logits[end_index],
                    "text": context[start_char: end_char]
                }
            )

valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
valid_answers

```

打印比较模型输出和标准答案（Ground-truth）是否一致:


```python
datasets["validation"][0]["answers"]
```




    {'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'],
     'answer_start': [177, 177, 177]}



**模型最高概率的输出与标准答案一致**

正如上面的代码所示，这在第一个特征上很容易，因为我们知道它来自第一个示例。

对于其他特征，我们需要建立一个示例与其对应特征的映射关系。

此外，由于一个示例可以生成多个特征，我们需要将由给定示例生成的所有特征中的所有答案汇集在一起，然后选择最佳答案。

下面的代码构建了一个示例索引到其对应特征索引的映射关系：


```python
import collections

examples = datasets["validation"]
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)
```

当`squad_v2 = True`时，有一定概率出现不可能的答案（impossible answer)。

上面的代码仅保留在上下文中的答案，我们还需要获取不可能答案的分数（其起始和结束索引对应于CLS标记的索引）。

当一个示例生成多个特征时，我们必须在所有特征中的不可能答案都预测出现不可能答案时（因为一个特征可能之所以能够预测出不可能答案，是因为答案不在它可以访问的上下文部分），这就是为什么一个示例中不可能答案的分数是该示例生成的每个特征中的不可能答案的分数的最小值。


```python
from tqdm.auto import tqdm

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # 构建一个从示例到其对应特征的映射。
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # 我们需要填充的字典。
    predictions = collections.OrderedDict()

    # 日志记录。
    print(f"正在后处理 {len(examples)} 个示例的预测，这些预测分散在 {len(features)} 个特征中。")

    # 遍历所有示例！
    for example_index, example in enumerate(tqdm(examples)):
        # 这些是与当前示例关联的特征的索引。
        feature_indices = features_per_example[example_index]

        min_null_score = None # 仅在squad_v2为True时使用。
        valid_answers = []

        context = example["context"]
        # 遍历与当前示例关联的所有特征。
        for feature_index in feature_indices:
            # 我们获取模型对这个特征的预测。
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # 这将允许我们将logits中的某些位置映射到原始上下文中的文本跨度。
            offset_mapping = features[feature_index]["offset_mapping"]

            # 更新最小空预测。
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # 浏览所有的最佳开始和结束logits，为 `n_best_size` 个最佳选择。
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 不考虑超出范围的答案，原因是索引超出范围或对应于输入ID的部分不在上下文中。
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 不考虑长度小于0或大于max_answer_length的答案。
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # 在极少数情况下我们没有一个非空预测，我们创建一个假预测以避免失败。
            best_answer = {"text": "", "score": 0.0}

        # 选择我们的最终答案：最佳答案或空答案（仅适用于squad_v2）
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

```

在原始结果上应用后处理问答结果：


```python
final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)
```

    正在后处理 10570 个示例的预测，这些预测分散在 10784 个特征中。



      0%|          | 0/10570 [00:00<?, ?it/s]



```python
from datasets import load_metric

metric = load_metric("squad_v2" if squad_v2 else "squad")
```

    <ipython-input-53-a4a02e9d074a>:3: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
      metric = load_metric("squad_v2" if squad_v2 else "squad")
    /usr/local/lib/python3.10/dist-packages/datasets/load.py:759: FutureWarning: The repository for squad contains custom code which must be executed to correctly load the metric. You can inspect the repository content at https://raw.githubusercontent.com/huggingface/datasets/2.19.0/metrics/squad/squad.py
    You can avoid this message in future by passing the argument `trust_remote_code=True`.
    Passing `trust_remote_code=True` will be mandatory to load this metric from the next major release of `datasets`.
      warnings.warn(



    Downloading builder script:   0%|          | 0.00/1.72k [00:00<?, ?B/s]



    Downloading extra modules:   0%|          | 0.00/1.11k [00:00<?, ?B/s]



```python
if squad_v2:
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
else:
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)
```




    {'exact_match': 74.66414380321665, 'f1': 83.56779611122091}



使用 `datasets.load_metric` 中加载 `SQuAD v2` 的评估指标


```python
from datasets import load_metric

metric = load_metric("squad_v2" if squad_v2 else "squad")
```

    /usr/local/lib/python3.10/dist-packages/datasets/load.py:759: FutureWarning: The repository for squad contains custom code which must be executed to correctly load the metric. You can inspect the repository content at https://raw.githubusercontent.com/huggingface/datasets/2.19.0/metrics/squad/squad.py
    You can avoid this message in future by passing the argument `trust_remote_code=True`.
    Passing `trust_remote_code=True` will be mandatory to load this metric from the next major release of `datasets`.
      warnings.warn(


接下来，我们可以调用上面定义的函数进行评估。

只需稍微调整一下预测和标签的格式，因为它期望的是一系列字典而不是一个大字典。

在使用`squad_v2`数据集时，我们还需要设置`no_answer_probability`参数（我们在这里将其设置为0.0，因为如果我们选择了答案，我们已经将答案设置为空）。


```python
if squad_v2:
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
else:
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)
```




    {'exact_match': 74.66414380321665, 'f1': 83.56779611122091}



### Homework：加载本地保存的模型，进行评估和再训练更高的 F1 Score


```python
trained_model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
```

    loading configuration file models/distilbert-base-uncased-finetuned-squad/config.json
    Model config DistilBertConfig {
      "_name_or_path": "models/distilbert-base-uncased-finetuned-squad",
      "activation": "gelu",
      "architectures": [
        "DistilBertForQuestionAnswering"
      ],
      "attention_dropout": 0.1,
      "dim": 768,
      "dropout": 0.1,
      "hidden_dim": 3072,
      "initializer_range": 0.02,
      "max_position_embeddings": 512,
      "model_type": "distilbert",
      "n_heads": 12,
      "n_layers": 6,
      "pad_token_id": 0,
      "qa_dropout": 0.1,
      "seq_classif_dropout": 0.2,
      "sinusoidal_pos_embds": false,
      "tie_weights_": true,
      "torch_dtype": "float32",
      "transformers_version": "4.24.0",
      "vocab_size": 30522
    }
    
    loading weights file models/distilbert-base-uncased-finetuned-squad/pytorch_model.bin
    All model checkpoint weights were used when initializing DistilBertForQuestionAnswering.
    
    All the weights of DistilBertForQuestionAnswering were initialized from the model checkpoint at models/distilbert-base-uncased-finetuned-squad.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use DistilBertForQuestionAnswering for predictions without further training.



```python
trained_trainer = Trainer(
    trained_model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```
