```python
import os

os.environ['HF_HOME'] = '/mnt/new_volume/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/new_volume/hf/hub'
```


```python
from transformers import pipeline

# 仅指定任务时，使用默认模型（不推荐）
pipe = pipeline("sentiment-analysis") #sentiment-analysis 参数指定了管道的类型，即情感分析
pipe("今儿上海可真冷啊")
```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(





    [{'label': 'NEGATIVE', 'score': 0.8957217335700989}]




```python
pipe("我觉得这家店蒜泥白肉的味道一般")
```




    [{'label': 'NEGATIVE', 'score': 0.9238731265068054}]




```python
# 默认使用的模型 distilbert-base-uncased-finetuned-sst-2-english
# 并未针对中文做太多训练，中文的文本分类任务表现未必满意
pipe("你学东西真的好快，理论课一讲就明白了")
```




    [{'label': 'NEGATIVE', 'score': 0.8578693866729736}]




```python
# 替换为英文后，文本分类任务的表现立刻改善
pipe("You learn things really quickly. You understand the theory class as soon as it is taught.")
```




    [{'label': 'POSITIVE', 'score': 0.9961802959442139}]




```python
pipe("Today Shanghai is really cold.")
```




    [{'label': 'NEGATIVE', 'score': 0.9995032548904419}]



### 批处理调用模型推理


```python
text_list = [
    "Today Shanghai is really cold.",
    "I think the taste of the garlic mashed pork in this store is average.",
    "You learn things really quickly. You understand the theory class as soon as it is taught."
]

pipe(text_list)
```




    [{'label': 'NEGATIVE', 'score': 0.9995032548904419},
     {'label': 'NEGATIVE', 'score': 0.9984821677207947},
     {'label': 'POSITIVE', 'score': 0.9961802959442139}]



## 使用 Pipeline API 调用更多预定义任务

## Natural Language Processing(NLP)


```python
from transformers import pipeline

classifier = pipeline(task="ner") #创建了一个命名实体识别（Named Entity Recognition，NER）的管道。命名实体识别是一种自然语言处理任务，其目标是从文本中识别出具有特定意义的命名实体，如人名、地名、组织机构名等。
```

    No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision f2482bf (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
preds = classifier("Hugging Face is a French company based in New York City.")
preds = [
    {
        "entity": pred["entity"],
        "score": round(pred["score"], 4),
        "index": pred["index"],
        "word": pred["word"],
        "start": pred["start"],
        "end": pred["end"],
    }
    for pred in preds
]
print(*preds, sep="\n")
```

    {'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
    {'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
    {'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
    {'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
    {'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
    {'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
    {'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}


#### 合并实体


```python
classifier = pipeline(task="ner", grouped_entities=True) #grouped_entities=True 参数告诉命名实体识别（NER）管道将相邻的同类实体组合在一起。
classifier("Hugging Face is a French company based in New York City.")
```

    No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision f2482bf (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    /usr/local/lib/python3.10/dist-packages/transformers/pipelines/token_classification.py:168: UserWarning: `grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to `aggregation_strategy="simple"` instead.
      warnings.warn(





    [{'entity_group': 'ORG',
      'score': 0.96746373,
      'word': 'Hugging Face',
      'start': 0,
      'end': 12},
     {'entity_group': 'MISC',
      'score': 0.99828726,
      'word': 'French',
      'start': 18,
      'end': 24},
     {'entity_group': 'LOC',
      'score': 0.99896103,
      'word': 'New York City',
      'start': 42,
      'end': 55}]



### Question Answering

**Question Answering**(问答)是另一个token-level的任务，返回一个问题的答案，有时带有上下文（开放领域），有时不带上下文（封闭领域）。每当我们向虚拟助手提出问题时，例如询问一家餐厅是否营业，就会发生这种情况。它还可以提供客户或技术支持，并帮助搜索引擎检索您要求的相关信息。


```python
from transformers import pipeline

question_answerer = pipeline(task="question-answering") #创建了一个问答系统的管道。问答系统是一种人工智能应用程序，能够回答关于给定文本的问题。
```

    No model was supplied, defaulted to distilbert/distilbert-base-cased-distilled-squad and revision 626af31 (https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad).
    Using a pipeline without specifying a model name and revision in production is not recommended.



```python
preds = question_answerer(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
print(
    f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
)
```

    score: 0.9327, start: 30, end: 54, answer: huggingface/transformers



这段代码是使用了一个已经创建好的问答系统管道来回答一个问题。具体来说：

question_answerer 是一个已经通过 Hugging Face Transformers 库创建的问答系统管道。
question 参数指定了要问的问题，这里是 "What is the name of the repository?"。
context 参数是提供给问答系统的文本信息，用来寻找答案。在这个例子中，文本是 "The name of the repository is huggingface/transformers"。
程序通过调用 question_answerer 管道来获取问题的答案，并将结果存储在 preds 变量中。
最后，通过打印输出了预测结果的相关信息，包括分数（score）、开始位置（start）、结束位置（end）、以及答案（answer）。


```python
preds = question_answerer(
    question="What is the capital of China?",
    context="On 1 October 1949, CCP Chairman Mao Zedong formally proclaimed the People's Republic of China in Tiananmen Square, Beijing.",
)
print(
    f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
)
```

    score: 0.9458, start: 115, end: 122, answer: Beijing


### Summarization

**Summarization**(文本摘要）从较长的文本中创建一个较短的版本，同时尽可能保留原始文档的大部分含义。摘要是一个序列到序列的任务；它输出比输入更短的文本序列。有许多长篇文档可以进行摘要，以帮助读者快速了解主要要点。法案、法律和财务文件、专利和科学论文等文档可以摘要，以节省读者的时间并作为阅读辅助工具。


```python
from transformers import pipeline

summarizer = pipeline(task="summarization",
                      model="t5-base",
                      min_length=8,
                      max_length=32,
)
```


```python
summarizer(
    """
    In this work, we presented the Transformer, the first sequence transduction model based entirely on attention,
    replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
    For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.
    On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art.
    In the former task our best model outperforms even all previously reported ensembles.
    """
)

```




    [{'summary_text': 'the Transformer is the first sequence transduction model based entirely on attention . it replaces recurrent layers commonly used in encoder-decode'}]




```python
summarizer(
    '''
    Large language models (LLM) are very large deep learning models that are pre-trained on vast amounts of data.
    The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities.
    The encoder and decoder extract meanings from a sequence of text and understand the relationships between words and phrases in it.
    Transformer LLMs are capable of unsupervised training, although a more precise explanation is that transformers perform self-learning.
    It is through this process that transformers learn to understand basic grammar, languages, and knowledge.
    Unlike earlier recurrent neural networks (RNN) that sequentially process inputs, transformers process entire sequences in parallel.
    This allows the data scientists to use GPUs for training transformer-based LLMs, significantly reducing the training time.
    '''
)

```




    [{'summary_text': 'large language models (LLMs) are very large deep learning models pre-trained on vast amounts of data . transformers are capable of un'}]




## Audio 音频处理任务

音频和语音处理任务与其他模态略有不同，主要是因为音频作为输入是一个连续的信号。与文本不同，原始音频波形不能像句子可以被划分为单词那样被整齐地分割成离散的块。为了解决这个问题，通常在固定的时间间隔内对原始音频信号进行采样。如果在每个时间间隔内采样更多样本，采样率就会更高，音频更接近原始音频源。

以前的方法是预处理音频以从中提取有用的特征。现在更常见的做法是直接将原始音频波形输入到特征编码器中，以提取音频表示。这样可以简化预处理步骤，并允许模型学习最重要的特征。


```python
from transformers import pipeline

classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er") #创建了一个音频分类的管道，并且使用了特定的预训练模型 superb/hubert-base-superb-er 来进行音频分类任务。
```

    Some weights of the model checkpoint at superb/hubert-base-superb-er were not used when initializing HubertForSequenceClassification: ['hubert.encoder.pos_conv_embed.conv.weight_g', 'hubert.encoder.pos_conv_embed.conv.weight_v']
    - This IS expected if you are initializing HubertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing HubertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of HubertForSequenceClassification were not initialized from the model checkpoint at superb/hubert-base-superb-er and are newly initialized: ['hubert.encoder.pos_conv_embed.conv.parametrizations.weight.original0', 'hubert.encoder.pos_conv_embed.conv.parametrizations.weight.original1']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
# 使用 Hugging Face Datasets 上的测试文件
preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds
```




    [{'score': 0.4532, 'label': 'hap'},
     {'score': 0.3622, 'label': 'sad'},
     {'score': 0.0943, 'label': 'neu'},
     {'score': 0.0903, 'label': 'ang'}]



### Automatic speech recognition（ASR）

**Automatic speech recognition**（自动语音识别）将语音转录为文本。这是最常见的音频任务之一，部分原因是因为语音是人类交流的自然形式。如今，ASR系统嵌入在智能技术产品中，如扬声器、电话和汽车。我们可以要求虚拟助手播放音乐、设置提醒和告诉我们天气。

第一个字典表示分类为 "hap"（愉快）类别，预测得分为 0.4532。
第二个字典表示分类为 "sad"（悲伤）类别，预测得分为 0.3622。
第三个字典表示分类为 "neu"（中性）类别，预测得分为 0.0943。
第四个字典表示分类为 "ang"（愤怒）类别，预测得分为 0.0903。


```python
from transformers import pipeline

# 使用 `model` 参数指定模型
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
```

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.



```python
text = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
text
```

    Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English.This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`.





    {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}



## Computer Vision 计算机视觉

**Computer Vision**（计算机视觉）任务中最早成功之一是使用卷积神经网络（CNN）识别邮政编码数字图像。图像由像素组成，每个像素都有一个数值。这使得将图像表示为像素值矩阵变得容易。每个像素值组合描述了图像的颜色。


### Image Classificaiton

**Image Classificaiton**(图像分类)将整个图像从预定义的类别集合中进行标记。像大多数分类任务一样，图像分类有许多实际用例，其中一些包括：

- 医疗保健：标记医学图像以检测疾病或监测患者健康状况
- 环境：标记卫星图像以监测森林砍伐、提供野外管理信息或检测野火
- 农业：标记农作物图像以监测植物健康或用于土地使用监测的卫星图像
- 生态学：标记动物或植物物种的图像以监测野生动物种群或跟踪濒危物种


```python
from transformers import pipeline

classifier = pipeline(task="image-classification")
```

    No model was supplied, defaulted to google/vit-base-patch16-224 and revision 5dca96d (https://huggingface.co/google/vit-base-patch16-224).
    Using a pipeline without specifying a model name and revision in production is not recommended.



```python
preds = classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
print(*preds, sep="\n")
```

    {'score': 0.4335, 'label': 'lynx, catamount'}
    {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
    {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
    {'score': 0.0239, 'label': 'Egyptian cat'}
    {'score': 0.0229, 'label': 'tiger cat'}


### Object Detection

与图像分类不同，目标检测在图像中识别多个对象以及这些对象在图像中的位置（由边界框定义）。目标检测的一些示例应用包括：

### Object Detection

与图像分类不同，目标检测在图像中识别多个对象以及这些对象在图像中的位置（由边界框定义）。目标检测的一些示例应用包括：

- 自动驾驶车辆：检测日常交通对象，如其他车辆、行人和红绿灯
- 遥感：灾害监测、城市规划和天气预报
- 缺陷检测：检测建筑物中的裂缝或结构损坏，以及制造业产品缺陷


```python
!pip install timm
```

    Requirement already satisfied: timm in /usr/local/lib/python3.10/dist-packages (0.9.16)
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from timm) (2.2.1+cu121)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (from timm) (0.17.1+cu121)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from timm) (6.0.1)
    Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.10/dist-packages (from timm) (0.20.3)
    Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from timm) (0.4.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (3.13.4)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (2023.6.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (4.66.2)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (4.11.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (24.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->timm) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->timm) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (3.1.3)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.19.3 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (2.19.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (12.1.105)
    Requirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (2.2.0)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch->timm) (12.4.127)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (1.25.2)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (9.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->timm) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (2024.2.2)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->timm) (1.3.0)



```python
from transformers import pipeline

detector = pipeline(task="object-detection")
```

    No model was supplied, defaulted to facebook/detr-resnet-50 and revision 2729413 (https://huggingface.co/facebook/detr-resnet-50).
    Using a pipeline without specifying a model name and revision in production is not recommended.



    model.safetensors:   0%|          | 0.00/102M [00:00<?, ?B/s]


    Some weights of the model checkpoint at facebook/detr-resnet-50 were not used when initializing DetrForObjectDetection: ['model.backbone.conv_encoder.model.layer1.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer2.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer3.0.downsample.1.num_batches_tracked', 'model.backbone.conv_encoder.model.layer4.0.downsample.1.num_batches_tracked']
    - This IS expected if you are initializing DetrForObjectDetection from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DetrForObjectDetection from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



    preprocessor_config.json:   0%|          | 0.00/290 [00:00<?, ?B/s]



```python
preds = detector(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
preds
```




    [{'score': 0.9864,
      'label': 'cat',
      'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]


