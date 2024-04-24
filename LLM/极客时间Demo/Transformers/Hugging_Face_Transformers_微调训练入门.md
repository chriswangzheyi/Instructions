# Hugging Face Transformers 微调训练入门

本示例将介绍基于 Transformers 实现模型微调训练的主要流程，包括：
- 数据集下载
- 数据预处理
- 训练超参数配置
- 训练评估指标设置
- 训练器基本介绍
- 实战训练
- 模型保存

# YelpReviewFull 数据集

**Hugging Face 数据集：[ YelpReviewFull ](https://huggingface.co/datasets/yelp_review_full)**

### 数据集摘要

Yelp评论数据集包括来自Yelp的评论。它是从Yelp Dataset Challenge 2015数据中提取的。

### 支持的任务和排行榜
文本分类、情感分类：该数据集主要用于文本分类：给定文本，预测情感。

### 语言
这些评论主要以英语编写。

### 数据集结构

#### 数据实例
一个典型的数据点包括文本和相应的标签。

来自YelpReviewFull测试集的示例如下：

```json
{
    'label': 0,
    'text': 'I got \'new\' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \\nI took the tire over to Flynn\'s and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers? I find that very unlikely. After arguing with the guy and telling him that his logic was far fetched he said he\'d give me a new tire \\"this time\\". \\nI will never go back to Flynn\'s b/c of the way this guy treated me and the simple fact that they gave me a used tire!'
}
```

#### 数据字段

- 'text': 评论文本使用双引号（"）转义，任何内部双引号都通过2个双引号（""）转义。换行符使用反斜杠后跟一个 "n" 字符转义，即 "\n"。
- 'label': 对应于评论的分数（介于1和5之间）。

#### 数据拆分

Yelp评论完整星级数据集是通过随机选取每个1到5星评论的130,000个训练样本和10,000个测试样本构建的。总共有650,000个训练样本和50,000个测试样本。

## 下载数据集


```python
!pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
```

```python
dataset
```




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




```python
dataset["train"][111]
```


    {'label': 2,
     'text': "As far as Starbucks go, this is a pretty nice one.  The baristas are friendly and while I was here, a lot of regulars must have come in, because they bantered away with almost everyone.  The bathroom was clean and well maintained and the trash wasn't overflowing in the canisters around the store.  The pastries looked fresh, but I didn't partake.  The noise level was also at a nice working level - not too loud, music just barely audible.\\n\\nI do wish there was more seating.  It is nice that this location has a counter at the end of the bar for sole workers, but it doesn't replace more tables.  I'm sure this isn't as much of a problem in the summer when there's the space outside.\\n\\nThere was a treat receipt promo going on, but the barista didn't tell me about it, which I found odd.  Usually when they have promos like that going on, they ask everyone if they want their receipt to come back later in the day to claim whatever the offer is.  Today it was one of their new pastries for $1, I know in the summer they do $2 grande iced drinks with that morning's receipt.\\n\\nOverall, nice working or socializing environment.  Very friendly and inviting.  It's what I've come to expect from Starbucks, so points for consistency."}




```python
import random
import pandas as pd
import datasets
from IPython.display import display, HTML
```


```python
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
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
```


这个函数的名称是 show_random_elements，它的作用是从给定的数据集中随机选择一定数量的示例，并展示这些示例的内容。


```python
show_random_elements(dataset["train"])
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4 stars</td>
      <td>My mom and I shops here for necessary things such as: food and  household supplies. A great place to shop, place is clean &amp; organized.  \n\nTheir little cafe-ish that they have inside is a good deal for their food! and it tastes pretty decent, too!  \n\nTheir optical area has a great selection of eye-wear. I fell in love with a few of them, too! The opticians are so friendly! They have cheaper contact lens here than walmart! Since I pay full price on my lens, I would go to walmart thinking it was the cheapest place I could buy my contact lens.  But then I finally decided to look at sam's contact lens prices &amp; wow I could have been saving at least $30 - $40 each time I buy a box of contact lens.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5 stars</td>
      <td>Yelp 100 Challenge 14' * 12/100\n\nThe Springs Preserve continues to be one of my favorite go-to places when the weather hints at a transition for spring. (Considering all the open areas here, maybe summer is not the best place to check it out) Since I turned 20, I only went to malls for art galleries and annual/semi-annual sales...or if I had no other choice. Other times, I liked to immerse myself in a quiet, naturistic place where birds sing and wind rustles through the leaves. \n\nSince I've gotten back into school &amp; work mode, I haven't had a lot of time to spare for my family and mommy, it being her day off on Monday, asked if we could have a picnic together. We both got up early, got ourselves tickets to the garden (it's free) and had a nice breakfast in one of the shaded tables next to the gallery. The weather was perfect that day, 75 degrees, warm enough for t-shirts and sunblock. We had a nice chat before heading to the artistically landscaped gardens. \n\nRecently, the preserve unraveled their new solar house - desertsol - open for people who are interested in sustainable living and looking for inspiration on how to build themselves an environmentally friendly house. I believe it's only available for viewing upon a general admission ticket purchase but from what I've heard, it's definitely worth it. New harvests have begun to bud, revealing tomato leaves and all sorts of beautiful flowers.\n\nIt's a great get away from hype that Vegas is and offers a relaxing atmosphere for nature lovers. If only it won't rain tomorrow (not complaining, I love the occasional gloomy days :), I'd been looking forward to their mardi gras event all week.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1 star</td>
      <td>I have had 2 appointments with Dr. Dolinar. Neither have been great, but this one warranted a review. I have been a type 1 Diabetic for about 8 years, so I know the in's and out's of the disease. I have never been treated so poorly in my life by a professional. I was told that my major in college doesn't appear to be a good fit with me, and was belittled to the point of tears. He didn't address any of my concerns and reason for seeking medical care either time of visiting him. I came in initially due to a new medication making me very, very ill; and was told I would have to come back in 6 weeks. I came back, and nothing. Didn't address any concerns of mine, or ask if anything has changed since last time. He continued to 'encourage' me to buy a book, that he happened to write, and that was it. Very disappointed and won't be coming back.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5 stars</td>
      <td>This one of my favorite restaurants in Pittsburgh. The menu features a variety of interesting options and changes on a regular basis. Their brunch is also one of the best in town. The chicken and waffles is amazing and the breakfast cocktails are unique and different.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3 stars</td>
      <td>A lil disappointed with the food. I love GR!! The cocktail shrimp was tasteless and over cooked, very chewy. The Mac and cheese was tasteless I was looking for a pungent cheese flavor and got barely any flavor. The tempura green beans were the best out of the 3 sides but a lil salty for my taste. My Wellington was cooked perfectly but the mushroom sauce thing inside was too salty for me. The meat was delicious! The bone in ribeyes my boyfriend and cousin ate were too charred and too fatty so it was hard to take down. Overall I would rate this place 2.5 stars and it breaks my heart because I love Gordon and couldnt wait to visit this place :(</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2 star</td>
      <td>This just simply isn't MEC.... where's the gear?! I dont go to MEC to browse for clothes. .... And while you have my attention, how about that new logo? PUT THE MOUNTAIN BACK.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4 stars</td>
      <td>Love the food, it's always consistently delicious. My usuals are toasted pastrami (yum!!!!), calamari Caesar salad or a slice of pizza and salad. It's on the regular rotation for lunch. Check it out!</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5 stars</td>
      <td>Best nail salon EVER! I do not and will not go anywhere else! Clean, fairly priced, friendly, accommodating, and full of talent!!</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4 stars</td>
      <td>I visited StripSteak only because a friend was in love with their macaroni and cheese and wanted to see if she could get it carryout after we'd attended a concert at The Beach in Mandalay Bay.  While the hostesses at the podium at first seemed a little uncertain about whether we'd be able to order it, they directed us to the bar where the bartender was more than happy to place the order for us.  \n\nThe bartender was excellent- he was very friendly and asked us where we were from. I thought great, I'm from Vegas so he'll probably ignore us after we tell him.  But no- he was just as friendly, and after my friend left to play in the casino while Iwaited for our order, he brought me water with lemon (and refilled the glass) and continued to chat.  \n\nI guess this is just a review of the bar and the bartender, but I must add that the restaurant was beautiful- shiny floors, and some of the tables were glowing from lights underneath them.  I want to come here now and experience the restaurant due to the good first impression I got.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4 stars</td>
      <td>It's so refreshing to see a local restauranteur give Phoenicians more vegetarian and vegan options when dining out, and thanks to Sam Fox of Fox Restaurant Concepts, True Food Kitchen hooks veggies like me up in a trendy setting. Everything on the menu is clearly outlined if it's vegetarian or vegan, so people with all types of diets will be happy.\n\nThe Biltmore Fashion Park restaurant even looks healthy, with a bright green motif and servers donning bright green capri pants-not the most fashionable get-up, but it mos def drives the point home that the eatery is going for a healthy, green vibe.\n\nAnd, like Fox's others restaurants, this one has an open kitchen, though here, the kitchen/prep area seemed to take up more than half the restaurant, which I thought was pretty odd.\n\nThe crowd of diners was varied. There were plenty of 20-somethings, a ton of middle-aged folks and even families with babies mixed in.\n\nWhile the atmosphere was slightly above average, the food was excellent. I started with a bowl of mushroom-filled miso soup ($6), which was vegan and chock full of shrooms, scallions and tofu. I loved it!\n\nI paired my meal with a Cucumber Refresher ($4), a cucumber-infused honey lemonade. It was sweet, refreshing and didn't make me feel guilty about drinking it.\n\nFor my entree, I ordered a ricotta ravioli ($13), which was covered in a pesto sauce, more mushrooms and red peppers. It was an interesting twist on traditional tomato sauce-covered pasta, and it really filled me up.\n\nFor me, I'll definitely be going back to try more tasty meat-free bites.</td>
    </tr>
  </tbody>
</table>


## 预处理数据

下载数据集到本地后，使用 Tokenizer 来处理文本，对于长度不等的输入数据，可以使用填充（padding）和截断（truncation）策略来处理。

Datasets 的 `map` 方法，支持一次性在整个数据集上应用预处理函数。

下面使用填充到最大长度的策略，处理整个数据集：


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

```python
show_random_elements(tokenized_datasets["train"], num_examples=1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
      <th>input_ids</th>
      <th>token_type_ids</th>
      <th>attention_mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 star</td>
      <td>BOTTOM LINE: TPSBFHDB (this place sucks big fat hairy donkey balls). The service was absolute poppycock and the food was rubbish.\n\nMe and my 5 DCs had a late dinner here after a long day at Osheaga mainly because most of the other kitchens along this strip were closed. We might of been able to avoid this disaster of a place if only we had consulted with Yelp first.\n\nWe all ordered vermicelli bowls since we were starving and it's quick to make...or so we though. We ended up waiting over 30 minutes for the dishes to arrive and they were brought 5-10 minutes apart! No, they were not busy - there was only another table of 4 that had ordered. Given that it takes all of 60 seconds to cook vermicelli and another 2-3 minutes to grill the meat...FAIL.\n\nThe EPIC FAIL, however, began when I enquired about my forgotten order. The waiter gave me a dirty look of acknowledgement and trotted off to the kitchen. My vermicelli bowl was soon plopped down in front of me with 2 small burnt pieces of beef and partially cooked vermicelli that was still hard! I complained about the laughably small portion (compared to my DC's orders) and was told that because I got a spring roll, I got less beef than everyone else. WTF? JTFC! Spring rolls comes with each vermicelli by default! I then showed him the raw vermicelli, and he gave me yet another exasperated look, took my bowl away and left without saying a word. \n\nI was so PO'ed at that point, I chased him down to cancel the order - no way was I gonna wait another 30 minutes nor risk them tampering with my food. He turned to me, tossed my bowl onto the kitchen table and left again without saying a word. No apology, no nothing. They were equally rude and indifferent to my DC who also had uncooked vermicelli and to my DC that they overcharged with items he never ordered.\n\nAnd to top it off, their vermicelli bowls really sucked (even accounting for the cooked portions of it): bland flavours, few veggies, no garnishes and a miserly spring roll filled with more vermicelli!\n\nI don't think I've ever experienced such rude and incompetent service, let alone completely cancel an order because of how quickly things went downhill. I couldn't even recommend this place to my worst enemy because it would give this place unwarranted business. Yelpers, you've been warned.</td>
      <td>[101, 139, 14697, 18082, 2107, 149, 11607, 2036, 131, 157, 10197, 26447, 23527, 2064, 113, 1142, 1282, 22797, 1992, 7930, 18419, 1274, 9144, 7318, 114, 119, 1109, 1555, 1108, 7846, 3618, 5005, 11157, 1105, 1103, 2094, 1108, 16259, 26652, 119, 165, 183, 165, 183, 2107, 1162, 1105, 1139, 126, 5227, 1116, 1125, 170, 1523, 4014, 1303, 1170, 170, 1263, 1285, 1120, 152, 21581, 15446, 2871, 1272, 1211, 1104, 1103, 1168, 3119, 1116, 1373, 1142, 6322, 1127, 1804, 119, 1284, 1547, 1104, 1151, 1682, 1106, 3644, 1142, 7286, 1104, 170, 1282, 1191, 1178, 1195, 1125, 18881, 1114, 15821, 1233, 1643, 1148, ...]</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]</td>
      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]</td>
    </tr>
  </tbody>
</table>


### 数据抽样

使用 1000 个数据样本，在 BERT 上演示小规模训练（基于 Pytorch Trainer）

`shuffle()`函数会随机重新排列列的值。如果您希望对用于洗牌数据集的算法有更多控制，可以在此函数中指定generator参数来使用不同的numpy.random.Generator。


```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

## 微调训练配置

### 加载 BERT 模型

警告通知我们正在丢弃一些权重（`vocab_transform` 和 `vocab_layer_norm` 层），并随机初始化其他一些权重（`pre_classifier` 和 `classifier` 层）。在微调模型情况下是绝对正常的，因为我们正在删除用于预训练模型的掩码语言建模任务的头部，并用一个新的头部替换它，对于这个新头部，我们没有预训练的权重，所以库会警告我们在用它进行推理之前应该对这个模型进行微调，而这正是我们要做的事情。


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#从预训练的 BERT 模型（"bert-base-cased"）创建了一个用于序列分类任务的模型。这个模型是一个用于序列分类任务的预训练模型，它可以接受输入序列并为其分配一个或多个类别标签。在这个例子中，模型被配置为有 5 个类别标签（num_labels=5）。
```


### 训练超参数（TrainingArguments）

完整配置参数与默认值：https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments

源代码定义：https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/training_args.py#L161

**最重要配置：模型权重保存路径(output_dir)**


```python
pip install transformers[torch]
```

```python
pip install accelerate -U
```

```python
from transformers import TrainingArguments

model_dir = "models/bert-base-cased-finetune-yelp"

# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=16,
                                  num_train_epochs=5,
                                  logging_steps=100)
```


```python
# 完整的超参数配置
print(training_args)
```

    TrainingArguments(
    _n_gpu=1,
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    bf16=False,
    bf16_full_eval=False,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    do_eval=False,
    do_predict=False,
    do_train=False,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_steps=None,
    evaluation_strategy=no,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    greater_is_better=None,
    group_by_length=False,
    half_precision_backend=auto,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy=every_save,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=5e-05,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=-1,
    log_level=passive,
    log_level_replica=passive,
    log_on_each_node=True,
    logging_dir=models/bert-base-cased-finetune-yelp/runs/Apr24_06-01-13_a53cc64c0d33,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=100,
    logging_strategy=steps,
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mp_parameters=,
    no_cuda=False,
    num_train_epochs=5,
    optim=adamw_hf,
    output_dir=models/bert-base-cased-finetune-yelp,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=16,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=['tensorboard'],
    resume_from_checkpoint=None,
    run_name=models/bert-base-cased-finetune-yelp,
    save_on_each_node=False,
    save_steps=500,
    save_strategy=steps,
    save_total_limit=None,
    seed=42,
    sharded_ddp=[],
    skip_memory_metrics=True,
    tf32=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_mps_device=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
    xpu_backend=None,
    )



```python
pip install evaluate
```


### 训练过程中的指标评估（Evaluate)

**[Hugging Face Evaluate 库](https://huggingface.co/docs/evaluate/index)** 支持使用一行代码，获得数十种不同领域（自然语言处理、计算机视觉、强化学习等）的评估方法。 当前支持 **完整评估指标：https://huggingface.co/evaluate-metric**

训练器（Trainer）在训练过程中不会自动评估模型性能。因此，我们需要向训练器传递一个函数来计算和报告指标。

Evaluate库提供了一个简单的准确率函数，您可以使用`evaluate.load`函数加载


```python
import numpy as np
import evaluate

metric = evaluate.load("/accuracy.py")
```


接着，调用 `compute` 函数来计算预测的准确率。

在将预测传递给 compute 函数之前，我们需要将 logits 转换为预测值（**所有Transformers 模型都返回 logits**）。


```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

#### 训练过程指标监控

通常，为了监控训练过程中的评估指标变化，我们可以在`TrainingArguments`指定`evaluation_strategy`参数，以便在 epoch 结束时报告评估指标。


```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir=model_dir,
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=16,
                                  num_train_epochs=3,
                                  logging_steps=30)
```

## 开始训练

### 实例化训练器（Trainer）

`kernel version` 版本问题：暂不影响本示例代码运行


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```


```python
trainer.train()
```



| Epoch | Training Loss | Validation Loss | Accuracy |
| ----: | ------------: | --------------: | -------: |
|     1 |      1.242100 |        1.090886 | 0.526000 |
|     2 |      0.901400 |        0.960115 | 0.591000 |
|     3 |      0.638200 |        0.978361 | 0.592000 |




```python
small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))
```


```python
trainer.evaluate(small_test_dataset)
```





### 保存模型和训练状态

- 使用 `trainer.save_model` 方法保存模型，后续可以通过 from_pretrained() 方法重新加载
- 使用 `trainer.save_state` 方法保存训练状态


```python
trainer.save_model(model_dir)
```

    Saving model checkpoint to models/bert-base-cased-finetune-yelp
    Configuration saved in models/bert-base-cased-finetune-yelp/config.json
    Model weights saved in models/bert-base-cased-finetune-yelp/pytorch_model.bin



```python
trainer.save_state()
```


```python
trainer.model.save_pretrained("./")
```

    Configuration saved in ./config.json
    Model weights saved in ./pytorch_model.bin

