## HF Transformers  微调训练模块 Trainer



## Demo代码



```python
1# ==========================================================
# 环境准备与依赖
# ==========================================================
# 若首次运行，请在命令行或 Notebook 第一个单元格执行：
# !pip install -U "transformers>=4.44" datasets evaluate accelerate torch

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate

# 固定随机种子，保证结果可复现（尤其是在shuffle、初始化等环节）
set_seed(42)

# ==========================================================
# 1️⃣ 加载 Yelp 评论数据集
# ==========================================================
# yelp_review_full 是 HuggingFace 官方提供的公开数据集，
# 包含 65 万条英文评论，标签从 0~4 分别表示 1~5 星评分。
dataset = load_dataset("yelp_review_full")

# 查看数据集的组成（train/test 划分）
print(dataset)

# 查看训练集中的第一条样本（包含 text 和 label）
print(dataset["train"][0])

# ==========================================================
# 2️⃣ 加载预训练分词器（Tokenizer）
# ==========================================================
# bert-base-cased：BERT 英文大小写敏感版本（会保留单词大写信息）
# 分词器负责把原始文本拆分为 token，并映射成模型输入的 ID。
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 定义一个分词函数，用于对每条样本的 text 字段进行编码。
def tokenize_function(examples):
    return tokenizer(
        examples["text"],         # 输入字段名为 "text"
        padding="max_length",     # 自动填充到最大长度（便于 batch 训练）
        truncation=True,          # 超出最大长度的文本会被截断
        max_length=256,           # 限定最大序列长度，Yelp 评论较长时很有用
    )

# ==========================================================
# 3️⃣ 对整个数据集批量分词
# ==========================================================
# map 会把 tokenize_function 应用于整个数据集。
# batched=True 表示每次处理一个 batch（速度更快）
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
)

# 将 "label" 列改名为 "labels"，Trainer 默认读取 "labels" 作为监督信号
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 移除原始的 "text" 列，节省内存与显存
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# ==========================================================
# 4️⃣ 加载预训练模型
# ==========================================================
# AutoModelForSequenceClassification 自动选择合适的模型架构。
# num_labels=5 代表是一个 5 分类任务（对应 Yelp 的 1~5 星）。
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=5
)

# ==========================================================
# 5️⃣ 定义评估指标函数
# ==========================================================
# 使用 evaluate 库加载常用指标，这里选 accuracy 和 f1
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

# compute_metrics 会在每次验证后自动调用
def compute_metrics(eval_pred):
    logits, labels = eval_pred          # eval_pred 是一个 (logits, labels) 元组
    preds = np.argmax(logits, axis=-1)  # 取每行最大值对应的类别
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# ==========================================================
# 6️⃣ 定义训练参数 TrainingArguments
# ==========================================================
# 这是 HuggingFace Trainer 的核心配置项
training_args = TrainingArguments(
    output_dir="./results",          # 模型、日志等输出目录
    eval_strategy="epoch",           # 每个 epoch 结束后进行一次验证（4.46+ 版本用 eval_strategy）
    save_strategy="epoch",           # 每个 epoch 保存一次模型
    learning_rate=2e-5,              # 微调时的学习率（BERT 通常在 1e-5 ~ 5e-5 之间）
    per_device_train_batch_size=8,   # 每张 GPU 上的 batch 大小（可按显存调节）
    per_device_eval_batch_size=8,
    num_train_epochs=3,              # 训练轮数
    weight_decay=0.01,               # 权重衰减（防止过拟合）
    logging_dir="./logs",            # 日志输出路径
    logging_steps=100,               # 每隔多少步打印一次日志
    load_best_model_at_end=True,     # 训练结束后自动加载最优模型（根据验证集指标）
    metric_for_best_model="accuracy",# 以 accuracy 作为最优模型判断标准
    report_to="none",                # 不上传到 wandb、tensorboard 等
)

# ==========================================================
# 7️⃣ 初始化 Trainer
# ==========================================================
# Trainer 封装了训练循环、评估与保存逻辑
trainer = Trainer(
    model=model,                             # 模型
    args=training_args,                      # 训练参数
    train_dataset=tokenized_datasets["train"], # 训练集
    eval_dataset=tokenized_datasets["test"],   # 验证集
    tokenizer=tokenizer,                     # 分词器（用于动态 padding）
    compute_metrics=compute_metrics,         # 指标计算函数
)

# ==========================================================
# 8️⃣ 开始训练与验证
# ==========================================================
# 训练过程会自动输出 loss、accuracy 等信息
trainer.train()

# 训练结束后在测试集上评估模型表现
eval_res = trainer.evaluate()
print("Eval Results:", eval_res)

# ==========================================================
# 9️⃣ 保存最优模型与分词器
# ==========================================================
# 将模型和分词器保存到本地 ./results/best 目录
trainer.save_model("./results/best")
tokenizer.save_pretrained("./results/best")

# ==========================================================
# 🔟 简单推理示例
# ==========================================================
# 随便输入两条评论测试模型效果
texts = [
    "The food was amazing and the service was excellent!",
    "Terrible experience. I will never come back.",
]

# 将文本编码为模型输入张量
enc = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",   # 返回 PyTorch 张量
)

import torch
# 关闭梯度计算，仅推理
with torch.no_grad():
    out = model(**enc)

# 获取预测类别（取最大概率所在的索引）
preds = out.logits.argmax(dim=-1).tolist()

# Yelp 标签是 0~4，我们显示为 1~5 星更直观
print("Predictions (1-5 stars):", [p + 1 for p in preds])

```



## 做了什么微调工作



| 阶段                      | 模型在做什么                           | 训练数据                                        | 目的                                 |
| ------------------------- | -------------------------------------- | ----------------------------------------------- | ------------------------------------ |
| **预训练 (pre-training)** | 让模型理解语言的基本规律               | 大规模无标签文本（例如 Wikipedia, BooksCorpus） | 学习通用语言知识                     |
| **微调 (fine-tuning)**    | 让模型适应某个具体任务（比如情感分类） | 少量**带标签**的数据（比如 Yelp 评论 + 星级）   | 把通用语言知识转化成可执行的任务能力 |





## 从代码角度看“微调”到底发生了什么

1. **加载一个已经训练好的语言模型**

   ```
   model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
   ```

   这一步不是随机初始化参数，而是从 Hugging Face 下载 BERT 的权重（通常是在 Wikipedia 上预训练几百亿 tokens 得到的）。
    所以它已经“懂英语”，知道语法、上下文、常见搭配等。

2. **加载你的任务数据（Yelp）**
    这是一个监督任务：输入是评论文本，输出是 1~5 星标签。

   ```
   ⭐️ Label: 4
   📝 Text: My wife took me here on my birthday for breakfast and it was excellent. The food was tasty and the service was fast. I definitely recommend this place if you like great breakfast and friendly staff.
   
   ⭐️ Label: 0
   📝 Text: I ordered a small cheese pizza and it came burnt. The crust was dry and the cheese tasted old. Never coming back here again.
   
   ⭐️ Label: 2
   📝 Text: The food was okay, not bad but nothing special. Service could have been faster. Might come back if I’m in the area.
   ```

3. **再训练几轮（num_train_epochs=3）**
    这时，模型不是从零学语言，而是在保留通用能力的基础上：

   - 最后几层权重被更新，适应“情感评分”这一具体任务；
   - 早期层（语言特征）大多保持不变，只做细微调整。

   所以这个阶段叫“fine-tuning”而不是“training from scratch”。

   

   从模型的角度看：

   ```
   一个 BERT 分类模型通常长这样：
   
   [Embedding 层]
   [Transformer 编码层 ×12]
   [分类层：一个线性层 + softmax 输出5类]
   
   微调时：
   
   1.Embedding 和前面的 Transformer 层
      这些层已经能把语言变成有意义的向量表示。
      训练时只会轻微调整（学习率小、变化慢）。
   
   2. 最后的分类层
      这层是新加的（针对 Yelp 任务）。
      它从 Transformer 的输出中学习“这句话是几星”。
      所以这部分训练变化最大。
   
   ➡️ 也就是说：
   
   模型前面的部分（理解语言的能力）保持不变；
   只调整最后几层来让它“学会打分”。
   
   这就叫 fine-tuning（微调），而不是 training from scratch（从零开始训练）。
   ```

   

4. **效果：**

   - 微调后模型能区分正面/负面评论；
   - 如果换成 IMDb 或 Amazon Review 数据再训练，它还能继续适应。





