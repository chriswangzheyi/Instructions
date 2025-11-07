# HF PEFT 调优工具



## LORA Adapter 配置 Demo



##### ① 导入模块

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
```

------

##### ② 创建 LoRA 配置对象

```
config = LoraConfig(
    r=8,                      # LoRA 矩阵的秩（秩越大，可训练参数越多）
    lora_alpha=32,            # LoRA 缩放因子，影响更新幅度
    target_modules=["c_attn", "c_proj"],  # GPT-2 的注意力层模块名称
    lora_dropout=0.05,        # Dropout，防止过拟合
    bias="none",              # 不更新 bias 参数
    task_type="CAUSAL_LM"     # 任务类型，这里是因果语言模型（GPT类）
)
```

------

##### ③ 让模型支持 LoRA

```
# 加载基础模型（例如 GPT-2）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 应用 LoRA 配置
model = get_peft_model(model, config)
```

------

##### ④ 打印可训练参数

```
model.print_trainable_parameters()
```

输出结果示例：

```
trainable params: 811,008 || all params: 125,250,816 || trainable%: 0.6475
```



| 项目                 | 含义                                                         |
| -------------------- | ------------------------------------------------------------ |
| **trainable params** | 当前可训练参数数量（LoRA 新增的部分）= **811,008 个**        |
| **all params**       | 模型全部参数数量（GPT-2 原始 + LoRA 插入）= **125,250,816 个** |
| **trainable%**       | 可训练参数占比 = **约 0.65%**                                |



## **实战** LoRA - OPT-6.7B **文本生成**



```python
# -*- coding: utf-8 -*-
# =========================================================
# 0) 安装依赖（Notebook 第一格执行一次即可）
# =========================================================
# !pip install -U "transformers>=4.46" peft accelerate bitsandbytes datasets




# =========================================================
# 1) 基础导入
# =========================================================
import os
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,  # 8bit/4bit 量化参数训练前的必要准备
)

# 梯子
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10887'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10887'

# 避免 CUDA 警告，确定设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

use_8bit = torch.cuda.is_available()
print("device:", device)

# =========================================================
# 2) 模型与分词器（8-bit 量化加载）
# =========================================================
model_id = "facebook/opt-6.7b"

# 加载分词器
# OPT 系列兼容 GPT2 tokenizer；用 AutoTokenizer 更通用
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# Causal LM 需要定义 pad_token，一般复用 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 以 8-bit 量化加载模型（显著降低显存占用）
# device_map="auto" 让 accelerate 自动把模型放到 GPU
if use_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,              # ← 关键：8-bit 量化加载
        device_map="auto",
    )

    # 把量化模型做一次“k-bit 训练准备”
    # 会开启输入梯度、禁用某些层的缓存等，让 8-bit/4-bit 下能稳定训练
    model = prepare_model_for_kbit_training(model)
else:
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model.to(device)

# =========================================================
# 3) 配置 LoRA（只在注意力/MLP的关键投影层插入 LoRA）
# =========================================================
# 对 OPT：注意力层通常包含 q_proj/k_proj/v_proj/out_proj
# MLP 层包含 fc1/fc2（可选，先从注意力做起也可以）
lora_config = LoraConfig(
    r=8,                      # LoRA 低秩维度（4~16 常用）
    lora_alpha=32,            # LoRA 缩放因子
    lora_dropout=0.05,        # 防过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # 先量化注意力部分
    bias="none",
    task_type="CAUSAL_LM",    # 任务：自回归语言建模
)

# 将 LoRA 适配器挂到模型上（冻结原模型，仅训练 LoRA）
model = get_peft_model(model, lora_config)

# 打印可训练参数占比（通常 <1%）
model.print_trainable_parameters()

# =========================================================
# 4) 构造一个简单演示数据集（也可以换成 wikitext-2 或你的私有语料）  
# 让模型学习输入输出的“格式”与“语气”，测试 LoRA 微调流程是否通畅
# =========================================================
texts = [
    "User: Hello, can you write a short poem about the ocean?\nAssistant:",
    "User: Explain what LoRA is in one sentence.\nAssistant:",
    "User: Give me three fun facts about dolphins.\nAssistant:",
    "User: Summarize why quantization helps large language models run faster.\nAssistant:",
]

raw_ds = Dataset.from_dict({"text": texts})

# 简单的编码函数（按最大长度截断/补齐）
# 把原始文本样本转换成模型可以训练的输入格式
max_length = 512
def tokenize(example):
    # 在纯 Causal LM 训练里，输入=标签；这里演示目的直接映射
    enc = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_ds = raw_ds.map(tokenize, remove_columns=["text"])

# DataCollator：按 Causal LM 组 batch；这里已 pad 成固定长度，collator 简单化
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =========================================================
# 5) 训练参数
# =========================================================
# 小数据演示：用 steps 评估/保存，fp16 打开混合精度，使用 bitsandbytes 的 8-bit 优化器
training_args = TrainingArguments(
    output_dir="./opt67b_lora_8bit_demo",
    eval_strategy="steps",           # 每隔多少步评估一次模型（这里按 step，而不是 epoch）
    eval_steps=20,                   # 每训练 20 步评估一次
    save_strategy="steps",           # 每隔固定步数保存一次模型
    save_steps=20,
    logging_steps=10,                # 每 10 步打印一次日志（loss、学习率等）

    num_train_epochs=1,              # 整个数据集完整训练1次
    per_device_train_batch_size=1,   # 每个 GPU（设备）每次迭代送入 1 个样本
    per_device_eval_batch_size=1,    # 评估阶段同样一次处理 1 个样本
    gradient_accumulation_steps=8,   # 每 8 次前向计算累积一次梯度，相当于 batch=8 的效果
    learning_rate=2e-4,              # 初始学习率（LoRA 常用 1e-4 ~ 3e-4）
    weight_decay=0.0,                # 权重衰减（防止过拟合），此处禁用

    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
    fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,

    optim="paged_adamw_8bit" if use_8bit else "adamw_torch",        # bitsandbytes 的 8-bit 优化器，省显存
    lr_scheduler_type="cosine",     # 学习率调度策略：余弦退火
    warmup_ratio=0.03,              # 预热阶段比例（前 3% 的 steps 逐步升高学习率，防止初期震荡）

    gradient_checkpointing=True,     # 激活检查点，进一步省显存
    ddp_find_unused_parameters=False,

    report_to="none",
    load_best_model_at_end=False,    # 演示用，关闭
)

# =========================================================
# 6) 训练（LoRA 适配器参数）
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    eval_dataset=tokenized_ds.select(range(2)),   # 演示：拿前两条当 eval
    data_collator=collator,
)

trainer.train()

# 保存 LoRA 适配器与分词器（小体积）
adapter_dir = "./opt67b_lora_8bit_demo/adapter"
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("LoRA adapter saved to:", adapter_dir)

# =========================================================
# 7) 简单推理测试（合并 LoRA 进行生成）
# =========================================================
model.eval()

def chat(prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,                        # 开启随机采样，而非贪心取最大概率
            temperature=0.8,
            top_p=0.9,                             # nucleus sampling，仅保留累计概率前 90% 的词进行采样
            repetition_penalty=1.1,                # 惩罚重复词，防止模型啰嗦
            pad_token_id=tokenizer.eos_token_id,   # 指定填充 token，防止警告或错位
        )
    return tokenizer.decode(gen[0], skip_special_tokens=True)

test_prompt = "User: Explain LoRA in one sentence.\nAssistant:"
print(chat(test_prompt))
```



文字描述流程：



```
## 🧩 一、环境准备
安装必要依赖并检测设备环境：

- `transformers`
- `peft`
- `bitsandbytes`
- `datasets`
- 检查是否可用 GPU（CUDA）

---

## 🧠 二、加载模型与分词器
- 选择模型：`facebook/opt-6.7b`
- 使用 `AutoTokenizer` 加载分词器，并补齐 `pad_token`
- 用 8-bit 量化方式加载模型（`load_in_8bit=True`），节省显存
- 使用 `prepare_model_for_kbit_training()` 做量化训练准备

---

## ⚙️ 三、插入 LoRA 结构
- 创建 `LoraConfig`（设置秩 `r=8`、缩放系数 `lora_alpha=32`、目标层如 `q_proj`、`v_proj`）
- 调用 `get_peft_model()` 将 LoRA 模块挂到模型指定层
- 冻结原模型参数，仅训练 LoRA 插入部分（约占 <1% 参数量）

---

## 📚 四、构造训练数据
- 自定义几条示例文本（如对话或指令形式）
- 使用分词器将文本转为 `input_ids` 与 `labels`
- 创建小型 `Dataset` 用于演示

---

## 🧩 五、配置训练参数
- 使用 `TrainingArguments` 设置训练细节：
  - `eval_strategy`、`save_strategy`
  - 学习率、批大小、梯度累积步数
  - 使用 `paged_adamw_8bit` 优化器节省显存
  - 开启 `fp16` 或 `bf16` 混合精度
  - 启用 `gradient_checkpointing` 进一步降低内存消耗

---

## 🚀 六、训练 LoRA 参数
- 使用 `Trainer` 执行 `.train()`，仅更新 LoRA 层参数
- 保存训练后的 **LoRA 适配器** 与分词器

---

## 💬 七、推理测试
- 切换到推理模式 `model.eval()`
- 输入测试提示（prompt）
- 调用 `generate()` 生成回答
- 查看模型在微调后的文本生成效果

```



## LoRA实战- OpenAl Whisper-large-v2



```python
# =========================================================
# 0) 环境检测 & 依赖导入
# =========================================================
import os
import torch
from datasets import load_dataset, DatasetDict, Audio

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

print("Torch:", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# =========================================================
# 1) 全局参数
# =========================================================
model_name_or_path = "openai/whisper-large-v2"   # Whisper 基座
language_abbr = "zh-CN"                          # Common Voice 语言代码
task = "transcribe"                              # 语音转文字
dataset_name = "mozilla-foundation/common_voice_11_0"

# 训练超参（示例用较小设置，跑通流程）
num_train_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
learning_rate = 1e-4
warmup_steps = 200
output_dir = "./whisper_lora_zh"
logging_steps = 50

# 是否抽样子集（示例用 True；正式训练请改为 False）
use_small_subset = True
train_take = 500    # 训练子集条数
eval_take = 50      # 验证子集条数

# =========================================================
# 2) 加载数据集（Common Voice 11.0 zh-CN）
#    Whisper 期望音频为 16k 采样率，字段为 "audio"，转录文本字段为 "sentence"
# =========================================================
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation")
common_voice["test"]  = load_dataset(dataset_name, language_abbr, split="test")

# 统一为 16k 采样率
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

if use_small_subset:
    common_voice["train"] = common_voice["train"].select(range(min(train_take, len(common_voice["train"]))))
    common_voice["test"]  = common_voice["test"].select(range(min(eval_take, len(common_voice["test"]))))

print(common_voice)

# =========================================================
# 3) 处理器（含特征提取 + 分词器）
#    Whisper 的 AutoProcessor 同时提供：
#    - feature_extractor：将波形 → log-Mel 频谱
#    - tokenizer：将文本 → token ids
# =========================================================
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language_abbr, task=task)

def prepare_batch(batch):
    # 取出音频波形与采样率
    audio = batch["audio"]
    # 1) 音频特征：80 维 log-Mel 频谱；Whisper 固定 16k
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # 2) 文本标签：转录文本 -> token ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# 映射预处理；保留必要字段
cols_to_remove = list(set(common_voice["train"].column_names) - {"audio", "sentence"})
cv_proc = DatasetDict()
cv_proc["train"] = common_voice["train"].map(prepare_batch, remove_columns=cols_to_remove, num_proc=1)
cv_proc["eval"]  = common_voice["test"].map(prepare_batch,  remove_columns=cols_to_remove, num_proc=1)

print(cv_proc["train"][0].keys())  # 应包含：input_features, labels

# =========================================================
# 4) 加载 Whisper 模型
#    - 有 GPU：使用 bitsandbytes 8bit 量化 + LoRA 微调（省显存）
#    - 无 GPU：回退为全精度；建议只做推理或选择小模型（如 tiny/base）
# =========================================================
use_bnb = torch.cuda.is_available()  # 只有在 CUDA 下才可用 bnb 量化

if use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # 少量 CPU offload，避免显存打满
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)  # k-bit 训练必要准备
else:
    print("[提示] 未检测到 CUDA；以全精度加载模型。建议改用 openai/whisper-base / small 以节省内存。")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)
    model.to(device)

# =========================================================
# 5) 注入 LoRA 适配器（仅训练极少量参数）
#    常见做法：对注意力权重 q_proj / v_proj 注入 LoRA
# =========================================================
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Whisper 的注意力投影
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # 查看可训练参数占比

# =========================================================
# 6) DataCollator & 训练参数
#    DataCollatorForSeq2Seq 会按最长样本对 batch 做动态 padding
# =========================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,  # 用 tokenizer 做文本 padding
    model=model,
    padding=True,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    num_train_epochs=num_train_epochs,

    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,       # eval 时用 generate() 生成文本
    fp16=torch.cuda.is_available(),   # 有 CUDA 则启用半精度
    logging_steps=logging_steps,
    report_to="none",                 # 不上报到 wandb 等
)

# =========================================================
# 7) Trainer 训练
#    注意：这里的 tokenizer 传入的是 processor.tokenizer（文本侧），
#    而输入特征由 DataCollator + input_features 提供
# =========================================================
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=cv_proc["train"],
    eval_dataset=cv_proc["eval"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

trainer.train()

# 保存 LoRA 适配器与处理器
adapter_dir = os.path.join(output_dir, "adapter")
trainer.model.save_pretrained(adapter_dir)
processor.save_pretrained(adapter_dir)
print("LoRA adapter saved to:", adapter_dir)

# =========================================================
# 8) 推理解码示例
#    取一条 eval 样本：audio -> input_features -> generate -> 文本
# =========================================================
model.eval()

def transcribe_example(sample):
    # 准备输入特征
    feats = processor.feature_extractor(
        sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"]
    ).input_features
    feats = torch.tensor([feats]).to(next(model.parameters()).device)

    with torch.no_grad():
        gen_ids = model.generate(
            feats,
            max_new_tokens=128,
            do_sample=False,   # 演示用贪心解码，更稳定
        )

    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text

sample = common_voice["test"][0]
pred = transcribe_example(sample)
print("预测结果：", pred)
print("真实文本：", sample["sentence"])
```

