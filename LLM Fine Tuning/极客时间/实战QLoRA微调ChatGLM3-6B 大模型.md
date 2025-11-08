# 实战QLoRA微调ChatGLM3-6B 大模型



## 重点说明内容

用 QLoRA 论文中介绍的量化技术：**NF4 数据类型、双量化和混合精度计算**，
 在 **ChatGLM3-6B** 模型上实现 QLoRA 微调。

------

##### 📦 数据准备

- **下载数据集**
- **设计 Tokenizer 函数** 处理样本（`map`、`shuffle`、`flatten`）
- **自定义批量数据处理类** `DataCollatorForChatGLM`

------

##### 🧠 训练模型

- **加载 ChatGLM3-6B 量化模型**
- **PEFT 量化模型预处理** `prepare_model_for_kbit_training`
- **配置 QLoRA 适配器**
   `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`
- **设置微调训练超参数** `TrainingArguments`
- **启动训练** `trainer.train()`
- **保存 QLoRA 模型** `trainer.model.save_pretrained()`

------

##### 🔍 模型推理

- **加载 ChatGLM3-6B 基础模型**
- **加载 ChatGLM3-6B QLoRA 模型（PEFT Adapter）**
- **对比微调前后的生成结果**





## 代码

```python
# =========================================================
# 0️⃣ 安装依赖
# =========================================================
# QLoRA 依赖 peft、transformers、bitsandbytes、datasets、accelerate
# 这里建议使用 Transformers ≥ 4.46
# =========================================================
# !pip install -U "transformers>=4.46" "peft>=0.10" "bitsandbytes" "accelerate" "datasets"

# =========================================================
# 1️⃣ 导入模块
# =========================================================
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig

# =========================================================
# 2️⃣ 配置模型加载参数（使用 4bit 量化）
# =========================================================
model_name = "THUDM/chatglm3-6b"

# bitsandbytes 量化配置：NF4 数据类型 + 双量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 使用 4-bit 量化加载
    bnb_4bit_quant_type="nf4",         # NormalFloat4 数据格式
    bnb_4bit_use_double_quant=True,    # 双量化，进一步减小显存占用
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度使用 bfloat16
)

# =========================================================
# 3️⃣ 加载 ChatGLM3-6B 基础模型与分词器
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True, #允许Transformers执行该模型仓库中上传的自定义Python代码
)

# 预处理：让模型适配 k-bit（4bit）训练模式
model = prepare_model_for_kbit_training(model)

# =========================================================
# 4️⃣ 配置 LoRA 适配器（QLoRA）
# =========================================================
# 只在注意力层添加低秩矩阵 LoRA，以节省参数
lora_config = LoraConfig(
    r=8,                          # LoRA 秩（低维子空间大小）
    lora_alpha=32,                # LoRA 缩放因子
    target_modules=[
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
    ],                            # ChatGLM3 的注意力与 MLP 层名称
    lora_dropout=0.05,            # 防止过拟合
    bias="none",
    task_type="CAUSAL_LM"         # 因果语言模型任务
)

# 将 LoRA 模块注入模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================================================
# 5️⃣ 加载微调数据集（AdvertiseGen）
# =========================================================
# 官方链接：https://huggingface.co/datasets/HasturOfficial/adgen
dataset = load_dataset("HasturOfficial/adgen", split="train[:1%]")  # 可先用 1% 做演示

# 数据示例：
# {"content": "促销活动：买一送一，限时优惠。", "summary": "商场促销活动广告"}
def format_sample(example):
    text = f"用户: {example['content']}\n广告: {example['summary']}"
    return {"text": text}

dataset = dataset.map(format_sample)

# =========================================================
# 6️⃣ Tokenize 编码
# =========================================================
def tokenize(batch):
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

# =========================================================
# 7️⃣ Data Collator（组 batch）
# =========================================================
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# =========================================================
# 8️⃣ 训练参数配置
# =========================================================
training_args = TrainingArguments(
    output_dir="./chatglm3_qlora_output",
    per_device_train_batch_size=1,   #每个 GPU 的 batch 大小
    gradient_accumulation_steps=4,   #梯度累积步数，等价于“虚拟 batch = 1×4 = 4”
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_strategy="steps",
    optim="paged_adamw_8bit",       # bitsandbytes 优化器
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
)

# =========================================================
# 9️⃣ 创建 Trainer 并开始微调
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=collator,
)
trainer.train()

# =========================================================
# 🔟 保存 LoRA 适配器
# =========================================================
model.save_pretrained("./chatglm3_qlora_adapter")
tokenizer.save_pretrained("./chatglm3_qlora_adapter")

print("✅ LoRA adapter 已保存至 ./chatglm3_qlora_adapter")

# =========================================================
# 🔁 推理测试（加载适配器）
# =========================================================
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
lora_model = PeftModel.from_pretrained(base_model, "./chatglm3_qlora_adapter")

prompt = "用户：写一个关于智能手表的广告文案。\n广告："
inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
outputs = lora_model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```



## 关键内容解释

### 什么是 NF4（NormalFloat4）

**NF4** 全称是 **Normal Float 4-bit**， 是 QLoRA（Quantized LoRA）论文提出的一种 **4位量化数据格式**，专门为大语言模型（LLM）量化优化的 **非均匀分布浮点数表示方法**。NF4 是一种统计分布感知的 4-bit 浮点格式，通过模拟正态分布（normal distribution）的权重分布来设计量化映射，能在 4bit 精度下接近 16bit 的表现。

在 LLM 量化中，我们要把模型权重从 16-bit（FP16/BF16）压缩成 4-bit。传统做法是：**线性量化（Linear Quantization）**：把最小值到最大值线性映射到 16 个等级。但模型权重往往 **不是均匀分布的** ，而是 **接近正态分布（Normal Distribution）**线性量化 → 把大部分权重“挤”在中间，很容易丢精度。NF4 的想法是：“既然权重服从正态分布，那我就按照正态分布去设计量化映射表。”





## target_modules 怎么填

`target_modules` 告诉 PEFT：“请在模型中哪些层上插入 LoRA 模块？”。通常我们只对注意力层（Q、K、V、O 投影矩阵）或部分 MLP 层做 LoRA。因为这些层**参数量大、影响力强**，能显著调节模型输出风格。通过下面的方式查看：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
# 打印模型结构
print(model)

```

输出类似（节选）：

```python
OPTForCausalLM(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(...)
      (layers): ModuleList(
        (0-23): OPTDecoderLayer(
          (self_attn): OPTAttention(
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (k_proj): Linear(...)
            (v_proj): Linear(...)
            (out_proj): Linear(...)
          )
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        )
      )
    )
  )
)
```

可以看到：

```python
self_attn:
  q_proj, k_proj, v_proj, out_proj
```

