# LoRA 精简 Demo

公式：\(h = W_0 x + \frac{\alpha}{r} BAx\)。`W0` 冻结，只训 `A`、`B`。

## 文件（只留这些）

| 文件 | 看什么 |
| --- | --- |
| `lora.py` | 心脏：`LoRALinear.forward` |
| `model.py` | 玩具小 Transformer |
| `demo_toy.py` | 秒级：秩 + 适配器开关 |
| `demo_qwen.py` | 同一套接到 Qwen3-0.6B（手写 lora.py） |
| `demo_peft.py` | **工程写法**：Hugging Face PEFT |
| `data/qingning.jsonl` | Qwen 用的虚构「青柠岛」问答（约 200 条改写样本） |
| `download_qwen.py` | 下载基座权重，与理解 LoRA 无关 |

`models/`、`adapters/` 是权重，不用读。

## 怎么跑

```bash
/opt/anaconda3/bin/python3 demo_toy.py          # 先跑这个
/opt/anaconda3/bin/python3 download_qwen.py     # 首次
/opt/anaconda3/bin/python3 demo_qwen.py
```

阅读顺序：`lora.py` → 跑 `demo_toy.py` → 可选 `demo_qwen.py`。

## 如果只想用真实 Qwen

玩具（`model.py` / `demo_toy.py`）可以不看。真模型入口是：

```bash
/opt/anaconda3/bin/python3 demo_qwen.py
/opt/anaconda3/bin/python3 demo_qwen.py --skip-train --interactive
```

## 真实项目最少要什么

核心逻辑上主要是两块：

| 文件 | 作用 |
| --- | --- |
| `lora.py` | LoRA 怎么挂、怎么训、怎么开关（教学用） |
| `demo_qwen.py` | 加载 Qwen、数据、训练、存 adapter、推理对比 |

另外还需要：训练数据（类似 `qingning.jsonl`）、基座权重、`torch` / `transformers` 等依赖。

## 工程写法（推荐对照着看）

真实项目更常见用 **PEFT**，对应本目录的 `demo_peft.py`：

```bash
/opt/anaconda3/bin/pip install -U peft
/opt/anaconda3/bin/python3 demo_peft.py
```

| 手写 `demo_qwen.py` | 工程 `demo_peft.py` |
| --- | --- |
| `inject_lora` | `LoraConfig` + `get_peft_model` |
| `freeze_non_lora` | PEFT 自动只训 LoRA |
| 自存 `adapter.pt` | `save_pretrained` → `adapter_model.safetensors` |
| `set_lora_enabled(False)` | `with model.disable_adapter()` |

数学一样；`lora.py` 帮你看懂公式，`demo_peft.py` 更接近上线代码。  
上线项目通常还要补：更大更干净的数据、评估、日志、配置、部署。

## 部署怎么做

训好的不是整模，是适配器目录（PEFT）：

```text
adapters/qwen3-0.6b-qingning-peft/
  adapter_config.json
  adapter_model.safetensors
```

两条常见路：

**A. 基座 + adapter（可热插拔）**

```bash
/opt/anaconda3/bin/python3 infer_deploy.py
```

代码里就是 `PeftModel.from_pretrained(base, adapter_dir)`。

**B. merge 成完整模型再部署（服务侧更简单）**

```bash
/opt/anaconda3/bin/python3 infer_deploy.py --merge-to models/Qwen3-0.6B-qingning-merged
/opt/anaconda3/bin/python3 infer_deploy.py --merged-model models/Qwen3-0.6B-qingning-merged
```

合并后可用普通 `AutoModelForCausalLM` 加载；接到 vLLM / TGI / 自建 FastAPI 时通常走这条。
