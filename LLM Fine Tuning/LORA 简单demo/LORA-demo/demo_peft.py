"""工程写法 Demo：Qwen3-0.6B + Hugging Face PEFT（不手写 lora.py）。

和 demo_qwen.py 做同一件事，但换成真实项目更常见的 API：

    手写 demo_qwen.py          工程 PEFT
    -------------------------  ---------------------------------
    inject_lora(...)           LoraConfig + get_peft_model(...)
    freeze_non_lora(...)       PEFT 自动只训 LoRA 参数
    save adapter.pt            model.save_pretrained(adapter_dir)
    load_adapter(...)          PeftModel.from_pretrained(...)
    set_lora_enabled(False)    with model.disable_adapter(): ...

依赖::

    /opt/anaconda3/bin/pip install -U peft

跑::

    /opt/anaconda3/bin/python3 demo_peft.py
    /opt/anaconda3/bin/python3 demo_peft.py --skip-train --interactive
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as exc:
    raise SystemExit(
        "未安装 peft。请先执行:\n"
        "  /opt/anaconda3/bin/pip install -U peft\n"
    ) from exc

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LOCAL = ROOT / "models" / "Qwen3-0.6B"
DEFAULT_DATA = ROOT / "data" / "qingning.jsonl"
DEFAULT_ADAPTER = ROOT / "adapters" / "qwen3-0.6b-qingning-peft"

PROBE_QUESTIONS = [
    "青柠岛的首都叫什么？",
    "青柠岛最有名的特产是什么？",
    "青柠岛的官方问候语是什么？",
    "用一句话介绍青柠岛。",
]


@dataclass
class TrainConfig:
    """真实项目里常把超参收成配置对象（或 yaml/json）。"""

    model_id: str
    data_path: Path
    adapter_dir: Path
    rank: int = 128
    alpha: int = 16
    targets: tuple[str, ...] = ("q_proj", "v_proj")
    dropout: float = 0.0
    steps: int = 80
    lr: float = 5e-5
    max_new_tokens: int = 64


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built():
        try:
            x = torch.zeros(1, device="mps")
            del x
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # 与 demo_qwen 相同：MPS 上 fp16 训练易 nan
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_model_id(model: str, local_dir: Path) -> str:
    has_weights = local_dir.exists() and (
        any(local_dir.glob("*.safetensors"))
        or any(local_dir.glob("*.bin"))
        or (local_dir / "config.json").exists()
    )
    if has_weights:
        print(f"使用本地模型: {local_dir}")
        return str(local_dir)
    return model if not Path(model).exists() else model


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"数据为空: {path}")
    return rows


def apply_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_example(tokenizer, user: str, assistant: str, device: torch.device) -> dict[str, torch.Tensor]:
    prompt_text = apply_template(
        tokenizer, [{"role": "user", "content": user}], add_generation_prompt=True
    )
    full_text = apply_template(
        tokenizer,
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    input_ids = full["input_ids"].to(device)
    labels = input_ids.clone()
    cut = min(len(prompt_ids), labels.size(1))
    labels[:, :cut] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full["attention_mask"].to(device),
    }


def load_base_model(model_id: str, dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    return tokenizer, model.to(device)


def build_lora_model(base_model, cfg: TrainConfig):
    """工程关键：用 LoraConfig 声明超参，get_peft_model 注入。"""
    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.targets),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, rows: list[dict[str, str]], device: torch.device, cfg: TrainConfig) -> bool:
    examples = [encode_example(tokenizer, row["user"], row["assistant"], device) for row in rows]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr)
    model.train()

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        batch = examples[(step - 1) % len(examples)]
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            print(f"  step {step:4d}/{cfg.steps}  loss 非有限值，停止")
            return False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == cfg.steps:
            print(f"  step {step:4d}/{cfg.steps}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    return True


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, device: torch.device, max_new_tokens: int) -> str:
    model.eval()
    prompt = apply_template(
        tokenizer, [{"role": "user", "content": question}], add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text


def print_compare(model, tokenizer, device: torch.device, questions: list[str], max_new_tokens: int) -> None:
    """PEFT 用 disable_adapter() 临时关掉外挂，等价于手写的 set_lora_enabled(False)。"""
    print()
    print("=" * 72)
    print("对照：同一问题，关适配器 = 基座；开适配器 = LoRA 后")
    print("=" * 72)
    for question in questions:
        with model.disable_adapter():
            off = generate_answer(model, tokenizer, question, device, max_new_tokens)
        on = generate_answer(model, tokenizer, question, device, max_new_tokens)
        print()
        print(f"Q: {question}")
        print(f"  关: {off}")
        print(f"  开: {on}")
        print(f"  变化: {'是' if off.strip() != on.strip() else '否'}")


def interactive_loop(model, tokenizer, device: torch.device, max_new_tokens: int) -> None:
    print()
    print("交互对比。输入问题；空行退出；/off /on /both")
    mode = "both"
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            break
        if raw in {"/off", "/on", "/both"}:
            mode = raw[1:]
            print(f"模式: {mode}")
            continue
        if mode in {"both", "off"}:
            with model.disable_adapter():
                print("  关:", generate_answer(model, tokenizer, raw, device, max_new_tokens))
        if mode in {"both", "on"}:
            print("  开:", generate_answer(model, tokenizer, raw, device, max_new_tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="工程写法：PEFT LoRA + Qwen3-0.6B")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--targets", default="q_proj,v_proj")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    targets = tuple(p.strip() for p in args.targets.split(",") if p.strip())
    model_id = resolve_model_id(args.model, args.local_dir)
    cfg = TrainConfig(
        model_id=model_id,
        data_path=args.data,
        adapter_dir=args.adapter_dir,
        rank=args.rank,
        alpha=args.alpha,
        targets=targets,
        steps=args.steps,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
    )

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"设备: {device}  dtype: {dtype}")
    print(f"配置: {asdict(cfg)}")

    # 1) 基座
    tokenizer, base = load_base_model(cfg.model_id, dtype, device)

    # 2) 训练：get_peft_model；推理：PeftModel.from_pretrained
    if args.skip_train:
        adapter_config = cfg.adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise SystemExit(
                f"找不到 PEFT 适配器: {cfg.adapter_dir}\n"
                "请先去掉 --skip-train 训练一次。"
            )
        print(f"加载 PEFT 适配器: {cfg.adapter_dir}")
        model = PeftModel.from_pretrained(base, str(cfg.adapter_dir))
        model.to(device)
    else:
        model = build_lora_model(base, cfg)
        rows = load_rows(cfg.data_path)
        print(f"开始训练  samples={len(rows)}  steps={cfg.steps}  lr={cfg.lr:g}")
        with model.disable_adapter():
            before = generate_answer(
                model, tokenizer, PROBE_QUESTIONS[0], device, cfg.max_new_tokens
            )
        print(f"训练前基座样例: {PROBE_QUESTIONS[0]}")
        print(f"  => {before}")

        ok = train(model, tokenizer, rows, device, cfg)
        if not ok:
            raise SystemExit("训练失败，未保存适配器。")

        # 工程标准落盘：adapter_config.json + adapter_model.safetensors
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(cfg.adapter_dir))
        tokenizer.save_pretrained(str(cfg.adapter_dir))
        (cfg.adapter_dir / "train_config.json").write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"适配器已保存: {cfg.adapter_dir}")

    # 3) 关/开对比
    print_compare(model, tokenizer, device, PROBE_QUESTIONS, cfg.max_new_tokens)

    if args.interactive:
        interactive_loop(model, tokenizer, device, cfg.max_new_tokens)
    else:
        print()
        print("交互对比:")
        print("  /opt/anaconda3/bin/python3 demo_peft.py --skip-train --interactive")


if __name__ == "__main__":
    main()
