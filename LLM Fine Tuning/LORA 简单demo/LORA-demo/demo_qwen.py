"""真实模型版：Qwen3-0.6B + 本目录 lora.py（不依赖 peft）。

流程和 demo_toy 一样，只是模型换成 0.6B：
    加载基座 → inject_lora → freeze → 训「青柠岛」→ 关/开对比

数据: data/qingning.jsonl（虚构设定，基座本来不知道标准答案）
跑:   /opt/anaconda3/bin/python3 demo_qwen.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora import freeze_non_lora, inject_lora, iter_lora, parameter_counts, set_lora_enabled

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LOCAL = ROOT / "models" / "Qwen3-0.6B"
DEFAULT_DATA = ROOT / "data" / "qingning.jsonl"
DEFAULT_ADAPTER = ROOT / "adapters" / "qwen3-0.6b-qingning"

# 前 3 条在训练集里；最后 1 条不在，用来粗看泛化
PROBE_QUESTIONS = [
    "青柠岛的首都叫什么？",
    "青柠岛最有名的特产是什么？",
    "青柠岛的官方问候语是什么？",
    "用一句话介绍青柠岛。",
]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built():
        try:
            x = torch.zeros(1, device="mps")
            del x
            return torch.device("mps")
        except Exception:
            pass  # 新 macOS 上 is_available 有时误报
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # 关键：MPS + fp16 训练易 nan（输出全变成 !!!!）；Mac 上用 fp32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

##决定从哪加载基座模型，如果有本地模型，则使用本地模型，否则使用远程模型。
def resolve_model_id(model: str, local_dir: Path) -> str:
    has_weights = local_dir.exists() and (
        any(local_dir.glob("*.safetensors"))
        or any(local_dir.glob("*.bin"))
        or (local_dir / "config.json").exists()
    )
    if has_weights:
        print(f"使用本地模型: {local_dir}")
        return str(local_dir)
    if Path(model).exists():
        return model
    return model

##加载训练数据
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

##转换训练数据位Qwen能识别的格式
def apply_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    """Qwen chat template；关掉 thinking，避免回答前面拖一长串推理。"""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)

## 把一问一答，做成一条能拿去训练的样本
def encode_example(tokenizer, user: str, assistant: str, device: torch.device) -> dict[str, torch.Tensor]:
    """拼一条 SFT 样本：prompt 段 label=-100，只在 assistant 回答上算 loss。"""
    # 只有「问题 + 请开始回答」
    prompt_text = apply_template(
        tokenizer,
        [{"role": "user", "content": user}],
        add_generation_prompt=True,
    )
    # 「问题 + 完整回答」
    full_text = apply_template(
        tokenizer,
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]  # 问题有多长
    full = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    input_ids = full["input_ids"].to(device)  # 整句（问题+回答）的 token
    labels = input_ids.clone()
    cut = min(len(prompt_ids), labels.size(1))
    labels[:, :cut] = -100  # 关键：和 toy demo 的 labels[:, :2]=-100 同一思想
    if labels.size(1) > 0 and int((labels != -100).sum()) == 0:
        raise RuntimeError("label mask 后没有可训练 token，检查 chat template")
    return {"input_ids": input_ids, "labels": labels, "attention_mask": full["attention_mask"].to(device)}

# 给模型一个问题，让它生成回答（不更新权重）
@torch.no_grad() #关掉梯度、进入评估模式 → 只推理，不训练
def generate_answer(model, tokenizer, question: str, device: torch.device, max_new_tokens: int) -> str:
    model.eval()
    prompt = apply_template(
        tokenizer,
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 贪心，方便对比关/开差异
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text

#段是对比展示：同一个问题，分别看「关适配器」和「开适配器」的回答。
def print_compare(model, tokenizer, device: torch.device, questions: list[str], max_new_tokens: int) -> None:
    """直观看差异：关=基座，开=LoRA。"""
    print()
    print("=" * 72)
    print("对照：同一问题，关适配器 = 基座；开适配器 = LoRA 后")
    print("=" * 72)
    for question in questions:
        set_lora_enabled(model, False)
        off = generate_answer(model, tokenizer, question, device, max_new_tokens)
        set_lora_enabled(model, True)
        on = generate_answer(model, tokenizer, question, device, max_new_tokens)
        print()
        print(f"Q: {question}")
        print(f"  关: {off}")
        print(f"  开: {on}")
        print(f"  变化: {'是' if off.strip() != on.strip() else '否'}")


def save_adapter(model, path: Path, meta: dict) -> None:
    """只存 A/B（几 MB），不存 0.6B 基座。"""
    path.mkdir(parents=True, exist_ok=True)
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save({"state": state, "meta": meta}, path / "adapter.pt")
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"适配器已保存: {path / 'adapter.pt'}")


def load_adapter(model, path: Path) -> dict:
    blob = torch.load(path / "adapter.pt", map_location="cpu", weights_only=False)
    missing = model.load_state_dict(blob["state"], strict=False)
    unexpected = [k for k in missing.unexpected_keys if "lora_" in k]
    if unexpected:
        raise RuntimeError(f"适配器键对不上: {unexpected[:5]}")
    return blob.get("meta", {})


def train(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    device: torch.device,
    steps: int,
    lr: float,
) -> bool:
    examples = [encode_example(tokenizer, row["user"], row["assistant"], device) for row in rows]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr)
    model.train()
    set_lora_enabled(model, True)

    t0 = time.time()
    for step in range(1, steps + 1):
        batch = examples[(step - 1) % len(examples)]
        out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"  step {step:4d}/{steps}  loss 非有限值，停止训练（常见于 fp16+MPS）")
            return False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{steps}  loss {loss.item():.4f}  ({elapsed:.0f}s)")
    return True


def interactive_loop(model, tokenizer, device: torch.device, max_new_tokens: int) -> None:
    print()
    print("交互对比。输入问题；空行退出；命令 /off /on 只看一侧。")
    mode = "both"
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            break
        if raw == "/off":
            mode = "off"
            print("只显示关适配器")
            continue
        if raw == "/on":
            mode = "on"
            print("只显示开适配器")
            continue
        if raw == "/both":
            mode = "both"
            print("两侧都显示")
            continue
        if mode in {"both", "off"}:
            set_lora_enabled(model, False)
            print("  关:", generate_answer(model, tokenizer, raw, device, max_new_tokens))
        if mode in {"both", "on"}:
            set_lora_enabled(model, True)
            print("  开:", generate_answer(model, tokenizer, raw, device, max_new_tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-0.6B LoRA 微调与开关对比")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--rank", type=int, default=8)  # 对应 LoraConfig.r
    parser.add_argument("--alpha", type=float, default=16)  # 对应 lora_alpha
    parser.add_argument(
        "--targets",
        default="q_proj,v_proj",  # 对应 target_modules
        help="逗号分隔。完整可试 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-train", action="store_true", help="只加载已有适配器做对比")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    model_id = resolve_model_id(args.model, args.local_dir)

    print(f"设备: {device}  dtype: {dtype}")
    if device.type == "cpu":
        print("当前是 CPU。0.6B + 60 step 大概要十几分钟；有 GPU 会快很多。")
    print(f"模型: {model_id}")
    print(f"LoRA: r={args.rank} alpha={args.alpha:g} targets={targets}")

    # ----- 1. 加载基座 -----
    print("加载 tokenizer / 模型（首次会下载）...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    model.to(device)

    # ----- 2. 注入 LoRA（和 demo_toy 同一套 API）-----
    replaced = inject_lora(model, r=args.rank, alpha=args.alpha, target_names=targets)
    model.to(device)
    freeze_non_lora(model)
    total, trainable = parameter_counts(model)
    print(f"替换了 {len(replaced)} 个线性层")
    print(f"参数: 总共 {total:,}  可训练 {trainable:,}  ({100 * trainable / total:.3f}%)")
    for path, in_f, out_f in replaced[:4]:
        print(f"  例: {path}  {in_f}x{out_f}")
    if len(replaced) > 4:
        print(f"  ... 另外 {len(replaced) - 4} 层")

    # ----- 3. 训练或加载适配器 -----
    if args.skip_train:
        if not (args.adapter_dir / "adapter.pt").exists():
            raise SystemExit(f"找不到适配器: {args.adapter_dir / 'adapter.pt'}，先去掉 --skip-train 训练一次")
        meta = load_adapter(model, args.adapter_dir)
        print(f"已加载适配器 meta={meta}")
    else:
        rows = load_rows(args.data)
        print(f"开始训练  samples={len(rows)}  steps={args.steps}  lr={args.lr:g}")
        set_lora_enabled(model, False)
        before = generate_answer(model, tokenizer, PROBE_QUESTIONS[0], device, args.max_new_tokens)
        print(f"训练前基座样例: {PROBE_QUESTIONS[0]}")
        print(f"  => {before}")
        ok = train(model, tokenizer, rows, device, steps=args.steps, lr=args.lr)
        if not ok:
            raise SystemExit("训练失败，未保存适配器。请用当前脚本重跑（已默认 fp32 + 更小 lr）。")
        save_adapter(
            model,
            args.adapter_dir,
            {
                "base_model": model_id,
                "rank": args.rank,
                "alpha": args.alpha,
                "targets": targets,
                "steps": args.steps,
                "lr": args.lr,
                "data": str(args.data),
                "dtype": str(dtype),
            },
        )

    # ----- 4. 关/开对比（学习时盯这里）-----
    print_compare(model, tokenizer, device, PROBE_QUESTIONS, args.max_new_tokens)

    # 抽查：学到的 ΔW 秩不应超过 r（第 r+1 个奇异值 ≈ 0）
    name, layer = next(iter_lora(model))
    delta = layer.delta_weight().detach().float().cpu()
    print()
    if not torch.isfinite(delta).all():
        print(f"ΔW 含非有限值 ({name})，跳过奇异值；适配器多半已训崩。")
    else:
        singular = torch.linalg.svdvals(delta)
        preview = " ".join(f"{v.item():.3f}" for v in singular[: layer.r + 1])
        print(f"ΔW 奇异值抽查 ({name}): {preview}")
        print(f"第 {layer.r + 1} 个应接近 0。")

    if args.interactive:
        interactive_loop(model, tokenizer, device, args.max_new_tokens)
    else:
        print()
        print("想自己提问对比:")
        print("  /opt/anaconda3/bin/python3 demo_qwen.py --skip-train --interactive")


if __name__ == "__main__":
    main()
