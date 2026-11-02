"""部署/推理最小示例（对应 demo_peft 训好的适配器）。

真实上线常见两条路：

1) 基座 + adapter 热插拔（省磁盘，可随时换/关适配器）
2) merge 进基座，导出一份完整权重（推理服务更简单，体积≈整模）

用法::

    # 路径 A：加载 PEFT 适配器直接聊
    /opt/anaconda3/bin/python3 infer_deploy.py

    # 路径 B：合并后保存完整模型，再只加载合并模型
    /opt/anaconda3/bin/python3 infer_deploy.py --merge-to models/Qwen3-0.6B-qingning-merged
    /opt/anaconda3/bin/python3 infer_deploy.py --merged-model models/Qwen3-0.6B-qingning-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise SystemExit("需要 peft：/opt/anaconda3/bin/pip install -U peft") from exc

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "models" / "Qwen3-0.6B"
DEFAULT_ADAPTER = ROOT / "adapters" / "qwen3-0.6b-qingning-peft"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def apply_template(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.no_grad()
def chat(model, tokenizer, question: str, device: torch.device, max_new_tokens: int) -> str:
    model.eval()
    prompt = apply_template(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text


def load_base(model_id: str, dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    return tokenizer, model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 部署推理示例")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="基座模型目录")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help="PEFT 适配器目录")
    parser.add_argument(
        "--merged-model",
        type=Path,
        default=None,
        help="若指定：只加载已 merge 的完整模型（不再读 adapter）",
    )
    parser.add_argument(
        "--merge-to",
        type=Path,
        default=None,
        help="把基座+adapter 合并导出到该目录后退出",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--question", type=str, default="青柠岛的首都叫什么？")
    args = parser.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"设备: {device}  dtype: {dtype}")

    # ---------- 路径 B：直接用合并后的整模 ----------
    if args.merged_model is not None:
        print(f"加载合并模型: {args.merged_model}")
        tokenizer, model = load_base(str(args.merged_model), dtype, device)
        print("Q:", args.question)
        print("A:", chat(model, tokenizer, args.question, device, args.max_new_tokens))
        return

    # ---------- 路径 A：基座 + PEFT adapter ----------
    if not args.adapter.exists():
        raise SystemExit(
            f"找不到适配器 {args.adapter}\n"
            "请先跑完: /opt/anaconda3/bin/python3 demo_peft.py --steps 400 --lr 1e-4"
        )

    print(f"加载基座: {args.base}")
    tokenizer, base = load_base(str(args.base), dtype, device)
    print(f"挂载适配器: {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.to(device)

    if args.merge_to is not None:
        # 部署常用：merge 后变成普通 CausalLM，vLLM/TGI/自建服务都更好接
        print(f"合并 LoRA 到基座并保存: {args.merge_to}")
        merged = model.merge_and_unload()
        args.merge_to.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(args.merge_to))
        tokenizer.save_pretrained(str(args.merge_to))
        print("完成。之后可用:")
        print(f"  /opt/anaconda3/bin/python3 infer_deploy.py --merged-model {args.merge_to}")
        return

    print("Q:", args.question)
    print("A:", chat(model, tokenizer, args.question, device, args.max_new_tokens))
    print()
    print("交互（空行退出）:")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print(chat(model, tokenizer, q, device, args.max_new_tokens))


if __name__ == "__main__":
    main()
