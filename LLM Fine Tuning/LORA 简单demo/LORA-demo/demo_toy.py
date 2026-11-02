"""玩具 LoRA：秒级跑完，看清两件事。

【1】秩不够 → 再怎么训也拟合不了目标 ΔW
【2】适配器可关 → 旧行为还在；全量微调会改掉 W0，关不掉

跑:  /opt/anaconda3/bin/python3 demo_toy.py
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from lora import (
    LoRALinear,
    freeze_non_lora,
    inject_lora,
    parameter_counts,
    self_check,
    set_lora_enabled,
)
from model import TinyGPT

# 词表很小，任务就是：hi <名字> → hello/ahoy <名字>
BOS, HI, HELLO, AHOY, EOS = 1, 3, 4, 5, 2
NAMES = ["ada", "bob", "cy", "dan", "eve", "fay", "gus", "hal"]
WORDS = ["<pad>", "<bos>", "<eos>", "hi", "hello", "ahoy", *NAMES]
NAME_IDS = list(range(6, 14))


def corpus(greeting: int) -> torch.Tensor:
    # 每条: bos hi name | greeting name eos
    return torch.tensor([[BOS, HI, n, greeting, n, EOS] for n in NAME_IDS])


def loss_fn(model: TinyGPT, data: torch.Tensor) -> torch.Tensor:
    logits = model(data[:, :-1])
    labels = data[:, 1:].clone()
    labels[:, :2] = -100  # 关键：mask 掉 prompt，只在回答段算 loss（指令微调同款）
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def train(model: TinyGPT, data: torch.Tensor, steps: int, lr: float) -> None:
    # 只优化 requires_grad=True 的参数（LoRA 时就只有 A、B）
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for _ in range(steps):
        loss = loss_fn(model, data)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def reply(model: TinyGPT, name_id: int) -> str:
    """贪心解码 3 个 token：greeting + name + eos。"""
    model.eval()
    idx = torch.tensor([[BOS, HI, name_id]])
    for _ in range(3):
        nxt = model(idx)[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return " ".join(WORDS[t] for t in idx[0, 3:].tolist())


def part1_rank() -> None:
    """用不同 r 去拟合一个「已知真实秩=4」的 ΔW。"""
    print("【1】秩不够就装不下")
    torch.manual_seed(0)
    d, true_r = 32, 4
    w0 = torch.randn(d, d) / d**0.5
    # 人工构造秩恰好为 true_r 的矩阵：B(d,r) @ A(r,d)
    delta = (torch.randn(d, true_r) / true_r**0.5) @ (torch.randn(true_r, d) / d**0.5)
    print(f"目标 ΔW 真实秩={true_r}。LoRA 的秩最多是 r。\n")
    print(f"{'r':>4}  {'相对误差':>10}  说明")
    for r in (1, 2, 4, 8):
        torch.manual_seed(r)
        layer = LoRALinear(d, d, r=r, alpha=float(r))  # α=r ⇒ scaling=1，只看秩
        with torch.no_grad():
            layer.weight.copy_(w0)
        opt = torch.optim.Adam([layer.lora_A, layer.lora_B], lr=0.02)
        target = w0 + delta
        for _ in range(800):
            x = torch.randn(256, d)
            loss = F.mse_loss(layer(x), F.linear(x, target))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        err = (torch.norm(layer.delta_weight() - delta) / torch.norm(delta)).item()
        note = "秩不够" if r < true_r else "拟合成功"
        print(f"{r:4d}  {err:10.2e}  {note}")
    print()


def part2_adapter() -> None:
    """同一基座：全量微调 vs LoRA，对比「关掉后还能否回到 hello」。"""
    print("【2】适配器可关，全量微调不能")
    hello, ahoy = corpus(HELLO), corpus(AHOY)

    # --- 预训练：只学 hello ---
    torch.manual_seed(0)
    base = TinyGPT(len(WORDS), d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=16)
    for p in base.parameters():
        p.requires_grad = True
    train(base, hello, 300, 3e-3)
    ckpt = copy.deepcopy(base.state_dict())  # 冻结一份基座权重，后面两种微调都从这里出发
    print(f"基座学会: hi ada => {reply(base, NAME_IDS[0])}")

    # --- 全量微调：所有参数都更新，W0 被改掉 ---
    full = TinyGPT(len(WORDS), d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=16)
    full.load_state_dict(ckpt)
    for p in full.parameters():
        p.requires_grad = True
    train(full, ahoy, 200, 3e-3)

    # --- LoRA：只训 A/B，W0 仍是 ckpt ---
    torch.manual_seed(4)
    lora = TinyGPT(len(WORDS), d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=16)
    lora.load_state_dict(ckpt)
    inject_lora(lora, r=4, alpha=4, target_names=["q_proj", "v_proj"])
    freeze_non_lora(lora)
    _, n_train = parameter_counts(lora)
    train(lora, ahoy, 200, 5e-3)

    # 关键对比：同一模型，只拨开关
    set_lora_enabled(lora, False)
    off = reply(lora, NAME_IDS[0])  # 应回到 hello
    set_lora_enabled(lora, True)
    on = reply(lora, NAME_IDS[0])  # 应是 ahoy

    print()
    print(f"{'方法':<12} {'可训练':>8}  {'开着时':<22} {'关掉后'}")
    print("-" * 60)
    print(f"{'全量微调':<10} {sum(p.numel() for p in full.parameters()):>8}  {reply(full, NAME_IDS[0]):<22} 无法关闭")
    print(f"{'LoRA r=4':<10} {n_train:>8}  {on:<22} {off}")
    print()
    print("同一句 hi ada：LoRA 开=新任务，关=旧任务。W0 没被改掉。")


if __name__ == "__main__":
    self_check()  # 先确认 lora.py 公式没写错
    part1_rank()
    part2_adapter()
