"""玩具 decoder-only Transformer（约 7 万参数）。

层名故意对齐 LLaMA / Qwen：
    注意力: q_proj, k_proj, v_proj, o_proj
    MLP:    up_proj, down_proj

demo 里默认只给 q_proj、v_proj 挂 LoRA（和论文/PEFT 常见做法一致）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 这四个 Linear 就是 inject_lora(target_names=...) 可能替换的对象
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # 上三角 True = 未来位置，因果掩码
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels = x.shape
        q = self.q_proj(x).view(bsz, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, steps, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:steps, :steps], float("-inf"))
        weights = scores.softmax(dim=-1)
        mixed = (weights @ v).transpose(1, 2).contiguous().view(bsz, steps, channels)
        return self.o_proj(mixed)


class Block(nn.Module):
    """Pre-Norm：LN → Attention / MLP，再残差加回。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)  # 也可挂 LoRA
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.down_proj(F.gelu(self.up_proj(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 128,
        max_len: int = 16,
    ):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, max_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 词表投影，本 demo 不挂 LoRA
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (batch, seq) token id → logits (batch, seq, vocab)。"""
        steps = idx.size(1)
        if steps > self.max_len:
            raise ValueError(f"序列长度 {steps} 超过 max_len {self.max_len}")
        pos = torch.arange(steps, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))
