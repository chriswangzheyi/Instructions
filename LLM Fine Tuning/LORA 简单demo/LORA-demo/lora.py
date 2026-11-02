"""LoRA 线性层 —— 本项目心脏。

论文公式（x 为行向量，和 nn.Linear 一致）::

    y = x @ W0.T + (alpha / r) * x @ A.T @ B.T

对照本文件：
    W0     -> self.weight          （冻结）
    A      -> self.lora_A          形状 (r, in)
    B      -> self.lora_B          形状 (out, r)，初始全 0
    alpha/r-> self.scaling

B=0 ⇒ 训练开始时输出 = 原模型。
部署时可 merge：W0 ← W0 + (alpha/r) * B @ A
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """用低秩支路 BA 代替全量更新 W0。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r 必须是正整数")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # 秩：BA 最多只能表达秩 ≤ r 的增量
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r  # 论文里的 α/r；常令 α=r 使 scaling=1
        self.dropout_p = dropout
        self.enabled = True  # False 时只用 W0，用来对比「关适配器」
        self.merged = False  # True 表示 ΔW 已加进 W0

        # 关键：只有 A、B 可训练；W0 永远 requires_grad=False
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)  #requires_grad=False → 不会更新
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        self.lora_A = nn.Parameter(torch.empty(r, in_features))  # 降维
        self.lora_B = nn.Parameter(torch.empty(out_features, r))  # 升维
        self.reset_parameters()

    # 给参数增初始值
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        # 关键：B=0 ⇒ BA=0 ⇒ 初始增量是 0（和 PEFT 一致）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    #把已经有的普通线性层，改造成 LoRA 层。
    #q_proj = nn.Linear(...)   # 普通层，已有训练好的权重
    #变为：q_proj = LoRALinear(...)  # 带 A、B 的层，但 W0 必须还是原来的权重
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """把现成 nn.Linear 换成 LoRA 层，并拷贝原权重到冻结的 W0。"""
        if not isinstance(linear, nn.Linear):
            raise TypeError("只能替换 nn.Linear")
        layer = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            bias=linear.bias is not None,
            dropout=dropout,
        )
        # 新 Parameter 默认在 CPU，必须跟原层同设备/精度
        layer = layer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def delta_weight(self) -> torch.Tensor:
        """ΔW = (α/r) B A，形状 (out, in)，秩 ≤ r。"""
        return self.scaling * (self.lora_B @ self.lora_A)

    def merge(self) -> None:
        """部署用：把 ΔW 写进 W0。合并后再关 enabled 也不会回到旧模型。"""
        if self.merged:
            return
        with torch.no_grad():
            self.weight.add_(self.delta_weight())
        self.merged = True

    def unmerge(self) -> None:
        """撤销 merge，恢复「W0 与适配器分离」。"""
        if not self.merged:
            return
        with torch.no_grad():
            self.weight.sub_(self.delta_weight())
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键路径：y = W0(x) + scaling * B(A(x))
        if self.training and self.merged:
            raise RuntimeError("合并后的 LoRA 不能继续训练，请先 unmerge()")
        base = F.linear(x, self.weight, self.bias)  # 冻结的原模型
        if self.merged or not self.enabled:
            return base  # 已合并，或手动关掉适配器
        if self.dropout_p > 0 and self.training:
            x = F.dropout(x, self.dropout_p)
        # x @ A^T @ B^T  ≡  ((x @ A^T) @ B^T)
        lora_out = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return base + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"alpha={self.alpha:g}, scaling={self.scaling:g}, dropout={self.dropout_p:g}"
        )


# ---------------------------------------------------------------------------
# 下面几个函数 = 迷你版 PEFT：注入 / 冻结 / 开关
# ---------------------------------------------------------------------------

#在全模型里找指定名字的线性层，并一批换成 LoRA 层。
def inject_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    target_names: list[str] | tuple[str, ...],
    dropout: float = 0.0,
) -> list[tuple[str, int, int]]:
    """按名字替换线性层。target_names 对应 PEFT 的 target_modules，如 q_proj,v_proj。"""
    wanted = set(target_names) # 需要替换的名字
    pairs: list[tuple[nn.Module, str, nn.Linear, str]] = [] #  ( 父模块,   短名字,   原来的Linear,  完整路径 )
    for prefix, module in model.named_modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in wanted:
                path = f"{prefix}.{name}" if prefix else name
                pairs.append((module, name, child, path))
    if not pairs:
        available = sorted(
            {
                name
                for module in model.modules()
                for name, child in module.named_children()
                if isinstance(child, nn.Linear)
            }
        )
        raise ValueError(f"没有找到名为 {sorted(wanted)} 的线性层。可选: {available}")
    found: list[tuple[str, int, int]] = []
    for module, name, child, path in pairs:
        setattr(module, name, LoRALinear.from_linear(child, r=r, alpha=alpha, dropout=dropout))
        found.append((path, child.in_features, child.out_features))
    return found


def freeze_non_lora(model: nn.Module) -> None:
    """除 lora_A / lora_B 外全部冻结 —— LoRA 省参数的关键。"""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def iter_lora(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """开关适配器。关 = 只用 W0（旧行为）；开 = W0+ΔW（新行为）。"""
    modules = list(iter_lora(model))
    if not modules:
        raise RuntimeError("模型里没有 LoRA 层")
    if not enabled and any(module.merged for _, module in modules):
        print("注意: 适配器已经 merge 进 W0，关掉开关不会回到旧模型，需要先 unmerge()")
    for _, module in modules:
        module.enabled = enabled


def lora_parameter_count(in_features: int, out_features: int, r: int) -> int:
    """LoRA 参数量 = r*(in+out)；全量 = in*out。方阵上 r < d/2 才更省。"""
    return r * (in_features + out_features)


@torch.no_grad()
def self_check() -> None:
    """冒烟测试：B=0 无增量、公式一致、merge 可逆、关开关回到 W0。"""
    torch.manual_seed(0)
    base = nn.Linear(8, 5, bias=True)
    layer = LoRALinear.from_linear(base, r=2, alpha=4)
    x = torch.randn(3, 8)

    y0 = layer(x)
    y_base = F.linear(x, layer.weight, layer.bias)
    if not torch.allclose(y0, y_base):
        raise AssertionError("B=0 时 LoRA 应该等于原线性层")
    if layer.weight.requires_grad or not layer.lora_A.requires_grad or not layer.lora_B.requires_grad:
        raise AssertionError("应该只训练 A 和 B")

    layer.lora_A.normal_()
    layer.lora_B.normal_()
    manual = y_base + layer.scaling * (x @ layer.lora_A.T @ layer.lora_B.T)
    y1 = layer(x)
    if not torch.allclose(y1, manual, atol=1e-5):
        raise AssertionError("前向和 (alpha/r) * B A 不一致")

    layer.eval()
    layer.merge()
    y_merged = layer(x)
    if not torch.allclose(y1, y_merged, atol=1e-5):
        raise AssertionError("merge 后输出变了")
    layer.unmerge()
    if not torch.allclose(y1, layer(x), atol=1e-5):
        raise AssertionError("unmerge 没有还原")

    layer.enabled = False
    if not torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias)):
        raise AssertionError("关闭适配器后应该回到 W0")
    layer.enabled = True
