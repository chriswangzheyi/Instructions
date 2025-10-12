# ddp_demo.py
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# -------------------------
# 小数据集：学习 y = 2x + 3
# -------------------------
class ToyDataset(Dataset):
    def __init__(self, n=10_000):
        rng = np.random.RandomState(42)
        self.x = rng.randn(n, 1).astype("float32")
        noise = 0.1 * rng.randn(n, 1).astype("float32")
        self.y = 2.0 * self.x + 3.0 + noise

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

# -------------------------
# 简单线性模型
# -------------------------
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)
    def forward(self, x):
        return self.net(x)

def setup_seed(seed: int, rank: int):
    # 不同 rank 稍微扰动，避免所有进程里完全同一顺序导致某些现象不好观察
    seed = seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_main_process() -> bool:  #只有 rank==0 的进程负责打印日志/保存模型，避免 N 份重复输出或相互覆盖。
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def ddp_main(rank: int, world_size: int, args):
    """
    rank: 进程在全局的编号 [0..world_size-1]
    world_size: 总进程数（通常=GPU数）
    """
    # 1) 初始化进程组
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # torchrun 默认 env://，单机可以 --standalone
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # 2) 设备与随机种子
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")  # 单机多卡：local_rank==rank,本进程只用 与 rank 相同编号的 GPU,防止多个进程抢同一张卡。
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    setup_seed(2025, rank)

    # 3) 构造数据集 + 分布式采样器
    dataset = ToyDataset(n=20000)
    sampler = DistributedSampler(
      dataset,
      num_replicas=world_size,  # 总共有多少个进程（通常=GPU数）
      rank=rank,                # 当前进程编号（0..world_size-1）
      shuffle=True
		)
    
    loader = DataLoader(
        dataset,
        batch_size=256,       # 每次从本 rank 的子集里取 256 条样本 作为一个 mini-batch
        sampler=sampler,      # DDP 关键点1：使用 DistributedSampler
        num_workers=0,        # 代表数据加载线程数。当为0时，数据加载在主进程中完成。
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # 4) 模型、优化器、DDP 包装
    model = LinearModel().to(device)
    # DDP 关键点2：先 to(device)，再 DDP 包装
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) #把模型包成分布式版本
    criterion = nn.MSELoss()   #定义损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.05)  #定义优化器：优化算法：SGD（随机梯度下降）

    # 5) 训练循环
    epochs = 5
    t0 = time.time() # 记录时间
    for epoch in range(epochs):
        # DDP 关键点3：每个 epoch 设置下采样器的 epoch，确保各 rank 的 shuffle 一致但不重复
        sampler.set_epoch(epoch) # 分布式训练中的关键语句之一：DistributedSampler 负责打乱（shuffle）和切分数据。如果你不调用 set_epoch，每个 epoch 的随机种子会相同

        running_loss = 0.0
        for step, (x, y) in enumerate(loader):
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True) #清空上一轮的梯度
            pred = model(x) #前向传播
            loss = criterion(pred, y) #计算当前 rank 的损失（MSE）
            loss.backward()      # DDP 关键点4：梯度在此处按参数名自动 AllReduce（平均）：PyTorch 正常计算本进程的梯度；DDP 注册的 hook 被触发；所有进程的梯度通过通信后端（NCCL/Gloo）进行 AllReduce 求平均，每个 rank 得到相同的平均梯度
            optimizer.step()     #用同步后的梯度更新模型参数
            running_loss += loss.item()  #累积每个 batch 的损失，用于计算 epoch 平均损失。

            if step % 50 == 0 and is_main_process():   
                print(f"[epoch {epoch} step {step}] loss={loss.item():.6f}")

        # 只在 rank0 汇报
        if is_main_process():
            print(f"Epoch {epoch} done. avg loss={running_loss/len(loader):.6f}")

    # 6) 只在主进程保存模型
    if is_main_process():
        # 从 DDP wrapper 取出原模型
        torch.save(model.module.state_dict(), "ddp_linear.pt")
        print("Saved model to ddp_linear.pt")
        # 打印最终参数，看看是否接近 y=2x+3
        W = model.module.net.weight.item()
        b = model.module.net.bias.item()
        print(f"Fitted W={W:.4f}, b={b:.4f} (expect ~2.0, ~3.0)")
        print(f"Total time: {time.time()-t0:.2f}s")

    # 7) 清理
    dist.destroy_process_group()

def parse_world_size_from_env():
    # torchrun 会设置 LOCAL_RANK/RANK/WORLD_SIZE 环境变量
    ws = os.environ.get("WORLD_SIZE", None)
    return int(ws) if ws is not None else 1

if __name__ == "__main__":
    world_size = parse_world_size_from_env()
    # 使用 torchrun 时，每个子进程会直接执行到这里；rank 从环境变量中读
    rank = int(os.environ.get("RANK", 0))

    # 单机调试（直接 python ddp_demo.py），也能跑一个进程理解流程
    ddp_main(rank, world_size, args=None)
