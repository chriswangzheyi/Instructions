# Cuda 安装教程（Windows版本）

以GeForce RTX 4070 Ti为例。



## 安装步骤



### 步骤1： 查询是否兼容

访问 NVIDIA 官方兼容性列表：

[CUDA GPU Compute Capability | NVIDIA Developer](https://developer.nvidia.com/cuda/gpus)



### 步骤2： 安装

打开 NVIDIA CUDA Toolkit 官方下载页：https://developer.nvidia.com/cuda-downloads



### 步骤3： 配置环境变量

「系统变量」中，找到「Path」，双击编辑

```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
```



### 步骤4： 验证CUDA安装是否成功

```bash
nvcc -V  
```

显示：

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Nov__7_19:25:04_Pacific_Standard_Time_2025
Cuda compilation tools, release 13.1, V13.1.80
Build cuda_13.1.r13.1/compiler.36836380_0
```



### 步骤5：使用 nvidia-smi 查看设备信息

```bash
nvidia-smi
```

显示：

![](\Images\1.png.png)



##### 1. 顶部基础信息

- `NVIDIA-SMI 566.03`：这是 NVIDIA 显卡管理工具的版本
- `Driver Version: 566.03`：当前安装的 NVIDIA 显卡驱动版本（这个版本支持 CUDA 12.7）
- `CUDA Version: 12.7`：显卡驱动**支持的最高 CUDA 版本**（你安装的 CUDA 版本不能超过 12.7）

##### 2. 显卡核心状态（GPU 0）

- `Name: NVIDIA GeForce RTX 4070`：你的显卡型号是 RTX 4070
- `Driver-Model: WDDM`：Windows 系统的显卡驱动模型（说明你用的是 Windows 系统）
- `Bus-Id: 00000000:01:00.0`：显卡在主板上的总线位置
- `Disp.A: On`：显卡正在输出显示（接了显示器）

##### 3. 硬件运行状态

- `Fan: 0%`：显卡风扇转速为 0（当前负载低，风扇没转）
- `Temp: 54°C`：显卡当前温度 54 度（正常范围）
- `Perf: P0`：显卡性能模式（P0 是最高性能模式）
- `Pwr:Usage/Cap: 37W / 220W`：当前功耗 37 瓦，显卡最大功耗 220 瓦（负载很低）
- `Memory-Usage: 2643MiB / 12282MiB`：已用显存 2.6GB，总显存 12GB
- `GPU-Util: 1%`：显卡使用率仅 1%（几乎没负载）



### 步骤6：基础 CUDA 程序测试

通过 NVIDIA 官方提供的示例程序，或一个简单的 Hello CUDA 程序，验证 CUDA 能否正常编译和运行。



```
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
__nvcc_device_query.exe
```



显示：

```
89
```

说明运行成功



## 测试真实服务



### 步骤1 安装conda

清华源 Miniconda 下载地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

测试：

```bash
C:\Users\zheyi>conda --version
conda 24.3.0
```

配置 Conda 清华源:

```bash
# 添加清华源镜像（覆盖默认源）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

conda config --set show_channel_urls yes
```



### 步骤2： 创建GPU专属Conda环境

创建 GPU 专属 Conda 环境

```bash 
# 创建名为gpu_env的环境，Python 3.10（兼容性最优）
conda create -n gpu_env python=3.10
```



激活 gpu_env 环境：

```bash
conda activate gpu_env
```



执行以下命令安装 PyTorch GPU 版本：

```bash
conda install pytorch torchvision torchaudio 
```



### Demo 1： 基础配置



```bash
conda activate gpu_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

配置环境变量：



```
在「用户变量」或「系统变量」中点击「新建」：
变量名：KMP_DUPLICATE_LIB_OK
变量值：TRUE
```

安装完成后，先执行简单的验证命令，确认 PyTorch 版本和 CUDA 支持：

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"
```

打印：

```
PyTorch版本: 2.5.1
CUDA可用: True
CUDA版本: 12.1
```



Python文件，test_cuda_pytorch.py：

```python
import torch
import platform

def test_cuda_pytorch():
    """
    完整的PyTorch CUDA测试函数
    """
    print("=" * 60)
    print("PyTorch GPU 环境测试 (Windows + CUDA)")
    print("=" * 60)
    
    # 1. 基础信息
    print(f"\n1. 系统信息: {platform.system()} {platform.release()}")
    print(f"2. PyTorch 版本: {torch.__version__}")
    print(f"3. CUDA 是否可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")
    
    if torch.cuda.is_available():
        # 2. GPU设备信息
        device_count = torch.cuda.device_count()
        print(f"4. 可用GPU数量: {device_count}")
        
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   - GPU {i}: {device}")
            print(f"     - Compute Capability: {capability[0]}.{capability[1]}")
            print(f"     - 总显存: {memory:.1f} GB")
        
        # 3. 设置默认设备
        device = torch.device("cuda:0")
        print(f"\n5. 使用默认GPU设备: {torch.cuda.get_device_name(device)}")
        
        # 4. 测试GPU张量运算
        print("\n6. 测试GPU张量运算:")
        # 在GPU上创建张量
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        z = x + y
        
        print(f"   - 输入张量1 (GPU): {x}")
        print(f"   - 输入张量2 (GPU): {y}")
        print(f"   - 运算结果 (GPU): {z}")
        print(f"   - 张量所在设备: {z.device}")
        
        # 5. 测试GPU加速的矩阵运算
        print("\n7. 测试GPU矩阵运算加速:")
        # 创建大矩阵测试性能
        size = 2048
        # CPU矩阵运算
        print("   - 正在执行CPU矩阵乘法...")
        cpu_a = torch.randn(size, size)
        cpu_b = torch.randn(size, size)
        cpu_start = torch.cuda.Event(enable_timing=True)
        cpu_end = torch.cuda.Event(enable_timing=True)
        cpu_start.record()
        cpu_c = torch.matmul(cpu_a, cpu_b)
        cpu_end.record()
        torch.cuda.synchronize()
        cpu_time = cpu_start.elapsed_time(cpu_end)
        
        # GPU矩阵运算
        print("   - 正在执行GPU矩阵乘法...")
        gpu_a = cpu_a.to(device)
        gpu_b = cpu_b.to(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        gpu_c = torch.matmul(gpu_a, gpu_b)
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_time = gpu_start.elapsed_time(gpu_end)
        
        print(f"   - CPU 运算时间: {cpu_time:.2f} ms")
        print(f"   - GPU 运算时间: {gpu_time:.2f} ms")
        print(f"   - 加速倍数: {cpu_time/gpu_time:.2f}x")
        
        # 验证结果一致性
        gpu_c_cpu = gpu_c.cpu()
        diff = torch.abs(cpu_c - gpu_c_cpu).max()
        print(f"   - CPU/GPU 结果最大差值: {diff:.6f} (越小越好)")
        
        print("\n✅ 所有测试通过！CUDA和PyTorch GPU版本工作正常。")
        
    else:
        print("\n❌ CUDA不可用！请检查：")
        print("   1. 显卡驱动是否安装正确")
        print("   2. CUDA Toolkit是否安装并配置环境变量")
        print("   3. PyTorch是否安装了GPU版本")
        print("   4. Conda环境是否激活")

if __name__ == "__main__":
    test_cuda_pytorch()
```



执行：

```
python test_cuda_pytorch.py
```



打印：

```bash
PyTorch GPU 环境测试 (Windows + CUDA)
============================================================

1. 系统信息: Windows 10
2. PyTorch 版本: 2.5.1
3. CUDA 是否可用: ✅ 是
4. 可用GPU数量: 1
   - GPU 0: NVIDIA GeForce RTX 4070 SUPER
     - Compute Capability: 8.9
     - 总显存: 12.0 GB

5. 使用默认GPU设备: NVIDIA GeForce RTX 4070 SUPER

6. 测试GPU张量运算:
   - 输入张量1 (GPU): tensor([1., 2., 3.], device='cuda:0')
   - 输入张量2 (GPU): tensor([4., 5., 6.], device='cuda:0')
   - 运算结果 (GPU): tensor([5., 7., 9.], device='cuda:0')
   - 张量所在设备: cuda:0

7. 测试GPU矩阵运算加速:
   - 正在执行CPU矩阵乘法...
   - 正在执行GPU矩阵乘法...
   - CPU 运算时间: 22.69 ms
   - GPU 运算时间: 20.10 ms
   - 加速倍数: 1.13x
   - CPU/GPU 结果最大差值: 0.000778 (越小越好)

✅ 所有测试通过！CUDA和PyTorch GPU版本工作正常。
```

