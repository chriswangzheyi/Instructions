# LSTM 情感分析项目

## 🚀 快速运行

### 方式1：使用脚本（最简单）

```bash
./运行.sh
```

### 方式2：手动运行

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 运行（使用示例数据）
export USE_SAMPLE_DATA=1
python sentiment_analysis_lstm.py
```

## ❌ 常见错误

### 错误：`ModuleNotFoundError: No module named 'torch'`

**原因**：没有激活虚拟环境

**解决**：
```bash
# 先激活虚拟环境
source venv/bin/activate

# 再运行
python sentiment_analysis_lstm.py
```

### 错误：直接点击文件运行失败

**原因**：需要在命令行中运行，且需要激活虚拟环境

**解决**：在终端运行 `./运行.sh`

## 📋 项目说明

- **sentiment_analysis_lstm.py**: 主程序
- **venv/**: Python 虚拟环境（已安装 torch）
- **运行.sh**: 一键运行脚本

## 💡 提示

1. **必须**在虚拟环境中运行
2. 可以直接用 `./运行.sh` 一键运行
3. 默认使用示例数据，30秒完成训练
