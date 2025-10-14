"""
LSTM 情感分析 - 快速入门示例
这是一个简化版本，用于快速理解和测试
"""

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

# ==================== 模型定义 ====================

class SimpleLSTM(nn.Module):
    """简化的 LSTM 情感分析模型"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        _, (hidden, _) = self.lstm(embedded)  # hidden: [1, batch_size, hidden_dim]
        output = self.fc(hidden.squeeze(0))  # [batch_size, output_dim]
        return output


# ==================== 示例：如何使用模型 ====================

def demo():
    """演示模型的使用"""
    
    print("=" * 60)
    print("LSTM 情感分析 - 快速示例")
    print("=" * 60)
    
    # 1. 创建一个简单的词汇表
    print("\n1. 创建词汇表...")
    vocab = {
        '<pad>': 0, '<unk>': 1,
        'this': 2, 'is': 3, 'a': 4, 'good': 5, 'bad': 6,
        'movie': 7, 'film': 8, 'great': 9, 'terrible': 10,
        'love': 11, 'hate': 12, 'amazing': 13, 'boring': 14
    }
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    model = SimpleLSTM(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2)
    print(f"模型结构:\n{model}")
    
    # 3. 准备输入数据
    print("\n3. 准备示例数据...")
    # 假设我们有两个句子：
    # "this is a good movie" -> [2, 3, 4, 5, 7]
    # "this is a bad film" -> [2, 3, 4, 6, 8]
    
    sentences = torch.tensor([
        [2, 3, 4, 5, 7],  # "this is a good movie"
        [2, 3, 4, 6, 8],  # "this is a bad film"
    ])
    print(f"输入张量形状: {sentences.shape}")
    
    # 4. 前向传播
    print("\n4. 模型推理...")
    model.eval()
    with torch.no_grad():
        output = model(sentences)
        predictions = torch.softmax(output, dim=1)
        predicted_classes = output.argmax(dim=1)
    
    print(f"输出形状: {output.shape}")
    print(f"预测概率:\n{predictions}")
    print(f"预测类别: {predicted_classes}")
    
    # 5. 解释输出
    print("\n5. 结果解释...")
    labels = ['negative', 'positive']
    for i, (probs, pred_class) in enumerate(zip(predictions, predicted_classes)):
        print(f"句子 {i+1}:")
        print(f"  预测: {labels[pred_class.item()]}")
        print(f"  负面概率: {probs[0].item():.4f}")
        print(f"  正面概率: {probs[1].item():.4f}")
    
    print("\n" + "=" * 60)
    print("注意：这只是演示模型的使用，模型尚未训练，")
    print("所以预测结果是随机的。")
    print("要获得有意义的结果，请运行 sentiment_analysis_lstm.py")
    print("=" * 60)


# ==================== LSTM 核心概念说明 ====================

def explain_lstm():
    """解释 LSTM 的关键概念"""
    
    print("\n" + "=" * 60)
    print("LSTM 核心概念说明")
    print("=" * 60)
    
    print("""
    LSTM (Long Short-Term Memory) 是一种特殊的 RNN，能够学习长期依赖关系。
    
    关键组件：
    1. 遗忘门 (Forget Gate): 决定丢弃什么信息
    2. 输入门 (Input Gate): 决定存储什么新信息
    3. 输出门 (Output Gate): 决定输出什么信息
    4. 细胞状态 (Cell State): 信息的"传送带"
    
    在情感分析中的应用：
    - Embedding 层：将词转换为向量表示
    - LSTM 层：捕捉句子中的上下文和语义关系
    - 全连接层：将 LSTM 输出映射到分类结果
    
    双向 LSTM (Bidirectional LSTM)：
    - 同时从前向后和从后向前处理序列
    - 能够捕捉更丰富的上下文信息
    - 输出维度是单向 LSTM 的两倍
    
    Pack Padded Sequence：
    - 避免对填充 (padding) 部分进行无效计算
    - 提高训练效率
    - 需要序列按长度降序排列
    """)
    
    print("=" * 60)


# ==================== 数据处理示例 ====================

def explain_data_processing():
    """解释数据处理流程"""
    
    print("\n" + "=" * 60)
    print("数据处理流程说明")
    print("=" * 60)
    
    print("""
    1. 文本 -> 分词 (Tokenization)
       "This is great!" -> ['this', 'is', 'great', '!']
    
    2. 词 -> 索引 (Word to Index)
       ['this', 'is', 'great'] -> [2, 3, 9]
    
    3. 填充 (Padding)
       为了批量处理，需要将不同长度的序列填充到相同长度
       [2, 3, 9] -> [2, 3, 9, 0, 0]  (假设最大长度为5)
    
    4. 批处理 (Batching)
       将多个序列组合成一个批次
       [[2, 3, 9, 0, 0],
        [5, 7, 0, 0, 0],
        [2, 3, 4, 5, 7]]
    
    5. 记录长度 (Length Tracking)
       记录每个序列的实际长度（用于 pack_padded_sequence）
       [3, 2, 5]
    """)
    
    print("=" * 60)


# ==================== 主函数 ====================

if __name__ == '__main__':
    # 运行演示
    demo()
    
    # 显示 LSTM 概念说明
    explain_lstm()
    
    # 显示数据处理说明
    explain_data_processing()
    
    print("\n" + "=" * 60)
    print("下一步：")
    print("运行 'python sentiment_analysis_lstm.py' 开始完整训练")
    print("=" * 60)

