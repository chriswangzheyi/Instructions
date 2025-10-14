# -*- coding: utf-8 -*-
"""
基于 LSTM 的情感分析 - 简化版（无需 torchtext.datasets）
使用自定义数据集进行演示
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import time


# ==================== 1. 简单的数据集 ====================

class SimpleIMDBDataset(Dataset):
    """简化的 IMDB 数据集（用于演示）"""
    
    def __init__(self, is_train=True):
        # 这里使用一些示例数据，实际项目中应该从文件加载
        if is_train:
            self.data = [
                ("This movie is great! I love it.", "pos"),
                ("Terrible film, waste of time.", "neg"),
                ("Amazing performance by the actors.", "pos"),
                ("Boring and predictable plot.", "neg"),
                ("Excellent cinematography and story.", "pos"),
                ("Very disappointing, would not recommend.", "neg"),
                ("A masterpiece! Must watch.", "pos"),
                ("Poor acting and bad script.", "neg"),
                ("Fantastic movie, highly recommended!", "pos"),
                ("Awful experience, total disaster.", "neg"),
            ] * 100  # 重复以增加数据量
        else:
            self.data = [
                ("Good movie, enjoyed it.", "pos"),
                ("Not good at all.", "neg"),
                ("Best film I've seen.", "pos"),
                ("Completely terrible.", "neg"),
            ] * 50
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==================== 2. 词汇表构建 ====================

def build_vocab_simple(dataset):
    """构建简单的词汇表"""
    word_count = {}
    
    for text, _ in dataset:
        words = text.lower().split()
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
    
    # 创建词汇表
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, count in word_count.items():
        if count >= 2:  # 出现至少2次
            vocab[word] = idx
            idx += 1
    
    return vocab


def text_to_indices(text, vocab):
    """将文本转换为索引"""
    words = text.lower().split()
    return [vocab.get(word, vocab['<unk>']) for word in words]


# ==================== 3. 数据处理 ====================

def collate_batch_simple(batch, vocab, device):
    """处理一个批次的数据"""
    texts = []
    labels = []
    lengths = []
    
    for text, label in batch:
        indices = text_to_indices(text, vocab)
        texts.append(indices)
        labels.append(1 if label == 'pos' else 0)
        lengths.append(len(indices))
    
    # 填充
    max_len = max(lengths)
    padded_texts = []
    for text in texts:
        padded = text + [vocab['<pad>']] * (max_len - len(text))
        padded_texts.append(padded)
    
    # 排序
    lengths = torch.tensor(lengths)
    lengths, sorted_idx = lengths.sort(descending=True)
    
    texts_tensor = torch.tensor(padded_texts)[sorted_idx]
    labels_tensor = torch.tensor(labels)[sorted_idx]
    
    return texts_tensor.to(device), lengths.to(device), labels_tensor.to(device)


# ==================== 4. LSTM 模型 ====================

class SimpleLSTM(nn.Module):
    """简化的 LSTM 模型"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, 
                 output_dim=2, n_layers=2, dropout=0.5, pad_idx=0):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed = pack_padded_sequence(embedded, text_lengths.cpu(), 
                                      batch_first=True, enforce_sorted=True)
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # 拼接最后一层的双向隐藏状态
        hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=-1))
        
        return self.fc(hidden)


# ==================== 5. 训练和评估 ====================

def train(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    
    for text, lengths, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(text, lengths)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(1) == labels).float().mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for text, lengths, labels in dataloader:
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def predict_sentiment(model, text, vocab, device):
    """预测单个句子的情感"""
    model.eval()
    
    indices = text_to_indices(text, vocab)
    length = torch.tensor([len(indices)])
    tensor = torch.tensor(indices).unsqueeze(0)
    
    tensor = tensor.to(device)
    length = length.to(device)
    
    with torch.no_grad():
        prediction = model(tensor, length)
        prob = torch.softmax(prediction, dim=1)
        pred_class = prediction.argmax(1).item()
    
    sentiment = 'positive' if pred_class == 1 else 'negative'
    confidence = prob[0][pred_class].item()
    
    return sentiment, confidence


# ==================== 6. 主函数 ====================

def main():
    print("=" * 60)
    print("LSTM 情感分析 - 简化演示版")
    print("=" * 60)
    
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 数据
    print("\n1. 准备数据...")
    train_dataset = SimpleIMDBDataset(is_train=True)
    test_dataset = SimpleIMDBDataset(is_train=False)
    
    vocab = build_vocab_simple(train_dataset)
    print(f"词汇表大小: {len(vocab)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=lambda b: collate_batch_simple(b, vocab, device)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_batch_simple(b, vocab, device)
    )
    
    # 模型
    print("\n2. 创建模型...")
    model = SimpleLSTM(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        output_dim=2,
        n_layers=2,
        dropout=0.5,
        pad_idx=vocab['<pad>']
    )
    model = model.to(device)
    
    # 训练设置
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print("\n3. 开始训练...")
    N_EPOCHS = 5
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        end_time = time.time()
        
        print(f'\nEpoch {epoch+1}/{N_EPOCHS} | 用时: {end_time-start_time:.1f}s')
        print(f'  训练: Loss={train_loss:.3f} | Acc={train_acc*100:.1f}%')
        print(f'  测试: Loss={test_loss:.3f} | Acc={test_acc*100:.1f}%')
    
    # 预测示例
    print("\n4. 预测示例...")
    test_sentences = [
        "This movie is amazing and wonderful!",
        "Terrible waste of time and money.",
        "Great acting and fantastic story.",
        "Boring and disappointing film."
    ]
    
    for sentence in test_sentences:
        sentiment, confidence = predict_sentiment(model, sentence, vocab, device)
        print(f"\n句子: '{sentence}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2%})")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

