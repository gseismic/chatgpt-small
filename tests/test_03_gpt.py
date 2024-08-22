import config
from chatgpt_3k.config import torch, nn, optim
from chatgpt_3k.gpt import (
    GPT, generate_sequence
)
#import matplotlib.pyplot as plt


def test_gpt_basic():
    # 使用示例
    embed_dim = 512
    num_heads = 8
    num_layers = 12
    ff_dim = 2048
    vocab_size = 10000
    batch_size = 5
    dropout = 0.1
    seq_len = 3

    gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size,
              seq_len=seq_len, dropout=dropout)

    src = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    output = gpt(src)
    print(f"Output shape: {output.shape}")  # 输出: [32, 20, vocab_size]


def test_gpt_generate():
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    vocab_size = 10000
    dropout = 0.1
    max_len = 20
    start_token = 0  # 起始token索引

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size,
              seq_len=max_len, dropout=dropout).to(device)

    # 温度调整生成
    generated_sequence = generate_sequence(gpt, start_token, max_len, vocab_size, device='cpu', temperature=0.7)
    print("Generated sequence with **temperature**:", generated_sequence)

    # Top-k 采样生成
    generated_sequence = generate_sequence(gpt, start_token, max_len, vocab_size, device='cpu', top_k=5)
    print("Generated sequence with **top-k** sampling:", generated_sequence)

    # Top-p 采样生成
    generated_sequence = generate_sequence(gpt, start_token, max_len, vocab_size, device='cpu', top_p=0.6)
    print("Generated sequence with **top-p** sampling:", generated_sequence)

def test_gpt_train_simple():
    # Note: 没有添加tokenizer
    # 设置模型参数
    embed_dim = 512
    num_heads = 8
    num_layers = 6      # Transformer 解码器层数
    ff_dim = 2048       # 前馈网络隐藏层维度
    vocab_size = 10000
    dropout = 0.1
    max_len = 20
    start_token = 0  # 起始token索引

    # 实例化模型
    gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size,
              seq_len=max_len, dropout=dropout)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gpt.parameters(), lr=0.0001)

    # 生成示例输入数据
    batch_size = 32
    input_seq = torch.randint(0, vocab_size, (batch_size, max_len))  # (batch_size, seq_len)

    # 训练步骤
    gpt.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # 生成目标序列，预测下一个词
        output = gpt(input_seq)  # 输出的形状: (batch_size, seq_len, vocab_size)
        
        # 将目标与输入错开，shift right one position to match LM objective
        # target: [batch_size, max_len-1]
        target = input_seq[:, 1:]  # 目标是预测下一个词 
        # output: [batch_size, max_len-1, vocab_size]
        output = output[:, :-1, :]  # 去掉最后一个词的预测
        #print(f'{output.shape=}')
        #print(f'{target.shape=}')
        
        # 计算损失并使用 .reshape()
        loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == '__main__':
    if 0:
        test_gpt_basic()
    if 0:
        test_gpt_generate()
    if 1:
        test_gpt_train_simple()
