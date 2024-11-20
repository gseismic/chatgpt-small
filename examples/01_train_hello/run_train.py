import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chatgpt_3k.gpt import GPT, generate_sequence
from dataset import MemDataset
from tokenizer import CharTokenizer, NumTokenizer

torch.manual_seed(0)

# 示例数据集
if 0:
    data = [
        "hello world this is a test",
        "this is a simple gpt model",
        "machine learning is awesome",
        "transformers are powerful",
    ]
    tokenizer = CharTokenizer()
    start_tokens = tokenizer.encode("h")[0] # 推断用
    epoch_verbose_circle = 10
    num_epoch = 30

if 1:
    data = [
        f'{i} x {j} = {i*j}'
        for j in range(10)
        for i in range(10)
    ]
    tokenizer = NumTokenizer()
    start_tokens = tokenizer.encode("3 x 5") # 推断用
    start_tokens = tokenizer.encode("2 x 5")[0] # 推断用
    print(f'{start_tokens=}')
    epoch_verbose_circle = 1
    num_epoch = 100

print(data)
batch_size = 2
dataset = MemDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size

# define model
# num_layers, embed_dim, num_heads, ff_dim, vocab_size, seq_len=512, dropout=0.1
gpt_config = {
    "num_layers": 2,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 512,
    "vocab_size": vocab_size,
    "seq_len": 20,
    "dropout": 0.1
}
# gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size, dropout=dropout)
gpt = GPT.from_config(gpt_config)
gpt.to(device)
max_seq_len = gpt_config['seq_len']

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gpt.parameters(), lr=0.001)

# 训练步骤
gpt.train()
for epoch in range(num_epoch):
    epoch_loss = 0
    for input_seq, target_seq in dataloader:
        # input_seq, target_seq: 移位已经在dataset中处理了
        print(f'{tokenizer.decode([input_seq])=}')
        print(f'{tokenizer.decode([target_seq])=}')
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        # 生成目标序列，预测下一个词
        output = gpt(input_seq)  # 输出的形状: (batch_size, seq_len, vocab_size)
        # 目标右移1位
        # 计算损失并使用 .reshape()
        loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))

        optimizer.zero_grad()   # 梯度重置为0
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % epoch_verbose_circle == 0:
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.6f}')

raise
# 推断
#start_tokens = tokenizer.encode("hello")[0]
#start_tokens = tokenizer.encode("this")[0]
max_generate_len = 20

# 温度调整生成
generated_sequence = generate_sequence(gpt, start_tokens, max_generate_len,
                                       vocab_size, max_seq_len=max_seq_len, device='cpu', temperature=0.7)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **temperature**:", decoded_sequence)

# Top-k 采样生成
generated_sequence = generate_sequence(gpt, start_tokens, max_generate_len,
                                       vocab_size,max_seq_len=max_seq_len,  device='cpu', top_k=3)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **top-k** sampling:", decoded_sequence)

# Top-p 采样生成
generated_sequence = generate_sequence(gpt, start_tokens, max_generate_len,
                                       vocab_size,max_seq_len=max_seq_len,  device='cpu', top_p=0.9)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **top-p** sampling:", decoded_sequence)
