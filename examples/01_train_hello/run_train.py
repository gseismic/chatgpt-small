import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chatgpt_3k.gpt import GPT, generate_sequence

torch.manual_seed(0)

# 本地文件
from dataset import MemDataset
from tokenizer import IndexTokenizer

# 示例数据集
#data = [
#    "hello world this is a test",
#    "this is a simple gpt model",
#    "machine learning is awesome",
#    "transformers are powerful",
#]

data = [
    "hello world this is a test",
]

batch_size = 1

tokenizer = IndexTokenizer()
dataset = MemDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(tokenizer.char_to_idx)

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

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gpt.parameters(), lr=0.001)

# 生成示例输入数据
# input_seq = torch.randint(0, vocab_size, (batch_size, gpt.seq_len))  # (batch_size, seq_len)

num_epoch = 300*3
num_epoch = 300

# 训练步骤
gpt.train()
for epoch in range(num_epoch):
    epoch_loss = 0
    for input_seq, target_seq in dataloader:
        # print(f'{tokenizer.decode([input_seq])=}')
        # print(f'{tokenizer.decode([target_seq])=}')
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()   # 梯度重置为0

        # 生成目标序列，预测下一个词
        output = gpt(input_seq)  # 输出的形状: (batch_size, seq_len, vocab_size)

        # output.shape=torch.Size([2, 18, 27])
        # target.shape=torch.Size([2, 18])
        # 目标右移1位
        # target: [batch_size, seq_len-1]
        target = input_seq[:, 1:]  # 目标是预测下一个词 
        # output: [batch_size, seq_len-1, vocab_size]
        output = output[:, :-1, :]  # 去掉最后一个词的预测

        # 计算损失并使用 .reshape()
        loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}')

# 推断

if 1:
    start_token = tokenizer.encode("hello")[0]
    start_token = tokenizer.encode("t")[0]
    max_generate_len = 20

    if 1:
        # 温度调整生成
        generated_sequence = generate_sequence(gpt, start_token, max_generate_len, vocab_size, device='cpu', temperature=0.7)
        decoded_sequence = tokenizer.decode(generated_sequence)
        print("Generated sequence with **temperature**:", decoded_sequence)

    if 1:
        # Top-k 采样生成
        generated_sequence = generate_sequence(gpt, start_token, max_generate_len, vocab_size, device='cpu', top_k=1)
        print(f'{repr(generated_sequence)=}')
        decoded_sequence = tokenizer.decode(generated_sequence)
        print("Generated sequence with **top-k** sampling:", decoded_sequence)

    if 1:
        # Top-p 采样生成
        generated_sequence = generate_sequence(gpt, start_token, max_generate_len, vocab_size, device='cpu', top_p=0.9)
        decoded_sequence = tokenizer.decode(generated_sequence)
        print("Generated sequence with **top-p** sampling:", decoded_sequence)
if 0:
    def generate_sequence2(model, start_token, max_len, vocab_size, temperature=1.0, top_k=0, top_p=0.9, device='cpu'):
        model.eval()
        generated = [start_token]
        for _ in range(max_len):
            input_seq = torch.tensor(generated[-max_len:], dtype=torch.long).unsqueeze(0).to(device)
            logits = model(input_seq)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_probs, dim=-1)
                next_token = top_k_indices[0, torch.multinomial(probs, 1).item()]
            else:
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_keep = cumulative_probs <= top_p
                sorted_indices_to_keep[sorted_indices_to_keep.sum()] = True  # Ensure at least one token is selected
                sorted_indices = sorted_indices[sorted_indices_to_keep]
                next_token = sorted_indices[torch.multinomial(sorted_probs[sorted_indices_to_keep], 1).item()]

            generated.append(next_token.item())

        return generated

    start_token = tokenizer.encode("hello")[0]
    generated_sequence = generate_sequence2(gpt, start_token, max_len=20, vocab_size=vocab_size, device=device)
    decoded_sequence = tokenizer.decode(generated_sequence)
    print(f"Generated sequence: {decoded_sequence}")

