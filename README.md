# chatgpt-3k
目标: 3000行代码以内创建微型的chatgpt应用 create a mini-ChatGPT within 3000 lines of code
(仅为学习目编写，没有做任何优化，请勿在真实训练中使用 as Tutorial）

## 说明
部分代码借助了ChatGPT完成 (同时也给出一些误导和错误代码）**Should Never Trust GPT too much**

## 目标
- [x] Transformer
- [x] GPT模型GPT model
- [x] 训练代码simple training code for GPT
- [ ] 奖励模型Reward-Model
- [ ] 偏好训练
- [ ] 微调代码Fine-tuning

## 已知问题
- [ ] 确保所有变量的device一致

## ChatGPT复现步骤
### 预训练Pre-training
成本函数Loss Function:
$$
L_lm = \sigma P(y|x_1, x_2, ..., x_n)
$$

### 微调Fine-tuning
成本函数Loss Function:
$$
L_{fine-tuning} = \alpha * L_lm  + L_task
$$
[微调流程](images/instruct-gpt.png)

## 参考资料Referrence
- [gp-2 official](https://github.com/nshepperd/gpt-2/blob/finetuning/train.py)
- [Generative Pretrained Transformers介绍](https://github.com/iVishalr/GPT)
- [nano-GPT](TODO)
- [GPT-lite](https://brunomaga.github.io/GPT-lite)
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
- [60行NumPy手搓GPT[翻译]](https://jiqihumanr.github.io/2023/04/13/gpt-from-scratch/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [The Annotated Transformer code](https://github.com/harvardnlp/annotated-transformer/blob/debc9fd747bb2123160a98046ad1c2d4da44a567/the_annotated_transformer.py#L326)
- [the_annotated_transformer.py](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)
- [Top-k & Top-p](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p#2-pick-from-amongst-the-top-tokens-top-k)
- Introduce History of Attention, Neural Turing Machine: [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/#neural-turing-machines)
- [ChatGPT 的模型训练](https://brightliao.com/2023/05/20/chatgpt-training/)
- [ChatGPT是怎么被训练出来的](https://www.bookai.top/docs/ChatGPT-tutorial/ChatGPT%E6%98%AF%E6%80%8E%E4%B9%88%E8%A2%AB%E8%AE%AD%E7%BB%83%E5%87%BA%E6%9D%A5%E7%9A%84)

## 使用
### 安装
```
cd chatgpt_3k
pip install .
```
### 例子1 Example01
详细见 `examples/01_train_hello`
`tokenizer.py`
```
import string


class IndexTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.char_to_idx = {char: idx for idx, char in enumerate(string.ascii_lowercase + ' ')}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text):
        # 忽略了未知的字符
        return [self.char_to_idx[char] for char in text.lower() if char in self.char_to_idx]

    def decode(self, tokens):
        return ''.join([self.idx_to_char[token] for token in tokens])
```

`data_set.py`
```
import torch
from torch.utils.data import Dataset

class MemDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len=20):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer.encode(text)
        encoded = encoded[:self.seq_len] + [0] * (self.seq_len - len(encoded))  # Padding/truncating
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        return input_seq, target_seq
```

`run_train.py`
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chatgpt_3k.gpt import GPT, generate_sequence
from dataset import MemDataset
from tokenizer import IndexTokenizer

torch.manual_seed(0)

# 示例数据集
data = [
    "hello world this is a test",
    "this is a simple gpt model",
    "machine learning is awesome",
    "transformers are powerful",
]

batch_size = 2
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
num_epoch = 30

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
start_token = tokenizer.encode("hello")[0]
max_generate_len = 20

# 温度调整生成
generated_sequence = generate_sequence(gpt, start_token, max_generate_len, vocab_size, device='cpu', temperature=0.7)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **temperature**:", decoded_sequence)

# Top-k 采样生成
generated_sequence = generate_sequence(gpt, start_token, max_generate_len, vocab_size, device='cpu', top_k=3)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **top-k** sampling:", decoded_sequence)

# Top-p 采样生成
generated_sequence = generate_sequence(gpt, start_token, max_generate_len, vocab_size, device='cpu', top_p=0.9)
decoded_sequence = tokenizer.decode(generated_sequence)
print("Generated sequence with **top-p** sampling:", decoded_sequence)

```

## 命名规则Naming
- 矩阵大写 e.g. X
- 维度(dimension)以d开头

## 思考
(1) Tokenizer问题
以本句话为例子： “告诉我本句的第3个词和倒数第3个字”
如果不是按字切割，如何获得严格的序数信息
(2) Multi-Head问题
* 当前默认使用8个头，对于英文世界可能够用，对于中文数据，是否需要特殊处理？考虑到中文的多意性，是否需要增大head数量?

## ChangeLogs
- [@2024-08-19] project created
- [@2024-08-22] v.0.0.2 code-review & training example 代码审查+训练代码
