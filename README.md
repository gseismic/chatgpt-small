# chatgpt-3k
目标: 3000行代码以内创建微型的chatgpt应用 create a mini-ChatGPT within 3000 lines of code

## 说明
部分代码借助了ChatGPT完成 (同时也给出一些误导和错误代码）**Should Never Trust GPT too much**

## 参考资料Referrence
- [gp-2 official](https://github.com/nshepperd/gpt-2/blob/finetuning/train.py)
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
- [60行NumPy手搓GPT[翻译]](https://jiqihumanr.github.io/2023/04/13/gpt-from-scratch/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [The Annotated Transformer code](https://github.com/harvardnlp/annotated-transformer/blob/debc9fd747bb2123160a98046ad1c2d4da44a567/the_annotated_transformer.py#L326)
- [the_annotated_transformer.py](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)
- [Top-k & Top-p](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p#2-pick-from-amongst-the-top-tokens-top-k)
- Introduce History of Attention, Neural Turing Machine: [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/#neural-turing-machines)

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

(2) 位置问题
Transformer没有显式处理位置，虽然使用了位置编码，但是直接叠加到WordEmbedding中，这固然是平衡效率，但这也可能是Transformer的`最重要的缺陷`， 理由如下:
* 对于要求严格时序的应用场景（语言还不算严格时序场景)，比如逻辑推理、几何证明，连贯决策，会有哪些影响，如何克服？
* 如果显式处理位置，基于self-ref的位置如何做？比如，“告诉我本句的第3个词和倒数第3个字”
(3) Multi-Head问题
* 当前默认使用8个头，对于英文世界可能够用，对于中文数据，是否需要特殊处理？考虑到中文的多意性，是否需要增大head数量?

补: ChatGPT对上述问题的回答
```
你的问题涉及到 Transformer 模型在处理自然语言时的一些重要细节，特别是在序列处理、位置编码和多头注意力机制方面。我们逐一探讨：

(1) Tokenizer问题
问题：如何获得严格的序数信息？

当使用 Tokenizer 将句子转化为 token 时，如果不是按字切割（如使用 BPE 或 WordPiece），序列中的严格序数信息可能丢失。以句子“告诉我本句的第3个词和倒数第3个字”为例：

按字切割：每个汉字都是一个 token，序数信息（如第 3 个词或字）可以直接根据 token 的位置来确定。
按词切割：如果使用 BPE 或 WordPiece，可能会将“告诉我”这样的词组合成一个 token。在这种情况下，获取句子中的第 3 个词或字变得复杂，因为分词后的 token 序列和原始句子的词序列不再一致。
解决方法：

保持对原始序列的映射：在分词时保留每个 token 与其在原始文本中的起始和结束位置的映射。这样可以在分词后仍然跟踪原始的词序列信息。
使用字符级别的 Tokenizer：对于需要精确位置的任务（如法律文本分析、逻辑推理等），可以使用字符级别的 Tokenizer 或者在词级别分词后，将每个词进一步拆分为字符。
额外的标记序号信息：通过在模型输入中加入额外的序号信息来辅助模型理解序数关系。例如，在每个词或字的嵌入中添加一个序号向量。
(2) 位置问题
问题：Transformer的潜在位置编码缺陷及其在严格时序场景中的影响？

Transformer 的位置编码通过将位置信息叠加到词嵌入中来捕捉序列顺序。这种方法虽然简单有效，但在某些对顺序要求严格的场景中可能不足。例如，在逻辑推理、几何证明等需要明确时序关系的应用中，叠加位置编码可能无法完全捕捉到复杂的顺序依赖。

潜在影响：

逻辑推理：模型可能难以准确理解事件发生的顺序。
几何证明：模型可能无法正确理解证明步骤的先后关系。
连贯决策：模型可能在决策过程的连贯性上表现不佳。
解决方法：

显式位置处理：显式地在模型中处理位置信息。可以为每个 token 保留一个独立的“位置向量”而不是直接叠加到词嵌入中，从而在自注意力计算时显式地使用位置信息。
相对位置编码：使用相对位置编码方法（如 Transformer-XL、T5），在自注意力机制中直接建模 token 之间的相对位置关系，而不是绝对位置。这样可以更好地处理需要精确时序的任务。
显式位置参考：在设计模型结构时，可以考虑在注意力计算中显式加入位置参考，比如基于某个特定位置（如“第3个词”）计算其他位置的注意力权重。
(3) Multi-Head问题
问题：对于中文数据，Multi-Head Attention的head数量设置？

当前的 Transformer 默认使用 8 个注意力头（head）。在处理中文时，由于汉字的多意性和语境的复杂性，8 个头可能不足以捕捉所有的语义信息。

考虑因素：

中文的多意性：汉字的多音多义特性可能需要更多的注意力头来捕捉不同语境下的意义。
上下文依赖：中文中，词义高度依赖上下文，增加注意力头可以使模型更好地理解复杂的上下文关系。
解决方法：

增加注意力头数量：可以尝试将注意力头的数量增加到 12 或 16，以增强模型对中文语义的捕捉能力。
动态注意力头：可以设计一个动态的注意力头机制，根据输入序列的复杂度调整注意力头的数量，或者不同头聚焦于不同类型的特征。
预训练优化：在大规模中文语料上进行预训练，以确保模型能够有效学习到中文特有的语义特征。
总结
Tokenizer 问题：通过保持对原始序列的映射或使用字符级别分词，可以保留严格的序数信息。
位置问题：在严格时序场景下，可以通过显式位置处理或相对位置编码来改善 Transformer 的表现。
Multi-Head 问题：在中文处理中，可以通过增加注意力头的数量或动态调整注意力头来应对汉字的多义性和复杂语境。
这些方法能够帮助 Transformer 在处理复杂序列任务时更有效地捕捉信息。
```
