# transformer PE计算公式
Transformer 的位置编码（Positional Encoding, PE）是通过对每个位置使用正弦和余弦函数生成的。这些位置编码向量被加到输入的词嵌入中，以便将位置信息融入到模型中。

Transformer 位置编码的计算公式如下：

### 1. 位置编码的数学公式

对于序列中位置 \( pos \) 和嵌入维度中的某个维度 \( i \)，位置编码的公式为：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

- \( pos \) 表示输入序列中 token 的位置。
- \( i \) 表示位置编码向量中的维度索引。
- \( d_{model} \) 表示词嵌入的维度大小（即嵌入向量的维度）。

### 2. 位置编码的直观理解

- 对于给定的 \( pos \)，奇数维度的值是余弦函数，偶数维度的值是正弦函数。
- 这些正弦和余弦函数的周期不同，因此每个位置会生成一个独特的编码向量，随着位置 \( pos \) 增加，编码向量以不同频率的正弦和余弦波动，保证了位置信息的唯一性和区分性。
- 位置编码与词嵌入相加，模型可以在多头注意力机制中使用这些位置信息来捕捉序列中的相对位置关系。

### 3. 代码示例

以下是 PyTorch 实现位置编码的代码示例：

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

# Example usage:
seq_len = 10
d_model = 16
pe = positional_encoding(seq_len, d_model)
print(pe)
```

### 4. 位置编码的特点

- **全局性**：由于编码涉及到全局位置 \( pos \) 的计算，因此它可以为 Transformer 模型提供关于序列整体位置的信息。
- **固定不变**：位置编码在模型训练期间是固定的，不像词嵌入会被更新。这保证了模型始终能基于相同的位置信息进行推理。
- **频谱特性**：由于不同维度的频率不同，位置编码能够捕捉到不同粒度的位置信息。

### 总结

位置编码公式为 Transformer 提供了一种将序列位置信息融入到模型中的方法，利用正弦和余弦函数的周期性来生成唯一且可区分的编码，确保模型能够有效地捕捉序列中的相对位置关系。

# 为什么要用exp + log方式计算
在计算Transformer的位置编码时，使用`exp`（指数）和`log`（对数）组合的方式是为了优化**计算效率**和**数值稳定性**。下面是详细原因：
## 位置编码公式回顾

位置编码的计算公式如下：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

其中：
- \( pos \) 是序列中的位置。
- \( i \) 是嵌入维度中的索引。
- \( d_{model} \) 是嵌入维度的大小。

## 使用 `exp` 和 `log` 的原因

### 1. 数学简化

原始公式中的幂运算 

$$
\frac{pos}{10000^{\frac{2i}{d_{model}}}}
$$ 

可能涉及到计算大数或小数，容易导致计算复杂度增加和数值不稳定。通过对公式进行以下等价变换：

$$
\frac{pos}{10000^{\frac{2i}{d_{model}}}} = pos \cdot \frac{1}{10000^{\frac{2i}{d_{model}}}} = pos \cdot e^{- \frac{2i}{d_{model}} \cdot \ln(10000)}
$$

我们可以将复杂的幂运算转换为更容易处理的指数和对数运算。

### 2. 计算效率

通过预计算常数部分 \( - \frac{2i}{d_{model}} \cdot \ln(10000) \)，整个计算过程可以实现为一系列简单的向量运算：

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

这样可以更高效地利用硬件的向量化操作，减少计算成本。

### 3. 数值稳定性

直接计算幂运算可能会导致数值溢出或精度损失。而使用 `exp` 和 `log`，可以将运算转化到对数空间，避免计算大数或小数，从而提高计算的数值稳定性，减少浮点数误差。

### 4. 通用性和可扩展性

通过这种方式计算的位置编码公式具有通用性，可以适用于不同长度的序列和不同的嵌入维度，无需对公式进行修改。这种方法简化了代码实现，使其更容易在不同应用场景下扩展。

## 总结

使用`exp`和`log`的组合方式来计算位置编码，主要是为了简化数学运算、提升计算效率、增强数值稳定性，以及提高代码的通用性和可扩展性。这种方法使得Transformer模型在处理不同长度的序列时表现更为稳定和高效。
