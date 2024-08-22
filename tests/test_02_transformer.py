import config
from chatgpt_3k.config import torch
from chatgpt_3k.transformer import (
    PositionalEncoding, MultiHeadAttention,
    Transformer
)
import matplotlib.pyplot as plt


def test_transformer_pe():
    # 示例数据
    batch_size = 2
    n_seq = 100
    d_model = 10
    X = torch.zeros(batch_size, n_seq, d_model)
    pe = PositionalEncoding(n_seq, d_model, pe_base=10000)
    X_pe = pe(X)
    print(X_pe, X_pe.shape)
    # plt.plot(X_pe)

def test_transformer_attention():
    # 使用示例
    embed_dim = 512
    num_heads = 8
    seq_len = 20
    batch_size = 32

    mha = MultiHeadAttention(embed_dim, num_heads)

    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)

    output, attn_weights = mha(query, key, value)
    print(f"Output shape: {output.shape}")  # 输出: [32, 20, 512]
    print(f"Attention Weights shape: {attn_weights.shape}")  # 输出: [32, 8, 20, 20]

def test_transformer_basic():
    # 使用示例
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    vocab_size = 10000
    dropout = 0.1
    seq_len = 90
    batch_size = 6
    # 单个d_model = embed_dim//num_heads = 512//8 = 64

    transformer = Transformer(num_layers, embed_dim, num_heads, ff_dim,
                              vocab_size, seq_len=seq_len, dropout=dropout)
    src = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    output = transformer(src, tgt)
    print(f"Output shape: {output.shape}")  # 输出: [32, 20, vocab_size]


if __name__ == '__main__':
    if 0:
        test_transformer_pe()
    if 0:
        test_transformer_attention()
    if 0:
        test_transformer_basic()
    # plt.show()
