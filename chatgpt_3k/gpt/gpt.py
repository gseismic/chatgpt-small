from ..config import nn
from ..transformer.positional_encoding import PositionalEncoding
from ..transformer.positionwise_feed_forward import PositionwiseFeedForward
from ..transformer.multi_head_attention import MultiHeadAttention


class GPTBlock(nn.Module):
    '''
    Note:
        GPT 只有解码部分，而 Transformer 的原始论文中包括编码器和解码器部分。
        这里的关键点在于 GPT 的设计目标和使用场景与原始 Transformer 模型的不同。让我们详细解释一下原因：

        GPT 设计目标
        自回归语言模型：

        GPT 的主要目标是生成语言（文本），这需要模型能够生成每个单词（或字符）时，基于已经生成的内容来预测下一个单词。
        这种生成模式要求模型在生成每个词时只能看到之前的词，因此只使用解码器结构中的自注意力机制来确保每个词只依赖于其左侧的词。
        单向自注意力：

        GPT 使用单向自注意力（自回归），即在计算每个位置的注意力时，只考虑当前位置之前的所有位置。这样可以确保生成文本的连贯性。
        原始 Transformer 的解码器部分支持这种自回归模式，通过掩蔽机制（masking）来实现。
    '''
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(self_attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x


class GPT(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, seq_len, dropout=0.1):
        super(GPT, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_len)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    @classmethod
    def from_config(cls, gpt_config):
        if isinstance(gpt_config, str):
            if gpt_config.endswith('.json'):
                import json
                with open(gpt_config) as f:
                    config = json.load(f)
            else:
                raise ValueError(f'Invalid Config File: `{gpt_config}`')
        elif isinstance(gpt_config, dict):
            config = gpt_config
        else:
            raise ValueError(f'Invalid Config: `{gpt_config}`')
        return GPT(**config)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) + self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.layer_norm(x)
        output = self.fc_out(x)
        return output
