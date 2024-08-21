# chatgpt-3k
目标: 3000行代码以内创建微型的chatgpt应用 create a mini-ChatGPT within 3000 lines of code

## 命名规则Naming Rules
- 矩阵大写 e.g. X
- 维度(dimension)以d开头

## 参考资料Referrence
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
- [60行NumPy手搓GPT[翻译]](https://jiqihumanr.github.io/2023/04/13/gpt-from-scratch/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [The Annotated Transformer code](https://github.com/harvardnlp/annotated-transformer/blob/debc9fd747bb2123160a98046ad1c2d4da44a567/the_annotated_transformer.py#L326)
- [the_annotated_transformer.py](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)
- [Top-k & Top-p](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p#2-pick-from-amongst-the-top-tokens-top-k)
- Introduce History of Attention, Neural Turing Machine: [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/#neural-turing-machines)

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
