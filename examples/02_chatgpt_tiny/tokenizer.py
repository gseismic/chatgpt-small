

# 定义一个简单的分词器
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_index = vocab.get(self.unk_token, 0)
        self.pad_index = vocab.get(self.pad_token, 1)
        self.eos_index = vocab.get(self.eos_token, 2)
    
    def encode(self, texts, max_len):
        input_ids = []
        for text in texts:
            tokens = text.split()  # 简单地用空格分词
            ids = [self.vocab.get(token, self.unk_index) for token in tokens]
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids += [self.pad_index] * (max_len - len(ids))
            input_ids.append(ids)
        return torch.tensor(input_ids)

