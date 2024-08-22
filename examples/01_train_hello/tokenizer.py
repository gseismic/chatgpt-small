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

