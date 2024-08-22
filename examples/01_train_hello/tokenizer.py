import string


class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {char: idx for idx, char in enumerate(string.ascii_lowercase + ' ')}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text):
        # 忽略了未知的字符
        print(self.char_to_idx)
        return [self.char_to_idx[char] for char in text.lower() if char in self.char_to_idx]

    def decode(self, tokens):
        return ''.join([self.idx_to_char[token] for token in tokens])


class NumTokenizer:
    # [0..9], ['x', ' ']
    def __init__(self):
        # self.dictionary = sorted(set(map(str, range(10))) | set(['x', '=', ' ', '\n']))
        # '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
        self.dictionary = string.printable
        self.char_to_idx = {char: idx for idx, char in enumerate(self.dictionary)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text):
        # 忽略了未知的字符
        # print(self.char_to_idx)
        #for char in text:
        #    print(char, self.char_to_idx.get(char))
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, tokens):
        return ''.join([self.idx_to_char[token] for token in tokens])

