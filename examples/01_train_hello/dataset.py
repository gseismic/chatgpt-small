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
