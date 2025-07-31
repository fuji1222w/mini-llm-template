import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.block_size = block_size
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx+self.block_size]
        y = self.token_ids[idx+1:idx+self.block_size+1]
        return torch.tensor(x), torch.tensor(y)
