import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=128, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.size()
        token = self.token_emb(x)         # (B, T, C)
        position = self.pos_emb[:, :T, :] # (1, T, C)
        x = token + position
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
