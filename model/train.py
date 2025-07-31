import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from utils.dataset import LMDataset
from model.gpt_model import MiniGPT

def train():
    block_size = 64
    batch_size = 32
    num_epochs = 10
    vocab_size = 8000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
    with open("data/tiny_corpus.txt", "r") as f:
        tokens = tokenizer.encode(f.read()).ids

    dataset = LMDataset(tokens, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MiniGPT(vocab_size, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), f"model_epoch{epoch+1}.pt")
