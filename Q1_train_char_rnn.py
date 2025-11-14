# Q1_char_rnn/train_char_rnn.py
# CS5760 – Homework 4
# Question 1: Character-Level RNN Language Model
#
# This script trains a small character-level RNN (LSTM) on a text corpus and
# samples text at different temperatures.
#
# To run:
#   python train_char_rnn.py
#
# Make sure you have PyTorch and matplotlib installed:
#   pip install torch matplotlib

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CharDataset(Dataset):
    """Character-level dataset producing (input_seq, target_seq) pairs."""

    def __init__(self, text: str, seq_len: int = 50):
        self.seq_len = seq_len
        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        # Encode entire corpus as indices
        self.encoded = np.array([self.char2idx[c] for c in text], dtype=np.int64)

    def __len__(self):
        # Last possible start index is len-1-seq_len
        return len(self.encoded) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.encoded[idx : idx + self.seq_len]
        y = self.encoded[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class CharRNN(nn.Module):
    """Embedding -> LSTM -> Linear -> (used with CrossEntropyLoss)."""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        x = self.embedding(x)              # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(x, hidden) # (batch, seq_len, hidden_size)
        logits = self.fc(out)              # (batch, seq_len, vocab_size)
        return logits, hidden


def load_text_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@torch.no_grad()
def sample(model: CharRNN,
           dataset: CharDataset,
           start_seq: str = "hello ",
           length: int = 300,
           temperature: float = 1.0) -> str:
    """Temperature-controlled sampling from a trained model."""
    model.eval()
    generated = start_seq

    # Encode the starting prompt
    input_indices = [dataset.char2idx.get(c, 0) for c in start_seq]
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=DEVICE)

    hidden = None
    # Warm up the hidden state with the prompt
    for i in range(input_tensor.size(1) - 1):
        _, hidden = model(input_tensor[:, i:i+1], hidden)

    current_idx = input_tensor[:, -1]  # last char index from prompt

    for _ in range(length):
        logits, hidden = model(current_idx.unsqueeze(1), hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_char = dataset.idx2char[int(next_idx[0])]
        generated += next_char
        current_idx = next_idx

    return generated


def main():
    # Hyperparameters (you can change these for experiments)
    seq_len = 50
    batch_size = 64
    num_epochs = 8
    embed_dim = 128
    hidden_size = 256
    lr = 1e-3

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "data", "tiny_corpus.txt")
    text = load_text_corpus(data_path)
    print(f"Loaded corpus with {len(text)} characters.")

    dataset = CharDataset(text, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = len(dataset.chars)
    print(f"Vocab size: {vocab_size}")

    model = CharRNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=1,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            # logits: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            loss = criterion(
                logits.view(-1, vocab_size),
                batch_y.view(-1)
            )
            loss.backward()

            # Gradient clipping (helps with exploding gradients)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Plot and save loss curve
    plt.figure()
    plt.plot(train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Char-RNN Training Loss")
    out_path = os.path.join(os.path.dirname(__file__), "train_loss.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss plot to {out_path}")

    # Sample generations at different temperatures
    for temp in [0.7, 1.0, 1.2]:
        print(f"\n=== Sampling with temperature = {temp} ===")
        text_out = sample(model, dataset, start_seq="hello ", length=300, temperature=temp)
        print(text_out)


if __name__ == "__main__":
    main()
