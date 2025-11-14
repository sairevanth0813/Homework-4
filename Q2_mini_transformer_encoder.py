# Q2_mini_transformer/mini_transformer_encoder.py
# CS5760 – Homework 4
# Question 2: Mini Transformer Encoder for Sentences
#
# This script builds a very small Transformer encoder in PyTorch, runs it on a few
# toy sentences, and prints contextual embeddings and attention weights.
#
# To run:
#   python mini_transformer_encoder.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention as in the Transformer paper.

    Q, K, V: (batch, heads, seq_len, d_k)
    mask: (batch, 1, 1, seq_len) or None
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (b, seq_len, d_model) -> (b, heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        out = self.W_o(attn_output)
        return out, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 128, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ff = FeedForward(d_model=d_model, dim_ff=dim_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + layer norm
        attn_out, attn_weights = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward + residual + layer norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class MiniTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        max_len: int = 100,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, dim_ff=dim_ff
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len) token ids
        x = self.embedding(x)
        x = self.pos_encoder(x)

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attn_maps.append(attn)
        return x, attn_maps


def simple_tokenize(sentences):
    """Very simple whitespace tokenizer.

    Returns:
      token2id, id2token, encoded_tensor
    """
    vocab = set()
    tokenized = []
    for s in sentences:
        tokens = s.strip().split()
        tokenized.append(tokens)
        vocab.update(tokens)

    vocab = sorted(list(vocab))
    token2id = {t: i + 1 for i, t in enumerate(vocab)}  # 0 = padding
    id2token = {i: t for t, i in token2id.items()}

    max_len = max(len(toks) for toks in tokenized)
    encoded = []
    for toks in tokenized:
        ids = [token2id[t] for t in toks]
        ids += [0] * (max_len - len(ids))  # pad with 0
        encoded.append(ids)

    encoded = torch.tensor(encoded, dtype=torch.long)
    return token2id, id2token, encoded


def main():
    sentences = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "a small cat slept",
        "the mat is blue",
    ]

    token2id, id2token, encoded = simple_tokenize(sentences)
    vocab_size = len(token2id) + 1  # +1 for padding index 0

    model = MiniTransformerEncoder(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_ff=128,
        max_len=encoded.size(1),
    )

    contextual_embeddings, attn_maps = model(encoded)

    print("Input token ids:")
    print(encoded)
    print("\nFinal contextual embeddings shape:", contextual_embeddings.shape)

    # Show attention map from the first layer, first head
    attn_layer0 = attn_maps[0]  # (batch, heads, seq, seq)
    head0 = attn_layer0[:, 0, :, :]  # (batch, seq, seq)

    print("\nAttention map for layer 1, head 1 (batch 0):")
    print(head0[0])


if __name__ == "__main__":
    main()
