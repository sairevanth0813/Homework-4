# Q3_attention/scaled_dot_product_attention.py
# CS5760 – Homework 4
# Question 3: Implement Scaled Dot-Product Attention
#
# This file implements the scaled dot-product attention function and
# tests it on random Q, K, V tensors, printing attention weights and outputs.

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Args:
        Q, K, V: Tensors of shape (batch, heads, seq_len, d_k)
        mask: Optional mask of shape (batch, 1, 1, seq_len) with 0s for positions to mask.

    Returns:
        output: (batch, heads, seq_len, d_k)
        attn:   (batch, heads, seq_len, seq_len) attention weights
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn


def test_attention():
    batch = 2
    heads = 2
    seq_len = 4
    d_k = 8

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_k)

    # Softmax without scaling (for comparison)
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))
    raw_softmax = F.softmax(raw_scores, dim=-1)

    # With scaling
    output, attn = scaled_dot_product_attention(Q, K, V, mask=None)

    print("Raw (unscaled) softmax for batch 0, head 0:")
    print(raw_softmax[0, 0])

    print("\nScaled attention weights for batch 0, head 0:")
    print(attn[0, 0])

    print("\nOutput vectors shape:", output.shape)


if __name__ == "__main__":
    test_attention()
