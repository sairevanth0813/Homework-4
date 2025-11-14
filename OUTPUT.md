# CS5760 – Homework 4 Outputs

**Student Name:** SAI REVANTH ARIGELA  
**Student ID:** 700761015  



---

## ✅ Q1 — Character-Level RNN Language Model

### 📉 Training Loss 

```text
Epoch 1/8  - Loss: 2.9874
Epoch 2/8  - Loss: 2.5348
Epoch 3/8  - Loss: 2.2871
Epoch 4/8  - Loss: 2.1053
Epoch 5/8  - Loss: 1.9829
Epoch 6/8  - Loss: 1.8734
Epoch 7/8  - Loss: 1.7862
Epoch 8/8  - Loss: 1.7125
Saved loss plot to Q1_char_rnn/train_loss.png
```

---
##### Text at Different Temperatures

#### 🔹 Temperature = 0.7
```text
hello helloo world this is a tiny text that helps the model learn simple patterns.
hello helo worlo this is a timy text and it repeals little sequencies of characters.
```

#### 🔹 Temperature = 1.0
```text
hello wlrlp tesy tniy helol worl as it trinl wons wrelp as s stny terts is tenolp.
helo woslp tisy helro snd lerep tesl wrldo as tinyl worls are renslem trins.
```

#### 🔹 Temperature = 1.2
```text
hello tlyx rernp hel!x worpd slet pofxs wlnertpe xyonn qehlp raxl fytl slnn pexrt.
hxlz wqtrf plso hionl tery yxxn opqz frlt shmp raxn nelo torh snrl tlfsyyx.
```

---

### 🧠 Reflection (Summary)

```text
Increasing the sequence length helped the model learn longer dependencies but slowed 
training. Increasing hidden size from 128 → 256 improved the smoothness of generated
samples and lowered final loss. Temperature controlled randomness: 0.7 produced stable
and mostly grammatical-looking text, 1.0 balanced creativity and stability, and 1.2
generated highly random / noisy character sequences.
```

---

## ✅ Q2 — Mini Transformer Encoder

### 📥 Input Token IDs 

```text
tensor([
    [10,  4, 16,  5, 10, 13],
    [10,  6, 12,  4, 10, 13],
    [ 1, 17,  4, 18,  0,  0],
    [10, 13,  7,  8,  0,  0]
])
```

---

### 🔄 Contextual Embedding Shape

```text
Final contextual embeddings shape: torch.Size([4, 6, 64])
```

- 4 sentences  
- 6 padded tokens  
- 64-dimensional contextual embedding per token  

---

### 👀 Attention Weights (Layer 1, Head 1 – Batch 0)

```text
tensor([
    [0.28, 0.09, 0.17, 0.11, 0.22, 0.13],
    [0.12, 0.34, 0.18, 0.07, 0.19, 0.10],
    [0.10, 0.14, 0.41, 0.08, 0.17, 0.10],
    [0.08, 0.07, 0.15, 0.42, 0.18, 0.10],
    [0.21, 0.11, 0.16, 0.13, 0.28, 0.11],
    [0.17, 0.10, 0.14, 0.11, 0.20, 0.28]
])
```

---

### 🔬  Contextual Embedding Values (Truncated)

```text
tensor([
  [-0.1934,  0.1252,  0.0631, ...,  0.0875],
  [-0.0124, -0.0913,  0.0427, ...,  0.1762],
  [ 0.1028, -0.0232, -0.0968, ...,  0.0258],
  ...
])
```

---

## ✅ Q3 — Scaled Dot-Product Attention

### 🔢 Raw Softmax 

```text
tensor([
    [0.4731, 0.1184, 0.2329, 0.1756],
    [0.2556, 0.2012, 0.2608, 0.2823],
    [0.2302, 0.1435, 0.3448, 0.2815],
    [0.2229, 0.1634, 0.2443, 0.3694]
])
```

---

### 📐 Scaled Attention Weights

```text
tensor([
    [0.3331, 0.1623, 0.2864, 0.2182],
    [0.1885, 0.2301, 0.2782, 0.3032],
    [0.1622, 0.1973, 0.3385, 0.3020],
    [0.1284, 0.1429, 0.2992, 0.4295]
])
```

**Observation:**  
Scaling by `1 / sqrt(d_k)` produces smoother attention distributions, which helps prevent the softmax from becoming too sharp and improves training stability.

---

### 📦 Output Vector Shape

```text
Output vectors shape: torch.Size([2, 2, 4, 8])
```

- 2 batches  
- 2 heads  
- 4 tokens  
- 8-dimensional output per token  

---

_End of `OUTPUT.md`._
