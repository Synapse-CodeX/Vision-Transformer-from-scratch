
# 🧠 Vision Transformer (ViT) From Scratch

> A complete implementation and educational walkthrough of the **Vision Transformer (ViT)** architecture introduced in **"An Image is Worth 16×16 Words" (2020)**.

---

# 📑 Table of Contents

- Introduction
- Complete Architecture
- Input Pipeline
- Patch Embedding
- CLS Token
- Positional Embedding
- Transformer Encoder
- Multi-Head Self Attention
- MLP Block
- Residual Connections
- Layer Normalization
- Classification Head
- Complete Shape Flow
- Key Tensor Shapes
- Advantages
- Limitations

---

# 📖 Introduction

Unlike CNNs, Vision Transformers treat an image as a sequence of patches.

Instead of learning with convolution kernels, ViT learns global relationships between image patches using **Self-Attention**.

---

# 🏗️ Complete Architecture

```text
Input Image
[B,3,224,224]
        │
        ▼
Patch Embedding
[B,196,768]
        │
        ▼
CLS Token
[B,197,768]
        │
        ▼
Positional Embedding
[B,197,768]
        │
        ▼
Transformer Encoder ×12
[B,197,768]
        │
        ▼
LayerNorm
[B,197,768]
        │
        ▼
CLS Token
[B,768]
        │
        ▼
Linear Head
768 → Number of Classes
        │
        ▼
Prediction
```

---

# 🖼️ Input Pipeline

Input:

```text
[B,3,224,224]
```

Where

| Symbol | Meaning |
|--------|---------|
| B | Batch Size |
| 3 | RGB Channels |
| 224 | Height |
| 224 | Width |

---

# 🧩 Patch Embedding

Image Size

```text
224 × 224
```

Patch Size

```text
16 × 16
```

Number of patches

```text
224 / 16 = 14

14 × 14 = 196 patches
```

Patch Embedding Pipeline

```text
[B,3,224,224]
        │
        ▼
Conv2D(kernel=16,stride=16)
        │
        ▼
[B,768,14,14]
        │
        ▼
Flatten
        │
        ▼
[B,768,196]
        │
        ▼
Transpose
        │
        ▼
[B,196,768]
```

---

# 🎯 CLS Token

A learnable token is prepended to the patch sequence.

```text
Before

[B,196,768]

↓

After

[B,197,768]
```

The CLS token aggregates information from every image patch and is the only token used for classification.

---

# 📍 Positional Embedding

Transformers have no notion of spatial position.

Learnable positional embeddings are therefore added.

```text
Patch Embedding
+
Position Embedding
=
Position Aware Tokens
```

Shape

```text
[B,197,768]
```

---

# 🤖 Transformer Encoder

Each encoder block contains

```text
Input
   │
   ▼
LayerNorm
   │
   ▼
Multi-Head Attention
   │
   ▼
Residual Add
   │
   ▼
LayerNorm
   │
   ▼
MLP
768 → 3072 → 768
   │
   ▼
Residual Add
```

Shape always remains

```text
[B,197,768]
```

---

# 🔥 Multi-Head Self Attention

## QKV Projection

```text
[B,197,768]
        │
        ▼
Linear
        │
        ▼
[B,197,2304]
```

```text
2304

=

768(Q)
+
768(K)
+
768(V)
```

---

## Split into Heads

ViT-Base uses

```text
12 Heads
```

Each head receives

```text
64 Features
```

Pipeline

```text
[B,197,2304]
        │
        ▼
Reshape
        │
        ▼
[B,197,3,12,64]
        │
        ▼
Permute
        │
        ▼
[3,B,12,197,64]
```

Result

```text
Q : [B,12,197,64]

K : [B,12,197,64]

V : [B,12,197,64]
```

---

## Scaled Dot Product Attention

```text
Q @ Kᵀ

↓

[B,12,197,197]
```

Scale

```text
(QKᵀ)/√64
```

Softmax

```text
Attention Weights
```

Context Aggregation

```text
Attention

@

Value

↓

[B,12,197,64]
```

Merge Heads

```text
[B,12,197,64]

↓

[B,197,12,64]

↓

[B,197,768]
```

---

# 🧠 MLP Block

```text
768
 │
 ▼
3072
 │
 ▼
GELU
 │
 ▼
3072
 │
 ▼
768
```

Residual

```text
Output = Input + MLP(Input)
```

---

# 🔁 Residual Connections

Residual Connections allow gradients to flow directly through deep networks.

```text
Output

=

Input

+

Block(Input)
```

Used after

- Multi-Head Attention
- MLP

---

# 📏 Layer Normalization

LayerNorm stabilizes training.

Used before

- Multi-Head Attention
- MLP

and once after the final encoder stack.

---

# 🎯 Classification Head

Only the CLS token is used.

```python
cls = x[:,0]
```

Shape

```text
[B,768]
```

Classification

```text
768

↓

Linear

↓

Number of Classes
```

Output

```text
[B,num_classes]
```

---

# 📊 Complete Shape Flow

```text
[B,3,224,224]
        │
        ▼
Patch Embedding
[B,196,768]
        │
        ▼
CLS Token
[B,197,768]
        │
        ▼
Position Embedding
[B,197,768]
        │
        ▼
Encoder ×12
[B,197,768]
        │
        ▼
LayerNorm
[B,197,768]
        │
        ▼
CLS Token
[B,768]
        │
        ▼
Linear Head
[B,num_classes]
```

---

# 📋 Key Tensor Shapes

| Stage | Shape |
|------|-------|
| Input | `[B,3,224,224]` |
| Patch Embedding | `[B,196,768]` |
| + CLS | `[B,197,768]` |
| Q/K/V | `[B,12,197,64]` |
| Attention Matrix | `[B,12,197,197]` |
| Context | `[B,12,197,64]` |
| Merge Heads | `[B,197,768]` |
| Final CLS | `[B,768]` |
| Output | `[B,num_classes]` |

---

# ✅ Advantages

- Captures long-range dependencies
- Global receptive field from the first layer
- Highly scalable
- State-of-the-art performance with large datasets
- Flexible architecture

---

# ❌ Limitations

- Requires large datasets
- Computationally expensive
- Self-attention complexity grows quadratically with the number of tokens
- Less data-efficient than CNNs on small datasets

---

# 📚 Reference

**Vision Transformer**

> Dosovitskiy et al.

**An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**

https://arxiv.org/abs/2010.11929

---

⭐ If you found this project helpful, consider giving it a star!
