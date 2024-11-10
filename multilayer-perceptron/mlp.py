#!/usr/bin/env python
# This file was generated from the code blocks in ./README.org.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one
X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 2))

emb = C[X]
emb.shape

W1 = torch.randn((6, 100))
b1 = torch.randn(100)

h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

W2 = torch.randn((100, 27))
b2 = torch.randn(27)

logits = h @ W2 + b2

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)

loss = -prob[torch.arange(32), Y].log().mean()
loss

# ========================= now made respectable =========================
