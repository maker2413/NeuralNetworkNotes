#!/usr/bin/env python
# This file was generated from the code blocks in ./README.org.

import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}

stoi['.'] = 0

stoi['e']

itos = {i:s for s,i in stoi.items()}

itos[5]

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        # This time we want to grab the integer value of our characters
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # Add then add to the count in our 2D array for each character
        N[ix1, ix2] += 1

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
