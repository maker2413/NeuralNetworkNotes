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

# This block of code will print ever character pair and the number of times it
# occurs. It will also shade each tile dark the more a pair appears.
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')

p = N[0].float()
p = p / p.sum()

p

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

itos[ix]

P = N.float()
P = P / P.sum(1, keepdim=True)

print("Results without neural network:")

g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    out = []
    ix = 0
    # Same while loop as before with probabilities all flatten
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

P = (N+1).float()
P = P / P.sum(1, keepdim=True)

log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
loss = nll/n
print("Current loss:", loss)

xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
