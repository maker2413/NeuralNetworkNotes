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

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs)
print(ys)

import torch.nn.functional as F

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

xenc = F.one_hot(xs, num_classes=27).float() # input to the net: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts is equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character

nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram:
    x = xs[i].item() # input character index
    y = ys[i].item() # label character index
    print('----------')
    print(f'bigram example {i+1}: {itos[int(x)]}{itos[int(y)]} (indexes {x},{y})')
    print('input to the neural net:', x)
    print('output probabilities from the neural net:', probs[i])
    print('label (actual next character):', y)
    p = probs[i, y]
    print('probability assigned by the net to the correct character:', p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())

xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples:', num)

# randomly initialize 27 neurons' weights. Each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
# requires_grad = True is required to tell PyTorch we want gradients
W = torch.randn((27, 27), generator=g, requires_grad=True)

xenc = F.one_hot(xs, num_classes=27).float() # input to the net: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts is equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character

loss = -probs[torch.arange(num), ys].log().mean()
loss.item()

W.grad = None # set the gradient to zero
loss.backward()

W.data += -0.1 * W.grad

xenc = F.one_hot(xs, num_classes=27).float() # input to the net: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts is equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(num), ys].log().mean()
loss.item()

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
print('back propagating nn...')
for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # input to the net: one-hot encoding
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts is equivalent to N
    probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean()
    # print(loss.item())

    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad

print(loss.item())

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[int(ix)])
        if ix == 0:
            break
    print(''.join(out[:(len(out) - 1)]))
