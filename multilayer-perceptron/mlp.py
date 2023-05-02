#!/usr/bin/env python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

# read in all the words
words = open('names.txt', 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
# The following code allows us to look at 3 character blocks of our provided
# text and predict the 4th character. We also are padding our data with dots in
# the event that we are at the start of the word.
block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words:
    # print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

# Now that we have built the dataset we are going to store it in a 2
# dimensional matrix.
# C = torch.randn((27, 2))

# We can then embed all of our integers of X with:
# emb = C[X]

# Declare our weights and biases
# W1 = torch.randn((6, 100))
# b1 = torch.randn(100)

# https://pytorch.org/docs/stable/generated/torch.cat.html
# emb @ W1 + b1
# torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1).shape
# https://pytorch.org/docs/stable/generated/torch.unbind.html
# torch.cat(torch.unbind(emb, 1), 1).shape
# http://blog.ezyang.com/2019/05/pytorch-internals/
# h = emb.view(-1, 6) @ W1 + b1
# h = torch.tanh(h)

# Then we declare our output layer of neurons
# W2 = torch.randn((100, 27))
# b2 = torch.randn(27)

# logits = h @ W2 + b2

# Just like before we declare our counts as an exp of our logits and then use
# that to calculate our prob
# counts = logits.exp()
# prob = counts/ counts.sum(1, keepdim=True)

# We can then calculate our loss
# loss = -prob[torch.arange(32), Y].log().mean()

# Now lets rewrite this with a set generator seed and make it a bit more
# presentable.
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

print(sum(p.nelement() for p in parameters)) # number of total parameters

for p in parameters:
    p.requires_grad = True

# https://pytorch.org/docs/stable/generated/torch.linspace.html
# linspace can be used to give us a range of numbers that we can use for our
# learning rate
# lre = learning rate exponent
lre = torch.linspace(-3, 0, 1000)
# lrs = learning rates
lrs = 10**lre
# print(lrs)

# We will use these to keep track of the learning rates we have used and the
# losses that resulted.
lri = []
lossi = []

# To optimize our back propagation since our dataset is much larger than
# what we have previously worked with. It is common to back propagate random
# batches of our data.
for i in range(100000):
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32, ))

    # forward pass
    emb = C[X[ix]] # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
    # Using PyTorch's cross_entropy function is much more efficient than calculate
    # counts, prob, and loss individually each pass.
    loss = F.cross_entropy(logits, Y[ix])
    # print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad # type: ignore

    # track stats
    # lri.append(lre[i])
    # lossi.append(loss.item())

# We can then plot our learning rates for analysis
# plt.plot(lri, lossi)
# plt.savefig('learningrate.png')

emb = C[X] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Y)

print(loss.item())
