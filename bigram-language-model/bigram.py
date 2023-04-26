#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt

# Let's start by putting all of the words in our names.txt in a variable
words = open('names.txt', 'r').read().splitlines()

# b will be our bigram dictionary
b = {}
# Let's iterate over all of the words we have
for w in words:
    # And let's add an imaginary start and end character to each of our words
    chs = ['<S>'] + list(w) + ['<E>']
    # Then we will iterate over each 2 character chunks of each word
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        # And add a count of each occurance of a 2 character pair to our
        # dictionary
        b[bigram] = b.get(bigram, 0) + 1

# From here we could sort our dictionary by most used character pairs like this
sorted(b.items(), key = lambda kv: -kv[1])

# But let's now store our character pairs in a pytorch 2D array
N = torch.zeros((27, 27), dtype=torch.int32)

# We have to store our characters as integers in our 2D array
# The following line will give us a sorted list (a-z) of all the unique
# characters used in our names.txt
chars = sorted(list(set(''.join(words))))
# Now we will create a look up table (string to int)
stoi = {s:i+1 for i,s in enumerate(chars)}
# We then can define a number for a custom character to denote the beginning
# and end of a word instead of having two special characters this time
stoi['.'] = 0

# This is our same for loops from above
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        # This time we want to grab the integer value of our characters
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # Add then add to the count in our 2D array for each character
        N[ix1, ix2] += 1

# Printing out our array is quite a bit of information and isn't really valuable
# Using matplot lib we can print out our array in a much more useful format
# Let's start by making an inverse mapping of our stoi table (int to string)
itos = {i:s for s,i in stoi.items()}

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
plt.savefig('probabilities.png')

# Now we can actually use the appearance count of each of these pairs to build
# a probability for each character pair
p = N[0].float()
p = p / p.sum()
# print(p)

# https://pytorch.org/docs/stable/generated/torch.Generator.html
# https://pytorch.org/docs/stable/generated/torch.multinomial.html
# If we initialize our generator to a manually set seed we can get the same
# results each time during development
# We can then use PyTorch's multinomial function to pull data our of our 2D
# array using these probablities
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# print(itos[ix])

# Now that we have proved that we can pull characters out we can loop through
# our array to pull out a series of characters
# We can make our loop a little more efficient by making a matrix for our
# probabilities
# https://pytorch.org/docs/stable/generated/torch.sum.html
# https://pytorch.org/docs/stable/notes/broadcasting.html
# We can add to N before calculating our probabilities to perform
# data smoothing: https://openreview.net/forum?id=H1VyHY9gg
P = (N+1).float()
P /= P.sum(1, keepdim=True)

# This for loop will give us 10 generated "names", however you will notice
# that most of these names are terrible. This is a limitation of a bigram
# language model, however this is better than if we had our loop grab values
# at absolute random for our 2D array.
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[int(ix)]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[int(ix)])
        if ix == 0:
            break

    print(''.join(out[:(len(out) - 1)]))

# From this point it would be common to evaluate the quality of this model.
log_likelihood = 0.0
n = 0

# Let's do that by printing the probability of each character pair in the first
# 3 words in our file and let's keep count of the log of our probability so that
# we can use that later to calculate the loss of our training like we did for
# micrograd
# https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

nll = -log_likelihood
#print(f'{nll=}')
# Loss
# The closer our loss is to 0 the better our model is given the data it was
# provided
#print(f'{nll/n}')

# From here we want to fine tune our Neural Network to reduce our loss. To
# do this we have the following:
# GOAL: maximize likelihood of the data w,r,t, model parameters (statistical
# modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# To do this let's first create the training set of bigrams (x,y):
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# Take note that tensor and Tensor are two separate things in PyTorch:
# https://pytorch.org/docs/stable/tensors.html#torch.Tensor
# https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor
# torch.tensor infers the dtype automatically, while torch.Tensor returns
# torch.FloatTensor: https://stackoverflow.com/a/63116398
xs = torch.tensor(xs)
ys = torch.tensor(ys)

# https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
import torch.nn.functional as F
# one_hot keeps the datatype and we want floats in xenc so we cast to float
xenc = F.one_hot(xs, num_classes=27).float()

# https://pytorch.org/docs/stable/generated/torch.randn.html
# We can feed in our generator so that we get the same results during testing
W = torch.randn((27, 27))
# @ is a matrix multiplication operator in pytorch
# This will in parallel evaulate all of the 27 neurons on the 5 inputs
logits = xenc @ W # logits = log counts
# (5, 27) @ (27, 27) -> (5, 27)
counts = logits.exp() # counts equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
# the last 2 lines here are together called a 'softmax'
# https://en.wikipedia.org/wiki/Softmax_function
# https://towardsdatascience.com/softmax-activation-function-explained-a7e1bc3ad60

# Since that was quite complicated let's break this down a bit
# Summary:
# randomly initialize 27 neurons' weights. Each neuron receives 27 inputs
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

# We see that we have a loss of 3.769304... on our current seed. Before we move
# away from our set seed lets optimize our neural network to reduce our loss.
# create the dataset
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

# We will optimize or network with a similar forward and backward pass like we
# did micrograd:
# randomly initialize 27 neurons' weights. Each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
# requires_grad = True is required to tell PyTorch we want gradients
W = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
print('back propagating nn...')
for k in range(100):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # input to the net: one-hot encoding
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts is equivalent to N
    probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # https://pytorch.org/docs/stable/generated/torch.arange.html
    loss = -probs[torch.arange(num), ys].log().mean()
    #print(loss.item())

    # backward pass
    # We first want to reset our gradients. We could set our gradients to 0, but we
    # can also set our gradient to none which will be interpreted as no gradient has
    # been set.
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad # type: ignore

# This does end up giving us about the same loss as we had before we started
# fine tuning our neural network but the gradient based approach is much more
# flexible. Although it is easy for us to calculate loss in a bigram data model
# and our, all things consider, small dataset of names.txt. This same approach
# of running our data through a softmax, back propagating, and tweaking our
# gradients will scale up all the way through transformer data models which
# would significantly harder to predict with a probaility matrix like we can
# with a bigram data model.

# Finally let's sample for our new neural net model
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

# We actually got the almost same results! Which is either lame or really cool
# depending on how you look at it.
