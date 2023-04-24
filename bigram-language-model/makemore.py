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
stoi['<.>'] = 0

# This is our same for loops from above
for w in words:
    chs = ['<.>'] + list(w) + ['<.>']
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
plt.savefig('2darray.png')

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
g = torch.Generator().manual_seed(2147483647)

# We can make our loop a little more efficient by making a matrix for our
# probabilities
# https://pytorch.org/docs/stable/generated/torch.sum.html
# https://pytorch.org/docs/stable/notes/broadcasting.html
P = N.float()
P /= P.sum(1, keepdim=True)

# This for loop will give us 10 generated "names", however you will notice
# that most of these names are terrible. This is a limitation of a bigram
# language model, however this is better than if we had our loop grab values
# at absolute random for our 2D array.
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out[:(len(out) - 1)]))
