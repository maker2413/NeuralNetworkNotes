#!/usr/bin/env python
import torch

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

print(sorted(b.items(), key = lambda kv: -kv[1]))
