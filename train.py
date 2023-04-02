#!/bin/python

# Get tiny shakespeare to use as a dataset
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters:", len(text))

print("\nLets look at the first 100 characters:\n", text[:100])

chars=sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stringtoint = { ch:i for i,ch in enumerate(chars) }
inttostring = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
encode = lambda string: [stringtoint[c] for c in string]
# decoder: take a list of integers, output a string
decode = lambda intlist: ''.join([inttostring[i] for i in intlist])

print(encode("Hello World!"))
print(decode([20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42, 2]))

