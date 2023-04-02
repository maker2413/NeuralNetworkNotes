#!/usr/bin/env python

# Get tiny shakespeare to use as a dataset
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters:", len(text))

print("\nLets look at the first 100 characters:\n", text[:1000])

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

import torch # https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

print(xb)
