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
