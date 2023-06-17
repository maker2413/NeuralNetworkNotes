#!/usr/bin/env python
# This file was generated from the code blocks in ./README.org.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
