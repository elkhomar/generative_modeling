from torch import nn
import torch.nn.functional as F

import torch
a = 20*[1] + [1-i/20 for i in range(20)] + 140*[0]
b = torch.bernoulli(torch.Tensor(a[0]))
print(b)