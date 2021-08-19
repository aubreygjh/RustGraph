import torch
import torch.nn as nn
import torch.nn.functional as F

i1 = torch.randn(5,8,2)
i2 = torch.randn(2)
print(i1)
print(i2)
for i in range(5):
    total = []
    print(F.cosine_similarity(i1[i],i2,dim=-1))
    for j in range(8):
        total.append(F.cosine_similarity(i1[i][j],i2,dim = 0))
    print(total)
