import torch.nn as nn
import torch
import numpy as np

softmax=nn.Softmax(dim=0)
logsoftmax=nn.LogSoftmax(dim=0)
input=torch.randn(2,1)

# input=Variable(torch.Tensor(input))

out1 = softmax(input)
out2 = logsoftmax(input)
print(torch.argmax(out1,dim=0))
print(torch.sum(torch.eq(torch.argmax(out1,dim=0),0)))
print(out1)
print(out2)
