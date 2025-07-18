'''currently this code is inefficient since attention head is computed sequentially, in next stages we will do a matrix multiplication
do to it in parallel
'''

from causal_attention import causalAttention_V1
import torch
import torch.nn as nn

class Multiheadattentionwrapper(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):

        super().__init__()
        self.heads = nn.ModuleList(
            [causalAttention_V1(d_in,d_out,context_length,dropout,qkv_bias) for _ in range(num_heads)]
        )
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads], dim=-1) #dim=-1 because we need to stack horizontally across columns; context_length here is number of tokens
    
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
d_in = inputs.shape[1]
d_out = 2
batch = torch.stack((inputs,inputs), dim=0)
context_length = batch.shape[1]
torch.manual_seed(123)
mha = Multiheadattentionwrapper(d_in,d_out,context_length,0.0,num_heads=2)    
context_vector = mha(batch)
print(context_vector)