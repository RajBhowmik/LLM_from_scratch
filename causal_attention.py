import torch
import torch.nn as nn

class causalAttention_V1(nn.Module):
    def __init__(self, d_in,d_out, context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out= d_out
        self.W_query = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout= nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        '''
        why transpose(1,2)?
        
        keys.T is a convenient shorthand only when the tensor is 2-D (matrix).
        With anything that has more than two dimensions PyTorch interprets .T as “reverse the entire order of the dimensions.”

        So transpose(1, 2) is the minimal swap we need:

        keep batch dimension where it is;

        put feature (d_out) as the inner dimension for the dot-product;

        leave token count (N) as rows/columns to end up with an (N × N) attention matrix.
        >>> keys.shape                 # (B, N, d)
            torch.Size([2, 6, 2])

            >>> keys.T.shape               # (d, N, B)   ❶ reversed order!
            torch.Size([2, 6, 2])

            >>> keys.transpose(1, 2).shape # (B, d, N)   ❷ only swap the last two dims
            torch.Size([2, 2, 6])
        
        '''
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens],-torch.inf # num_tokens to take care of smaller length
        )
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values

        return context_vector

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
ca = causalAttention_V1(d_in,d_out,context_length,0.0)    
context_vector = ca(batch)
print(context_vector)